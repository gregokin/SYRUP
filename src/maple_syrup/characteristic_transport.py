"""Characteristic mobile-sediment transport with bounded phase bins (Phase 7b candidate kernel).

Candidate replacement for the lateral operator `T` of the Phase 1 contract
(docs/phase1/interface_contract.md section 4.1). NOT yet wired into the
event, checkpoint or completion drivers: `sediment_transport.transport_step`
remains the production operator until the event integration is reviewed.
Same `TransportStep` contract, same MAPLE face layout, same request
semantics; this kernel adds a SYRUP-owned sub-cell position state.

Why (docs/phase7b/characteristic_design.md)
--------------------------------------------
The Phase 5 operator is a single well-mixed pool per cell and class, so a
parcel's residence in a cell is exponential and mass that enters a cell
leaves it with probability `L / (L + dx)` instead of the physical
`exp(-dx / L)` of the legacy distance convention (`flow_distrib.for` 44-46,
102-116). With `dx = 0.5 m` and `L` of centimetres that is orders of
magnitude too much transmission (Phase 7 probe, docs/phase7/acceptance.md).

Model (exact within a step of piecewise-frozen `v`, `r = 1/L`, settle)
-----------------------------------------------------------------------
Mobile mass in a cell is a set of PACKETS with a distance `x` in `[0, dx)`
from the cell's upstream face along its D4 outflow. Actual MAPLE pickup is
injected as a new packet at `x = 0` (the legacy upstream-face distance
origin of `flow_distrib.for`; Codex selected this convention to follow the
user's preference for MAHLERAN-like physics, the user did not themselves
specify the coordinate convention). Over a
substep `dt` a packet at `x` moves along its characteristic:

    reaches the downstream face after tau = (dx - x) / v   (if v dt >= dx - x)
    survival over a travelled distance s is exp(-r s); the lost part
    deposits in the cell being traversed (memoryless hazard 1 / L per metre)

If it crosses (Courant `v dt / dx <= 0.5` per substep, so at most ONE
face per packet per substep): the surviving mass is an exact face crossing;
at an outlet it is an export request; a dry / no-capacity receiver settles
it through the returned deposition request; otherwise it continues for the
remaining time `dt - tau` at the RECEIVER's `v` and `r`, deposits there,
and starts a new packet at `x = v_receiver (dt - tau)`. Nothing is
deposited ahead of the mass (no instantaneous remote deposit).

Bounded state: packets landing in the same cell/class are merged into `B`
position bins of width `dx / B`; every bin stores its mass (as a FRACTION
of the authoritative MAPLE mobile total of the cell/class) and ONE
representative distance

    x_rep = L log( sum_i m_i exp(x_i / L) / sum_i m_i )     (r > 0)
    x_rep = sum_i m_i x_i / sum_i m_i                        (r = 0)

computed as a shifted log-sum-exp: shifted by the largest position with
`expm1`/`log1p` when `r (xmax - xmin) <= NARROW_SPREAD` (weights cannot
cancel, accurate as `r -> 0`), otherwise shifted by the largest log-weight
`log m + r x` as a positive sum (a heavy constituent far below a light one
at `xmax` would otherwise round the subtractive sum to `-M`). The logarithmic
mean preserves `sum_i m_i exp(-(dx - x_i) / L)` EXACTLY, i.e. the mass the
bin will eventually push across its face under the cell's current `L`,
and lies within the constituents' position range (only round-off
containment is applied, never mass clipping). What it approximates is the
TIMING of the merged mass and its fate when `L` or `v` change later; the
approximation shrinks with `B` (see the continuous-source tests).

Contract (per call)
-------------------
Inputs: the network, the PRE-pickup mobile mass `M`, the phase state
(fractions and positions), the ACTUAL pickup `P` (MAPLE ground truth),
`v`, `r`, `settle`, `dt`, `n_substeps`, `implementation`. Component masses
are `M f`; the pickup packet is appended (not merged with older mass before
it has travelled) in the FIRST substep only. Returned:
`TransportStep.mobile_after_transfer_kg = T = M_remaining + Dep + E`, with
`mobile_before = M + P` for the identity `T - (M + P) = In - Out`; requests
`Dep`, `E` stay in the pool of their cell until MAPLE applies them;
`phase_after` describes `M_remaining` (the pool AFTER the requests). MAPLE
remains the only mass authority; fractions are a distribution, never a
second inventory.

Conservation and validation: every operation is a product by a factor in
`[0, 1]` or a sum of such products, so `W >= 0`, `Dep + E <= T` per cell,
`sum T = sum (M + P)` per class within a declared FP64 bound built from the
actual MAPLE `summation_error_bound_kg` with an explicit operation count
(no physical mass floor); `T - (M + P) = In - Out` per cell within the same
policy; fraction sums within `summation_error_bound_kg(B + 1, 1)`;
positions in `[0, dx)` and inside their bin. Inputs are validated
(shape, dtype, namespace, finiteness, sign, inactive cells, phase
invariants, Courant) with one batched flag read before any result exists;
a Courant violation raises the recoverable `TransportStepRejected`. No
caller-owned array is modified. `v = 0` is a valid no-motion step; `r = 0`
is a valid no-deposition law.

Backends: `implementation="array"` is namespace-generic (NumPy or CuPy):
one candidate slot per (cell, class, bin + pickup), `O(n nc (B + 1))`
memory, destination keys, MAPLE `scatter_add` / `scatter_max` /
`scatter_min` for the merge (deterministic sorted path on CuPy), no Python
loop over cells; `implementation="numba"` is the compiled host kernel of
`characteristic_numba.py` mirroring the same passes and expressions (no
silent fallback in either direction).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import ModuleType
from typing import Any

import numpy as np
from maple.surface.voxels._numerics import summation_error_bound_kg

from maple_syrup.sediment_transport import (
    TransportError,
    TransportNetwork,
    TransportStep,
    TransportStepRejected,
)

__all__ = [
    "DEFAULT_COURANT_MAX",
    "DEFAULT_N_BINS",
    "IMPLEMENTATIONS",
    "MAX_N_BINS",
    "MAX_SUBSTEPS",
    "NARROW_SPREAD",
    "CharacteristicStep",
    "PhaseState",
    "bin_index",
    "characteristic_step",
    "empty_phase_state",
    "fraction_sum_tolerance",
    "operation_count",
    "validate_phase_state",
]

_EPS = float(np.finfo(np.float64).eps)
IMPLEMENTATIONS = ("array", "numba")
DEFAULT_N_BINS = 32  # candidate default; 8..128 supported for refinement studies, none accepted yet
MAX_N_BINS = 128
DEFAULT_COURANT_MAX = 0.5  # at most one face crossing per packet per substep
MAX_SUBSTEPS = 1_000_000
_COURANT_REJECTION = "characteristic sediment Courant number v dt / (n_substeps dx) exceeds"
# Sentinels for the masked extrema (MAPLE scatter extrema require finite
# values): below every valid position, and below every log-weight of a
# positive FP64 mass (log(5e-324) = -744) at a position >= 0.
_BELOW_ANY_POSITION = -1.0
_BELOW_ANY_LOG_WEIGHT = -1.0e300
# Logarithmic-mean branch selection: with r (xmax - xmin) <= NARROW_SPREAD the
# shifted-by-xmax expm1 form has weights in [expm1(-NARROW_SPREAD), 0] and
# cannot cancel; beyond it the log-weight-shifted positive sum is used.
NARROW_SPREAD = 0.5


def _real(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TransportError(f"{name} must be a real number, got {type(value).__name__}")
    return float(value)


def _strict_int(value: Any, name: str, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not (low <= int(value) <= high):
        raise TransportError(f"{name} must be an int in [{low}, {high}], got {value!r}")
    return int(value)


def bin_index(position_m: Any, dx_m: float, n_bins: int, xp: ModuleType) -> Any:
    """Bin of a position: `min(floor(x / dx * B), B - 1)` as int64. The ONE
    expression used for state validation, merging and the compiled kernel
    (`characteristic_numba._bin`), so a representative that lies between
    two constituents of a bin maps to that same bin (floor is monotone)."""
    return xp.minimum(xp.floor(position_m / dx_m * n_bins), n_bins - 1).astype(np.int64)


def fraction_sum_tolerance(n_bins: int) -> float:
    """`|sum_b f_b - 1|` allowance: MAPLE's summation policy for `B`
    quotients and their sum, at unit scale."""
    return summation_error_bound_kg(int(n_bins) + 1, 1.0)


def operation_count(n_substeps: int, n_bins: int) -> int:
    """Roundings per cell/class element feeding the mass identities: per
    substep every one of the `B + 1` slots is formed (product), decayed
    (exp / expm1 and product), assigned and summed into a cell total, its
    crossing is scattered into a receiver total and the arrival deposit is
    formed and scattered (about twelve per slot), plus the endpoint
    combinations. A declared count for `summation_error_bound_kg`, not a
    physical floor."""
    return int(n_substeps) * 12 * (int(n_bins) + 1) + 8


# --- phase state --------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class PhaseState:
    """Sub-cell position distribution of the mobile mass, SYRUP-owned.

    `fraction` `(ny, nx, nc, B)` FP64: share of the cell/class MAPLE mobile
    total in each position bin (sum 1 where the total is positive);
    `position_m` `(ny, nx, nc, B)` FP64: representative distance of each
    bin from the cell's upstream face, in `[0, dx)`, inside its bin, 0 on
    empty bins. Canonical empty field: `fraction[..., 0] = 1`, all other
    fractions 0, all positions 0 (also required wherever the mobile total
    is 0). Both arrays are frozen where the backend can enforce it."""

    fraction: Any
    position_m: Any
    dx_m: float
    n_bins: int
    xp: ModuleType

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.fraction.shape[0]), int(self.fraction.shape[1]))

    @property
    def n_classes(self) -> int:
        return int(self.fraction.shape[2])


def empty_phase_state(shape: tuple[int, int], n_classes: int, n_bins: int, dx_m: float, *,
                      xp: ModuleType | None = None) -> PhaseState:
    """Canonical empty state (no mobile mass anywhere)."""
    from maple.core.backend import freeze

    namespace = np if xp is None else xp
    ny, nx = int(shape[0]), int(shape[1])
    nc = _strict_int(n_classes, "n_classes", 1, 1 << 30)
    nb = _strict_int(n_bins, "n_bins", 1, MAX_N_BINS)
    dx = _real(dx_m, "dx_m")
    if not (math.isfinite(dx) and dx > 0.0):
        raise TransportError(f"dx_m must be finite and > 0, got {dx_m!r}")
    fraction = namespace.zeros((ny, nx, nc, nb), dtype=np.float64)
    fraction[..., 0] = 1.0
    position = namespace.zeros((ny, nx, nc, nb), dtype=np.float64)
    return PhaseState(fraction=freeze(fraction), position_m=freeze(position), dx_m=dx, n_bins=nb, xp=namespace)


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


def _phase_checks(phase: PhaseState, mobile_kg: Any, network: TransportNetwork, checks: Any) -> None:
    """Enqueue the phase invariants (module docstring) on `checks`; the
    structural questions (types, shapes) raise immediately."""
    from maple.core.backend import finite_flag, negative_flag, true_flag

    if not isinstance(phase, PhaseState):
        raise TransportError("phase must be a PhaseState (use empty_phase_state)")
    xp = network.xp
    if phase.xp is not xp:
        raise TransportError(f"phase state namespace {phase.xp.__name__!r} differs from the network's {xp.__name__!r}")
    nb = _strict_int(phase.n_bins, "phase.n_bins", 1, MAX_N_BINS)
    if _real(phase.dx_m, "phase.dx_m") != network.dx_m:
        raise TransportError(f"phase dx_m {phase.dx_m} differs from the network dx_m {network.dx_m}")
    ny, nx = network.shape
    nc = int(mobile_kg.shape[-1])
    _require({"phase.fraction": phase.fraction, "phase.position_m": phase.position_m}, (ny, nx, nc, nb), xp,
             np.float64)
    f, x = phase.fraction, phase.position_m
    dx = network.dx_m
    checks.require(finite_flag(f), "phase.fraction must be finite everywhere")
    checks.require(finite_flag(x), "phase.position_m must be finite everywhere")
    checks.forbid(negative_flag(f), "phase.fraction must be >= 0 everywhere")
    checks.forbid(negative_flag(x), "phase.position_m must be >= 0 everywhere")
    checks.forbid(true_flag(x >= dx), "phase.position_m must be < dx everywhere")
    total = mobile_kg[..., None]
    occupied = total > 0.0
    fsum = f.sum(axis=-1)[..., None]
    checks.forbid(true_flag(occupied & (xp.abs(fsum - 1.0) > fraction_sum_tolerance(nb))),
                  "phase.fraction must sum to 1 within the declared bound where the mobile total is positive")
    canonical = xp.zeros((1, 1, 1, nb), dtype=np.float64)
    canonical[..., 0] = 1.0
    checks.forbid(true_flag(~occupied & ((f != canonical) | (x != 0.0))),
                  "phase state must be canonical (fraction e_0, position 0) where the mobile total is 0")
    checks.forbid(true_flag((f == 0.0) & (x != 0.0)), "phase.position_m must be 0 on empty bins")
    bins = xp.arange(nb, dtype=np.int64).reshape(1, 1, 1, nb)
    checks.forbid(true_flag((f > 0.0) & (bin_index(x, dx, nb, xp) != bins)),
                  "phase.position_m must lie inside its own bin")


def validate_phase_state(phase: PhaseState, mobile_kg: Any, network: TransportNetwork) -> None:
    """Raise `TransportError` unless `mobile_kg` is a valid `(ny, nx, nc)`
    FP64 mobile field (finite, >= 0, 0 on inactive cells) and `phase` a
    valid partition of it on `network` (one batched flag read)."""
    from maple.core.backend import (
        DeferredChecks,
        finite_flag,
        is_array,
        negative_flag,
        true_flag,
    )

    if not isinstance(network, TransportNetwork):
        raise TransportError("network must be a TransportNetwork (use transport_network)")
    xp = network.xp
    if not is_array(mobile_kg) or len(mobile_kg.shape) != 3 or mobile_kg.shape[-1] < 1:
        raise TransportError("mobile_kg must be a (ny, nx, nc) array")
    ny, nx = network.shape
    _require({"mobile_kg": mobile_kg}, (ny, nx, int(mobile_kg.shape[-1])), xp, np.float64)
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


# --- result -----------------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class CharacteristicStep:
    """`transport` is the unchanged `TransportStep` contract (`T`, requests,
    faces, budgets; `mobile_before_by_class_kg` is `sum (M + P)`).
    `phase_after` partitions `mobile_remaining_kg = T - Dep - E`, the pool
    after MAPLE applies the requests. `mobile_before_kg = M + P` per cell;
    `arrival_deposition_kg` is the part of the deposition request booked in
    a cell by mass that entered it during the step (hazard plus dry
    settling); `crossing_kg = Out + E` per source cell."""

    transport: TransportStep
    phase_after: PhaseState
    mobile_before_kg: Any
    mobile_remaining_kg: Any
    arrival_deposition_kg: Any
    crossing_kg: Any
    n_bins: int
    implementation: str


# --- array substep ---------------------------------------------------------------------------
def _substep_array(xp: ModuleType, network: TransportNetwork, W: Any, X: Any, P: Any, V: Any, R: Any, S: Any,
                   dx: float, dt: float, nb: int) -> tuple:
    """One characteristic substep on masses `W (n, nc, B)`, positions `X`,
    pickup packet `P (n, nc)` at `x = 0`. Returns new masses/positions and
    the per-cell/class bookkeeping (see `characteristic_step`)."""
    from maple.core.backend import scatter_add, scatter_max, scatter_min

    n, nc = V.shape
    zeros_slot = xp.zeros((n, nc, 1), dtype=np.float64)
    Wc = xp.concatenate([W, P[..., None]], axis=-1)  # (n, nc, B + 1): existing bins + pickup packet
    Xc = xp.concatenate([X, zeros_slot], axis=-1)
    S3 = S[..., None]
    V3 = V[..., None]
    R3 = R[..., None]
    travel = V3 * dt
    # Crossing test and the source travel use the SAME expressions, so a
    # non-crossing packet's new position `x + travel` is < dx exactly.
    cross = (Xc + travel >= dx) & ~S3
    travel_src = xp.where(cross, dx - Xc, travel)
    decay_src = -xp.expm1(-R3 * travel_src)
    lost = xp.where(S3, 0.0, Wc * decay_src)
    survivor = xp.where(S3, 0.0, Wc * xp.exp(-R3 * travel_src))
    settled_src = xp.where(S3, Wc, 0.0)
    dep_decay = lost.sum(axis=-1)
    settled = settled_src.sum(axis=-1)

    outlet3 = network.outlet_flat[:, None, None]
    crossing = xp.where(cross, survivor, 0.0)
    export = xp.where(outlet3, crossing, 0.0).sum(axis=-1)
    arrive = cross & ~outlet3
    out_internal = xp.where(arrive, survivor, 0.0).sum(axis=-1)

    receiver = network.receiver_index
    S_r = S[receiver]
    V_r = V[receiver]
    R_r = R[receiver]
    positive_v = V3 > 0.0
    # tau = (dx - x) / v <= dt mathematically when crossing; FP division may
    # exceed dt by round-off, so the remaining time is floored at 0.
    remaining = xp.where(cross & positive_v, dt - travel_src / xp.where(positive_v, V3, 1.0), 0.0)
    remaining = xp.maximum(remaining, 0.0)
    distance = V_r[..., None] * remaining  # <= v_r dt <= courant_max dx < dx
    settle_r3 = S_r[..., None]
    moving_arrival = arrive & ~settle_r3
    R_r3 = R_r[..., None]
    lost_arrival = xp.where(moving_arrival, survivor * (-xp.expm1(-R_r3 * distance)), 0.0)
    settled_arrival = xp.where(arrive & settle_r3, survivor, 0.0)
    survivor2 = xp.where(moving_arrival, survivor * xp.exp(-R_r3 * distance), 0.0)

    in_internal = xp.zeros((n, nc), dtype=np.float64)
    scatter_add(in_internal, receiver, out_internal)
    arrival_decay = xp.zeros((n, nc), dtype=np.float64)
    scatter_add(arrival_decay, receiver, lost_arrival.sum(axis=-1))
    arrival_settled = xp.zeros((n, nc), dtype=np.float64)
    scatter_add(arrival_settled, receiver, settled_arrival.sum(axis=-1))

    # Candidates: one destination per slot.
    self_index = xp.arange(n, dtype=np.int64)[:, None, None]
    dest_cell = xp.where(arrive, receiver[:, None, None], self_index)
    dest_x = xp.where(cross, distance, Xc + travel)
    dest_mass = xp.where(cross, survivor2, survivor)
    has = dest_mass > 0.0
    dest_bin = bin_index(dest_x, dx, nb, xp)
    cls = xp.arange(nc, dtype=np.int64)[None, :, None]
    key = (dest_cell * nc + cls) * nb + dest_bin

    size = n * nc * nb
    mass_new = xp.zeros(size, dtype=np.float64)
    scatter_add(mass_new, key, dest_mass)
    above = 2.0 * dx
    xmax = xp.full(size, _BELOW_ANY_POSITION, dtype=np.float64)
    scatter_max(xmax, key, xp.where(has, dest_x, _BELOW_ANY_POSITION))
    xmin = xp.full(size, above, dtype=np.float64)
    scatter_min(xmin, key, xp.where(has, dest_x, above))
    R_bin = xp.broadcast_to(R[:, :, None], (n, nc, nb)).reshape(-1)
    r_cand = R_bin[key]
    safe_cand = xp.where(has, dest_mass, 1.0)
    # Two evaluations of the logarithmic mean, selected per bin (see
    # `_log_mean_select`): the NARROW form shifts by the largest position
    # (weights expm1(r (x - xmax)) in [expm1(-NARROW_SPREAD), 0], no
    # cancellation, accurate as r -> 0); the WIDE form shifts by the largest
    # log-weight log m + r x, a positive sum with a leading term of exactly 1,
    # immune to a heavy constituent far below a light one at xmax.
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
    # Round-off containment only: the mean lies within the constituents' range.
    x_rep = xp.minimum(xp.maximum(x_rep, xmin), xmax)
    x_rep = xp.where(occupied, x_rep, 0.0)
    return (mass_new.reshape(n, nc, nb), x_rep.reshape(n, nc, nb), dep_decay, settled, export, out_internal,
            in_internal, arrival_decay, arrival_settled)


# --- step -----------------------------------------------------------------------------------
def characteristic_step(
    network: TransportNetwork,
    mobile_before_kg: Any,
    phase: PhaseState,
    pickup_kg: Any,
    sediment_velocity_m_s: Any,
    deposition_rate_per_m: Any,
    settle_mask: Any,
    dt_s: float,
    *,
    n_substeps: int = 1,
    courant_max: float = DEFAULT_COURANT_MAX,
    implementation: str = "array",
) -> CharacteristicStep:
    """Apply the characteristic operator for `dt_s` (module docstring).

    `mobile_before_kg` is the PRE-pickup MAPLE mobile mass `(ny, nx, nc)`
    that `phase` partitions; `pickup_kg` the ACTUAL pickup of this step
    (injected once, at `x = 0`, in the first substep). `sediment_velocity_m_s`
    (>= 0), `deposition_rate_per_m` (>= 0, `1/L`, 0 = no deposition) are
    `(ny, nx, nc)` FP64, `settle_mask` bool. Pure: raises `TransportError`
    (or the recoverable `TransportStepRejected` for the Courant limit)
    before returning anything."""
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

    if not isinstance(network, TransportNetwork):
        raise TransportError("network must be a TransportNetwork (use transport_network)")
    if implementation not in IMPLEMENTATIONS:
        raise TransportError(f"implementation must be one of {IMPLEMENTATIONS}, got {implementation!r}")
    xp = network.xp
    if implementation == "numba" and xp is not np:
        raise TransportError("implementation 'numba' runs on host NumPy arrays only; the network lives in "
                             f"{xp.__name__!r} and no host/device transfer is performed")
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
    nc = int(mobile_before_kg.shape[-1]) if hasattr(mobile_before_kg, "shape") and len(mobile_before_kg.shape) == 3 else 0
    if nc < 1:
        raise TransportError("mobile_before_kg must be (ny, nx, nc) with nc >= 1, got shape "
                             f"{getattr(mobile_before_kg, 'shape', None)}")
    shape3 = (ny, nx, nc)
    floats = {"mobile_before_kg": mobile_before_kg, "pickup_kg": pickup_kg,
              "sediment_velocity_m_s": sediment_velocity_m_s, "deposition_rate_per_m": deposition_rate_per_m}
    _require(floats, shape3, xp, np.float64)
    _require({"settle_mask": settle_mask}, shape3, xp, np.bool_)

    checks = DeferredChecks()
    for name, array in floats.items():
        checks.require(finite_flag(array), f"{name} must be finite everywhere")
        checks.forbid(negative_flag(array), f"{name} must be >= 0 everywhere")
    _phase_checks(phase, mobile_before_kg, network, checks)
    nb = phase.n_bins
    active = network.active_flat
    outlet = network.outlet_flat
    inactive3 = (~active)[:, None]
    M = mobile_before_kg.reshape(n, nc)
    P = pickup_kg.reshape(n, nc)
    V = sediment_velocity_m_s.reshape(n, nc)
    R = deposition_rate_per_m.reshape(n, nc)
    S = settle_mask.reshape(n, nc)
    F = phase.fraction.reshape(n, nc, nb)
    X = phase.position_m.reshape(n, nc, nb)
    checks.forbid(true_flag(inactive3 & (M != 0.0)), "mobile_before_kg must be 0 on inactive cells")
    checks.forbid(true_flag(inactive3 & (P != 0.0)), "pickup_kg must be 0 on inactive cells")
    checks.forbid(true_flag(inactive3 & (V != 0.0)), "sediment_velocity_m_s must be 0 on inactive cells")

    dx = network.dx_m
    dt_sub = dt / n_sub
    with errstate(xp=xp, all="ignore"):
        a = V * (dt_sub / dx)
        max_courant = xp.max(xp.where(active[:, None], a, 0.0))
        checks.require(finite_flag(a), "characteristic sediment Courant number overflowed FP64")
        checks.forbid(true_flag(active[:, None] & (a > cr_max)),
                      f"{_COURANT_REJECTION} courant_max = {cr_max}; step rejected (retry with a smaller dt "
                      "or more substeps)")
        exponent = (V * R) * dt_sub
        max_exponent = xp.max(xp.where(active[:, None], exponent, 0.0))
        checks.require(finite_flag(exponent), "deposition hazard v r dt overflowed FP64")
    # Input validation must complete before any arithmetic on the phase
    # arrays that assumes the invariants (one batched read).
    try:
        checks.resolve()
    except ValueError as exc:
        message = str(exc)
        raise (TransportStepRejected if message.startswith(_COURANT_REJECTION) else TransportError)(message) from None

    with errstate(xp=xp, all="ignore"):
        W = M[..., None] * F  # component masses from the authoritative total
        Xw = X
        dep_decay = xp.zeros((n, nc), dtype=np.float64)
        settled = xp.zeros((n, nc), dtype=np.float64)
        export = xp.zeros((n, nc), dtype=np.float64)
        out_internal = xp.zeros((n, nc), dtype=np.float64)
        in_internal = xp.zeros((n, nc), dtype=np.float64)
        arrival = xp.zeros((n, nc), dtype=np.float64)
        no_pickup = xp.zeros((n, nc), dtype=np.float64)
        if implementation == "numba":
            from maple_syrup import characteristic_numba

            run = characteristic_numba.run_substep
        else:
            run = None
        for k in range(n_sub):
            packet = P if k == 0 else no_pickup  # pickup injected ONCE
            if run is None:
                pieces = _substep_array(xp, network, W, Xw, packet, V, R, S, dx, dt_sub, nb)
            else:
                pieces = run(network, W, Xw, packet, V, R, S, dx, dt_sub, nb)
            W, Xw, d_decay, d_settled, d_export, d_out, d_in, d_arr_decay, d_arr_settled = pieces
            dep_decay += d_decay + d_arr_decay
            settled += d_settled + d_arr_settled
            export += d_export
            out_internal += d_out
            in_internal += d_in
            arrival += d_arr_decay + d_arr_settled

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

        # Phase after the requests: fractions of the remaining pool.
        occupied = remaining > 0.0
        canonical = xp.zeros((1, 1, nb), dtype=np.float64)
        canonical[..., 0] = 1.0
        fraction_new = xp.where(occupied[..., None], W / xp.where(occupied, remaining, 1.0)[..., None], canonical)
        position_new = xp.where(W > 0.0, Xw, 0.0)

        # Face crossings in MAPLE layout (same tables and empty-orientation
        # guard as sediment_transport.transport_step).
        crossing = out_internal + export
        x_gross = xp.zeros((ny, nx + 1, nc), dtype=np.float64)
        y_gross = xp.zeros((ny + 1, nx, nc), dtype=np.float64)
        x_net = xp.zeros((ny, nx + 1, nc), dtype=np.float64)
        y_net = xp.zeros((ny + 1, nx, nc), dtype=np.float64)
        if network.cells_x.size:
            vx = crossing[network.cells_x]
            scatter_add(x_gross, (network.rows_x, network.faces_x), vx)
            scatter_add(x_net, (network.rows_x, network.faces_x), vx * network.sign_x[:, None])
        if network.cells_y.size:
            vy = crossing[network.cells_y]
            scatter_add(y_gross, (network.faces_y, network.cols_y), vy)
            scatter_add(y_net, (network.faces_y, network.cols_y), vy * network.sign_y[:, None])

    checks = DeferredChecks()
    for name, array in (("mobile_after_transfer", T), ("deposition_request", deposition),
                        ("export_request", export), ("internal_transfer_in", in_internal),
                        ("internal_transfer_out", out_internal), ("x_face_gross", x_gross),
                        ("y_face_gross", y_gross), ("x_face_net", x_net), ("y_face_net", y_net),
                        ("decay_deposition", dep_decay), ("settled", settled), ("phase fraction", fraction_new),
                        ("phase position", position_new), ("remaining mass", W)):
        checks.require(finite_flag(array), f"characteristic transport produced non-finite {name}")
    for name, array in (("mobile_after_transfer", T), ("deposition_request", deposition),
                        ("export_request", export), ("internal_transfer_in", in_internal),
                        ("internal_transfer_out", out_internal), ("x_face_gross", x_gross),
                        ("y_face_gross", y_gross), ("decay_deposition", dep_decay), ("settled", settled),
                        ("phase fraction", fraction_new), ("phase position", position_new), ("remaining mass", W)):
        checks.forbid(negative_flag(array), f"characteristic transport produced negative {name}")
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
    checks.forbid(true_flag(export[~outlet] != 0.0), "export requested at a non-outlet cell")
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
    return CharacteristicStep(transport=transport, phase_after=phase_after, mobile_before_kg=grid(before),
                              mobile_remaining_kg=grid(remaining), arrival_deposition_kg=grid(arrival),
                              crossing_kg=grid(crossing), n_bins=nb, implementation=implementation)
