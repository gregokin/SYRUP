"""Background/species preferential intake on a FLOATING porous column (D2; NumPy and compiled Numba).

A pure trial stage on a real `floating_soil_layers.FloatingSoilColumn` / `FloatingSoilState` (never a `SoilColumn` or `SpatialSoilColumn`; the old classes keep refusing each other). The law is the continuous
pathway of `preferential_infiltration` (one aggregate rate `k = -log1p(-beta)/3600`, ONE aggregate rate cap, fixed receiving weights, top layer first; each receiver bounded by its room and the remaining
surface water) with the ACTUAL porous capacity of the layer `sat_m = theta_sat * h`, `h = max(0, min(D, e_{l+1}) - e_l)`. The derivation of the receiver map is the SAME function the common and spatial operators call
(`preferential_infiltration._derive_pathway`); the receiving weights are FIXED and are never renormalised over porous layers: a receiver with no porous capacity (a rock slot, `h = 0`, room 0) simply admits
nothing and its share STAYS ON THE SURFACE (the existing rejected-share rule; no other receiver is topped up). Surface water that matrix intake already credited must be passed as the reduced pool by the caller (one aggregate
supply, no double credit).

    op = prepare_floating_preferential(column, a3=..., max_fraction_per_hour=..., max_intake_rate_m_s=..., background_receiver_weights=..., species_receiver_weights=(), species_contributions=None)
    st = floating_preferential_step(op, surface_water_m, state, dt_s)      # FloatingPreferentialStep (fresh state, same clock)

INACTIVE cells carry no capacity law: their bound is `+inf` in the reader (finiteness and `>= 0` are still validated on every cell), their receiver column is 0, their water and surface are passed through bit for
bit and excluded from the active budgets. One physical body per backend: NumPy `preferential_infiltration._reference_step_body` through lazy floating rows; Numba the FLOATING compile-time variant of the shared
kernel (`preferential_infiltration_numba._build(numba, False, True)`). CUDA owners dispatch from the same entry points to `floating_preferential_infiltration_cuda` (current device gate FIRST, metadata, then
device/host admission, then the shared body; the cold derivation uses the bounded counted 32-byte reducer; the host receives ONE 72-byte raw packet rebuilt into the unchanged 9-float packet).
"""

from __future__ import annotations

import contextlib
import hashlib
import math
import threading
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from maple_syrup import floating_soil_layers as fsl
from maple_syrup import preferential_infiltration as pi
from maple_syrup import subsurface as ss
from maple_syrup.conservation import volume_roundoff_bound_m3

__all__ = [
    "FLOATING_PREFERENTIAL_VERSION",
    "FloatingPreferentialMemory",
    "FloatingPreferentialOperator",
    "FloatingPreferentialStep",
    "estimate_floating_preferential_memory",
    "floating_preferential_step",
    "prepare_floating_preferential",
    "weight_record_bytes",
]

FLOATING_PREFERENTIAL_VERSION = "1"
OPERATOR_TAG = "maple_syrup.floating_preferential.operator.v1"
ROW_TEMP_DOUBLES = 8  # (N,) rows of the lazily derived h / bound / tolerance of one layer
_F8 = 8
_CONSTRUCT = threading.local()


@contextlib.contextmanager
def _minting():
    _CONSTRUCT.allowed = True
    try:
        yield
    finally:
        _CONSTRUCT.allowed = False


@dataclass(frozen=True)
class FloatingPreferentialMemory:
    backend: str
    operation: str
    n_layers: int
    n_cells: int
    outputs_bytes: int
    scratch_bytes: int
    retained_bytes: int
    required_bytes: int
    scope: str = "modelled host payload bytes of this call; caller state/surface and the owner's retained maps are excluded"


PY_RECORD_ALLOWANCE_BYTES = 4096  # declared Python object allowance of ONE weight record (the `ReceiverWeights` object, its owned arrays' headers and the species metadata slot)


def weight_record_bytes(n_layers: int, n_species: int) -> int:
    """Host bytes of the retained and transient weight records of ONE preparation: `n_species + 1` records (the background record is counted even when absent: a bounded upper bound), each holding the raw
    and the normalised length-L float64 vectors (`2 L 8`), the build/seal scratch of ONE record at a time (`3 L 8`: the validated copy, the normalised vector and a sealing copy) and the declared Python
    allowance. Counted in the HOST budget BEFORE any record is built; no 'small' assumption."""
    L, S = fsl._count(n_layers, "n_layers", minimum=1), fsl._count(n_species, "n_species")
    return (S + 1) * (2 * L * _F8 + 3 * L * _F8 + PY_RECORD_ALLOWANCE_BYTES)


def estimate_floating_preferential_memory(n_layers: int, n_cells: int, *, backend: str = "numpy", operation: str = "step", n_species: int = 0, reserve_bytes: int = 0) -> FloatingPreferentialMemory:
    L, N = fsl._count(n_layers, "n_layers", minimum=1), fsl._count(n_cells, "n_cells", minimum=1)
    if backend not in ("numpy", "numba"):
        raise fsl.FloatingBackendUnavailableError(f"this module serves the numpy and numba owners, got {backend!r}")
    if operation not in ("prepare", "step"):
        raise fsl.FloatingSoilError(f"operation must be 'prepare' or 'step', got {operation!r}")
    retained = (L * N + 2 * N) * _F8  # receiver map + rate + (at most) a per-cell max rate
    S = fsl._count(n_species, "n_species")
    if operation == "prepare":
        scratch = (L * N + 6 * N + 4 * N) * _F8 + 4 * N  # derivation transients (cover, denominator, layer sum, temporaries, masks)
        weights = weight_record_bytes(L, S)  # retained records + per-record build/seal scratch + declared Python allowance
        return FloatingPreferentialMemory(backend, operation, L, N, 0, scratch + weights, retained, retained + scratch + weights + fsl._count(reserve_bytes, "reserve_bytes"))
    outputs = (L * N + L * N + 2 * N) * _F8 + pi.ROW_COUNT * N * _F8
    scratch = (4 * L * N * _F8 if backend == "numpy" else 2 * L * _F8) + ROW_TEMP_DOUBLES * N * _F8 + 3 * N * _F8
    return FloatingPreferentialMemory(backend, operation, L, N, outputs, scratch, 0, outputs + scratch + fsl._count(reserve_bytes, "reserve_bytes"))


@dataclass(frozen=True, eq=False)
class FloatingPreferentialOperator:
    """Sealed operator minted ONLY by `prepare_floating_preferential`: the column reference, the sealed derived maps `(rate (N,), max-rate (1,) or (N,), receiver (L, N))`, the weight records and metadata."""

    column: Any
    backend: str
    n_layers: int
    n_cells: int
    column_sha256: str
    sha256: str
    layouts: Any
    max_fraction_per_hour: float
    n_species: int
    n_species_enabled: int
    background: Any
    species: tuple
    retained_bytes: int
    _law: tuple
    tag: str = OPERATOR_TAG
    device_id: int | None = None  # the CUDA owner's device (None for a CPU owner)

    def __post_init__(self) -> None:
        if not getattr(_CONSTRUCT, "allowed", False):
            raise fsl.FloatingSoilError("a FloatingPreferentialOperator is created only by prepare_floating_preferential (no direct construction, replace() or copy)")

    def __copy__(self):
        raise fsl.FloatingSoilError("a FloatingPreferentialOperator is not copyable; prepare another")

    def __deepcopy__(self, memo):
        raise fsl.FloatingSoilError("a FloatingPreferentialOperator is not copyable; prepare another")

    def __reduce_ex__(self, protocol):
        raise fsl.FloatingSoilError("a FloatingPreferentialOperator is not serialisable")

    def summary(self) -> dict:
        return {"tag": self.tag, "version": FLOATING_PREFERENTIAL_VERSION, "backend": self.backend, "n_layers": self.n_layers, "n_cells": self.n_cells, "column_sha256": self.column_sha256,
                "sha256": self.sha256, "layouts": dict(self.layouts), "max_fraction_per_hour": self.max_fraction_per_hour, "n_species": self.n_species, "n_species_enabled": self.n_species_enabled,
                "retained_bytes": self.retained_bytes, "pure_trial": True, "generation_bound": False, "exactly_once": False,
                "law": "continuous preferential pathway of preferential_infiltration, actual porous capacity; fixed receiving weights, zero-capacity shares rejected to the surface"}


@dataclass(frozen=True, eq=False)
class FloatingPreferentialStep:
    surface_water_m: Any
    state: Any
    intake_m: Any
    intake_total_m: Any
    dt_s: float
    budget: pi.PreferentialBudget
    memory: FloatingPreferentialMemory


def _input_kind(column: Any, value: Any, name: str) -> str:
    if isinstance(value, (bool, np.bool_)):
        raise fsl.FloatingSoilError(f"{name} must be numeric, got a Boolean")
    if fsl.sp._is_real_scalar(value):
        return "scalar"
    fsl._check_vec(column, name, value, (column.n_cells,))
    return "cell"


def _values(value: Any, kind: str, name: str, low: float, high: float, *, allow_inf: bool, n: int) -> tuple[np.ndarray, float | None]:
    if kind == "scalar":
        v = float(value)
        if math.isnan(v) or (not allow_inf and not math.isfinite(v)):
            raise fsl.FloatingSoilError(f"{name} must be finite, got {value!r}")
        if v < low or v > high:
            raise fsl.FloatingSoilError(f"{name} must lie in [{low!r}, {high!r}], got {v!r}")
        return np.broadcast_to(np.float64(v), (n,)), v
    if bool(np.isnan(value).any()) or (not allow_inf and not bool(np.isfinite(value).all())):
        raise fsl.FloatingSoilError(f"{name} must {'not contain NaN' if allow_inf else 'be finite'} everywhere (inactive cells included)")
    lo, hi = float(value.min()), float(value.max())
    if lo < low or hi > high:
        raise fsl.FloatingSoilError(f"{name} must lie in [{low!r}, {high!r}] everywhere (inactive cells included), got [{lo!r}, {hi!r}]")
    return value, None


def _seal_map(array: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(array, dtype=np.float64)
    flat = np.frombuffer(a.tobytes(), dtype=np.float64)  # bytes-backed read-only storage (one counted copy)
    return flat if a.ndim == 1 else flat.reshape(a.shape)


def rebind_preferential_column(op: FloatingPreferentialOperator, new_column: fsl.FloatingSoilColumn, *, memory_budget_bytes: Any = None, reserve_bytes: Any = 0, available_bytes_fn: Callable[[], int] | None = None,
                               host_memory_budget_bytes: Any = None, host_reserve_bytes: Any = 0, host_available_bytes_fn: Callable[[], int] | None = None) -> FloatingPreferentialOperator:
    """Rebinding of a preferential operator to a NEW prepared column of the same physical domain (an event-boundary property refresh). The sealed derived maps (rate, max-rate, receiver weights) depend on the explicit a3 and
    the receiving weights only -- never on field capacity, saturation or depth -- and are SHARED by reference (no array copy) with the raw weight records. The result carries the CANONICAL identity: exactly the operator
    `prepare_floating_preferential` would mint on `new_column` from the same raw laws (the same `_identity` byte recipe over the new column sha, layouts, max fraction, background/species records and the law arrays), so a durable
    new root reconstructed from the raw spec reproduces it. Cost: one cold streamed hash over the law arrays (O(L N) time, bounded chunk buffers, once per event boundary; never in a step). Requires the exact operator type and a
    column of the same dimensions, backend, device, area, active mask and nominal edges (a different domain is refused). CPU: unchanged behaviour (the quota arguments are refused there). CUDA: the current device is gated first
    (`fsl._gate`), then the two-pool admission of the hash staging buffer (`min(L N, STAGE_ELEMS)` float64 on the host, nothing on the device) runs BEFORE the buffer exists, then the identity streams the device law arrays
    through that bounded buffer (`hash_device_array`: no host mirror, no CPU implicit conversion). Geometry views keep their own hot identity (unchanged)."""
    column = _check_operator(op)
    fsl._check_column(new_column)
    fsl._gate(new_column)
    cuda = column.backend == "cuda" or new_column.backend == "cuda"
    if not cuda and (memory_budget_bytes is not None or host_memory_budget_bytes is not None or reserve_bytes or host_reserve_bytes or available_bytes_fn is not None or host_available_bytes_fn is not None):
        raise fsl.FloatingSoilError("the admission arguments of rebind_preferential_column apply only to the cuda backend")
    if (new_column.backend != column.backend or (new_column.n_layers, new_column.n_cells) != (column.n_layers, column.n_cells) or new_column.cell_area_m2 != column.cell_area_m2
            or new_column.mask_sha256 != column.mask_sha256 or new_column.edges_sha256 != column.edges_sha256 or new_column.device_id != column.device_id):
        raise fsl.FloatingSoilError("the new column is not the physical domain of the operator's column (backend, dimensions, area, active mask or nominal edges); refused")
    if cuda:
        from maple_syrup import floating_soil_layers_cuda as fcu

        quota = {"memory_budget_bytes": memory_budget_bytes, "reserve_bytes": reserve_bytes, "available_bytes_fn": available_bytes_fn, "host_memory_budget_bytes": host_memory_budget_bytes,
                 "host_reserve_bytes": host_reserve_bytes, "host_available_bytes_fn": host_available_bytes_fn}
        n = min(column.n_layers * column.n_cells, _fpc().STAGE_ELEMS)
        fcu.admit(fcu._rc().require_cuda(), column.device_id, 0, n * 8, quota, "preferential canonical rebind (identity hash staging buffer)")  # BEFORE the buffer exists
        host_buf = np.empty(n, dtype=np.float64)
        sha = _identity(new_column, op._law, op.layouts, op.max_fraction_per_hour, op.background, op.species, host_buf)
        del host_buf
    else:
        sha = _identity(new_column, op._law, op.layouts, op.max_fraction_per_hour, op.background, op.species)
    with _minting():
        return FloatingPreferentialOperator(new_column, op.backend, op.n_layers, op.n_cells, new_column.sha256, sha, op.layouts, op.max_fraction_per_hour, op.n_species, op.n_species_enabled, op.background,
                                            op.species, op.retained_bytes, op._law, device_id=op.device_id)


def _identity(column: Any, law: tuple, kinds: dict, max_fraction: float, background: Any, species: Sequence[Any], host_buf: np.ndarray | None = None) -> str:
    from maple_syrup import spatial_matrix_infiltration as sm
    from maple_syrup import spatial_preferential_infiltration as spi

    h = hashlib.sha256()
    for part in (OPERATOR_TAG, column.sha256, column.backend, str((column.n_layers, column.n_cells)), repr(float(max_fraction)), repr(sorted(kinds.items()))):
        h.update(part.encode())
        h.update(b"\0")
    for label, rec in (("background", background), *((f"species{s}", w) for s, w in enumerate(species))):
        spi._hash_weights(h, label, rec)
    if column.backend == "cuda":  # the SAME byte recipe streamed from the device arrays through the counted bounded host staging buffer
        for name, a in zip(("rate_s", "max_rate_m_s", "receiver"), law, strict=True):
            _fpc().hash_device_array(h, name, "map", a, host_buf)
        return h.hexdigest()
    for name, a in zip(("rate_s", "max_rate_m_s", "receiver"), law, strict=True):
        sm._hash_array(h, name, "map", a)
    return h.hexdigest()


def _fpc():
    from maple_syrup import floating_preferential_infiltration_cuda as fpc

    return fpc


def _quota(column: Any, memory_budget_bytes, reserve_bytes, available_bytes_fn, host_memory_budget_bytes, host_reserve_bytes, host_available_bytes_fn) -> dict:
    if column.backend != "cuda" and (host_memory_budget_bytes is not None or host_reserve_bytes or host_available_bytes_fn is not None):
        raise fsl.FloatingSoilError("host_memory_budget_bytes / host_reserve_bytes / host_available_bytes_fn apply only to the cuda backend")
    return {"memory_budget_bytes": memory_budget_bytes, "reserve_bytes": reserve_bytes, "available_bytes_fn": available_bytes_fn, "host_memory_budget_bytes": host_memory_budget_bytes,
                "host_reserve_bytes": host_reserve_bytes, "host_available_bytes_fn": host_available_bytes_fn}


def _admit(column: Any, operation: str, quota: dict, what: str, n_species: int = 0) -> FloatingPreferentialMemory:
    """CPU: the modelled host payload against the budget/reading. CUDA: device and host separately (`floating_soil_layers_cuda.admit`)."""
    reserve = fsl._count(quota["reserve_bytes"], "reserve_bytes")
    if column.backend == "cuda":
        from maple_syrup import floating_soil_layers_cuda as fcu

        t = _fpc().operation_terms(operation, column.n_layers, column.n_cells)
        weights = weight_record_bytes(column.n_layers, n_species) if operation == "prepare" else 0  # host records: counted in the HOST budget (they never enter the device budget)
        fcu.admit(fcu._rc().require_cuda(), column.device_id, t["outputs"] + t["scratch"], t["host"] + weights, quota, what)
        return FloatingPreferentialMemory("cuda", operation, column.n_layers, column.n_cells, t["outputs"], t["scratch"], weights, t["outputs"] + t["scratch"] + t["host"] + weights + reserve)
    est = estimate_floating_preferential_memory(column.n_layers, column.n_cells, backend=column.backend, operation=operation, n_species=n_species, reserve_bytes=reserve)
    budget = quota["memory_budget_bytes"]
    fsl._admit(est.required_bytes, None if budget is None else fsl._count(budget, "memory_budget_bytes"), quota["available_bytes_fn"], what)
    return est


def _prepare_cuda(column: Any, a3: Any, kinds: dict, max_fraction: float, max_intake_rate_m_s: Any, background: Any, species: tuple, species_contributions: Any) -> tuple:
    """CUDA preparation (admission, metadata checks and weight records were done by the caller): module load -> every-cell device validation of the cell inputs (a scalar keeps a one-element device array) -> ONE
    device copy of a per-cell max rate -> the COMMON derivation (`pi._derive_pathway`, species streamed lazily) with the bounded counted cold reducer -> private logically immutable device arrays -> bounded
    streamed identity hash. Returns `(law, n_enabled, sha256)`."""
    from maple_syrup import floating_soil_layers_cuda as fcu

    fpc, cp = _fpc(), fcu._rc().require_cuda()
    L, N = column.n_layers, column.n_cells
    try:
        fpc.load_kernels(cp, column.device_id)
    except ss.SubsurfaceBackendUnavailableError as exc:
        raise fsl.FloatingBackendUnavailableError(str(exc)) from exc
    views = []
    for value, kind, name, lo, hi in ((a3, kinds["a3"], "a3", 0.0, 1.0), (max_intake_rate_m_s, kinds["max_intake_rate_m_s"], "max_intake_rate_m_s", 0.0, math.inf)):
        if kind == "scalar":
            v = float(value)
            if math.isnan(v) or v < lo or v > hi:
                raise fsl.FloatingSoilError(f"{name} must lie in [{lo!r}, {hi!r}], got {v!r}")
            views.append(cp.array(np.array([v], dtype=np.float64)))  # one-element device array (8 B)
        else:
            if not fpc.validate_range(cp, column.device_id, value, lo, hi):
                raise fsl.FloatingSoilError(f"{name} must lie in [{lo!r}, {hi!r}] on every cell (inactive cells included; NaN refused)")
            views.append(value)
    a3_dev, mr_dev = views
    a3_view = cp.broadcast_to(a3_dev, (N,)) if kinds["a3"] == "scalar" else a3_dev  # a zero-copy read-only broadcast of a scalar; a cell array is the caller's own (read, never written)
    mr = mr_dev if kinds["max_intake_rate_m_s"] == "scalar" else cp.array(mr_dev, dtype=np.float64, copy=True)
    reducer = fpc.ColdReducer(cp, column.device_id)
    ctx = pi._DeriveContext(cp, L, N, column._maps["active"], lambda s_i, c: fsl._check_vec(column, f"species_contributions[{s_i}]", c, (N,)), reducer)
    rate, receiver, beta, n_enabled = pi._derive_pathway(ctx, a3_view, max_fraction, background, species, species_contributions)
    del beta, a3_view, a3_dev
    law = (rate, mr, receiver)
    host_buf = np.empty(min(L * N, fpc.STAGE_ELEMS), dtype=np.float64)
    sha = _identity(column, law, kinds, max_fraction, background, species, host_buf)
    del host_buf
    return law, n_enabled, sha


def prepare_floating_preferential(column: fsl.FloatingSoilColumn, *, a3: Any, max_fraction_per_hour: Any, max_intake_rate_m_s: Any, background_receiver_weights: Any,
                                  species_receiver_weights: Sequence[Any] = (), species_contributions: Iterable[Any] | None = None, memory_budget_bytes: Any = None, reserve_bytes: Any = 0,
                                  available_bytes_fn: Callable[[], int] | None = None, host_memory_budget_bytes: Any = None, host_reserve_bytes: Any = 0,
                                  host_available_bytes_fn: Callable[[], int] | None = None) -> FloatingPreferentialOperator:
    """Mint the sealed operator for a FLOATING column (numpy | numba | cuda). Inputs and meanings as `preferential_infiltration.prepare_preferential`; `a3` (in `[0, 1]`) and `max_intake_rate_m_s` (`>= 0`, `inf`
    allowed) are explicit host scalars or exact `(N,)` float64 arrays of the owner's namespace (no texture inference). Order: current-device gate (CUDA) -> owner/metadata refusals -> ADMISSION -> weight
    records -> value validation on EVERY cell -> the common derivation (species streamed lazily; CUDA: the bounded counted cold reducer) -> sealing -> the one-off compile -> bounded streamed identity hash -> mint."""
    fsl._gate(column)
    fsl._check_column(column)
    L, N = column.n_layers, column.n_cells
    max_fraction = fsl._real(max_fraction_per_hour, "max_fraction_per_hour")
    if not (0.0 <= max_fraction <= 1.0):
        raise fsl.FloatingSoilError(f"max_fraction_per_hour must lie in [0, 1], got {max_fraction!r}")
    if isinstance(species_receiver_weights, (str, bytes)) or not isinstance(species_receiver_weights, Sequence):
        raise fsl.FloatingSoilError("species_receiver_weights must be a sequence of per-species length-L vectors")
    S = len(species_receiver_weights)
    if S and species_contributions is None:
        raise fsl.FloatingSoilError(f"{S} species were given but species_contributions is None")
    if not S and species_contributions is not None:
        raise fsl.FloatingSoilError("species_contributions were given without species_receiver_weights")
    quota = _quota(column, memory_budget_bytes, reserve_bytes, available_bytes_fn, host_memory_budget_bytes, host_reserve_bytes, host_available_bytes_fn)
    kinds = {"a3": _input_kind(column, a3, "a3"), "max_intake_rate_m_s": _input_kind(column, max_intake_rate_m_s, "max_intake_rate_m_s")}
    _admit(column, "prepare", quota, "floating preferential preparation", S)
    background = None if background_receiver_weights is None else pi.receiver_weights(background_receiver_weights, L, "background_receiver_weights")
    species = tuple(pi.receiver_weights(w, L, f"species_receiver_weights[{s}]") for s, w in enumerate(species_receiver_weights))
    if column.backend == "cuda":
        law, n_enabled, sha = _prepare_cuda(column, a3, kinds, max_fraction, max_intake_rate_m_s, background, species, species_contributions)
        with _minting():
            return FloatingPreferentialOperator(column, "cuda", L, N, column.sha256, sha, kinds, max_fraction, S, n_enabled, background, species, int(sum(a.nbytes for a in law)), law,
                                                device_id=column.device_id)
    a3_view, _ = _values(a3, kinds["a3"], "a3", 0.0, 1.0, allow_inf=False, n=N)
    mr_view, mr_scalar = _values(max_intake_rate_m_s, kinds["max_intake_rate_m_s"], "max_intake_rate_m_s", 0.0, math.inf, allow_inf=True, n=N)
    active = column._maps["active"]

    def check_contribution(s_i: int, c: Any) -> None:
        fsl._check_vec(column, f"species_contributions[{s_i}]", c, (N,))

    ctx = pi._DeriveContext(np, L, N, active, check_contribution)
    rate, receiver, beta, n_enabled = pi._derive_pathway(ctx, a3_view, max_fraction, background, species, species_contributions)
    del beta, a3_view
    sealed_receiver, sealed_rate = _seal_map(receiver), _seal_map(rate)
    del receiver, rate
    sealed_mr = _seal_map(np.array([mr_scalar], dtype=np.float64) if mr_scalar is not None else mr_view)
    law = (sealed_rate, sealed_mr, sealed_receiver)
    if column.backend == "numba":
        from maple_syrup import preferential_infiltration_numba as pn

        try:
            pn.compiled_floating_kernel()
        except ss.SubsurfaceBackendUnavailableError as exc:
            raise fsl.FloatingBackendUnavailableError(str(exc)) from exc
        except Exception as exc:
            raise fsl.FloatingBackendUnavailableError(f"the compiled floating preferential kernel failed to build ({type(exc).__name__}: {exc}); there is no NumPy fallback") from exc
    sha = _identity(column, law, kinds, max_fraction, background, species)
    with _minting():
        return FloatingPreferentialOperator(column, column.backend, L, N, column.sha256, sha, kinds, max_fraction, S, n_enabled, background, species, int(sum(a.nbytes for a in law)), law)


def _check_operator(op: Any) -> fsl.FloatingSoilColumn:
    """Metadata-only revalidation (O(1), no hashing, no copy). A CUDA owner: the current device is checked FIRST."""
    if type(op) is not FloatingPreferentialOperator:
        raise fsl.FloatingSoilError(f"op must be a FloatingPreferentialOperator, got {type(op).__name__}")
    column = op.column
    fsl._gate(column)
    fsl._check_column(column)
    if op.column_sha256 != column.sha256 or (op.n_layers, op.n_cells) != (column.n_layers, column.n_cells) or op.backend != column.backend:
        raise fsl.FloatingSoilError("the operator is not bound to its column (metadata mismatch)")
    if column.backend == "cuda":
        shapes = ((op.n_cells,), (1,) if op.layouts["max_intake_rate_m_s"] == "scalar" else (op.n_cells,), (op.n_layers, op.n_cells))
        for name, a, shape in zip(("rate", "max_rate", "receiver"), op._law, shapes, strict=True):
            if type(a) is not fsl._ns_type(column) or a.dtype != np.float64 or tuple(a.shape) != shape or not a.flags.c_contiguous or int(a.device.id) != column.device_id or op.device_id != column.device_id:
                raise fsl.FloatingSoilError(f"operator law input {name} lost its exact type/dtype/shape/device; refused")
    return column


def rebind_preferential_geometry_view(op: FloatingPreferentialOperator, view: fsl.FloatingSoilColumn) -> FloatingPreferentialOperator:
    """HOT rebinding of a CPU preferential operator to an official geometry view of the SAME prepared column. The sealed derived maps (rate, max-rate and the receiver weights) depend on the explicit a3 and weights
    only, never on the depth, and are SHARED unchanged; the porous capacity that limits the receivers is read from the view's actual slabs at every step. O(1): nothing hashed or copied."""
    column = _check_operator(op)
    fsl._check_column(view)
    fsl._gate(view)  # CUDA: the current device again for the view, before any identity work
    root = column.sha256 if column.parent_sha256 is None else column.parent_sha256
    if (view.parent_sha256 != root or view.backend != column.backend or view.device_id != column.device_id or (view.n_layers, view.n_cells) != (column.n_layers, column.n_cells)
            or view.cell_area_m2 != column.cell_area_m2 or view.mask_sha256 != column.mask_sha256 or view.edges_sha256 != column.edges_sha256):
        raise fsl.FloatingSoilError("the view is not a geometry view of the operator's prepared column (parent identity, backend, device, dimensions, area, active mask or edges); refused")
    sha = hashlib.sha256(f"{OPERATOR_TAG}\0view\0{op.sha256}\0{view.sha256}".encode()).hexdigest()
    with _minting():
        return FloatingPreferentialOperator(view, op.backend, op.n_layers, op.n_cells, view.sha256, sha, op.layouts, op.max_fraction_per_hour, op.n_species, op.n_species_enabled, op.background, op.species,
                                            op.retained_bytes, op._law, device_id=op.device_id)


def _numpy_readers(column: fsl.FloatingSoilColumn):
    m, kinds = column._maps, column.layouts
    coef = volume_roundoff_bound_m3(ss.LAYER_OPERATION_TERMS, 1.0)
    e, depth, active = m["edges"], m["soil_depth"], m["active"]
    ts, kts = m["theta_saturation"], kinds["theta_saturation"]

    def sat(k: int) -> np.ndarray:
        h = np.maximum(np.minimum(depth, e[k + 1]) - e[k], 0.0)
        return np.where(active, fsl._row(ts, kts, k) * h, np.inf)  # inactive placeholders carry no capacity law

    def cap(k: int) -> np.ndarray:
        h = np.maximum(np.minimum(depth, e[k + 1]) - e[k], 0.0)
        return np.where(active, coef * (fsl._row(ts, kts, k) * h), 0.0)

    return fsl._Rows(sat), fsl._Rows(cap)


def _run(op: FloatingPreferentialOperator, surface: Any, water: Any, dt: float) -> pi.StepArrays:
    column = op.column
    L, N = op.n_layers, op.n_cells
    cell_tol = pi.cell_closure_coefficient(L)
    rate, max_rate, recv = op._law
    scalar = op.layouts["max_intake_rate_m_s"] == "scalar"
    m = column._maps
    if op.backend == "numba":
        from maple_syrup import preferential_infiltration_numba as pn

        return pn.floating_launch(surface, water, m["active"].view(np.uint8), dt, rate=rate, max_rate=max_rate, k_mr=0 if scalar else 1, recv=recv, edges=m["edges"], depth=m["soil_depth"],
                                  theta_sat=m["theta_saturation"].reshape(-1), k_sat={"scalar": 0, "shared": 1, "cell": 2}[column.layouts["theta_saturation"]],
                                  coef=volume_roundoff_bound_m3(ss.LAYER_OPERATION_TERMS, 1.0), cell_tol=cell_tol)
    sat, cap = _numpy_readers(column)
    ctx = pi._StepContext(L, N, m["active"], cell_tol)
    return pi._reference_step_body(ctx, rate, np.broadcast_to(max_rate, (N,)) if scalar else max_rate, recv, sat, cap, surface, water, dt)


def floating_preferential_step(op: FloatingPreferentialOperator, surface_water_m: Any, state: fsl.FloatingSoilState, dt_s: Any, *, memory_budget_bytes: Any = None, reserve_bytes: Any = 0,
                               available_bytes_fn: Callable[[], int] | None = None, host_memory_budget_bytes: Any = None, host_reserve_bytes: Any = 0,
                               host_available_bytes_fn: Callable[[], int] | None = None) -> FloatingPreferentialStep:
    """One pure preferential stage over `dt_s`. `surface_water_m` `(N,)` is the standing surface water (the pool left after matrix intake, never credited twice), `state` a `FloatingSoilState` of the operator's
    column. Order: operator (CUDA: current device first) -> state/surface metadata -> dt -> admission -> allocation -> body -> ONE packet reduction -> reject or publish. Inputs are read only; the clock is
    preserved; `dt = 0` is the identity for valid inputs; a refusal publishes nothing. CUDA owner: exact `cupy.ndarray`s of the owner's device, nothing transferred, ONE 72-byte raw packet read back."""
    column = _check_operator(op)
    t_s = fsl._check_state(column, state)
    dt = ss._check_dt(dt_s)
    fsl._check_vec(column, "surface_water_m", surface_water_m, (column.n_cells,))
    quota = _quota(column, memory_budget_bytes, reserve_bytes, available_bytes_fn, host_memory_budget_bytes, host_reserve_bytes, host_available_bytes_fn)
    est = _admit(column, "step", quota, "floating preferential step", op.n_species)
    if column.backend == "cuda":
        from maple_syrup import floating_soil_layers_cuda as fcu

        surface_o, water_o, intake_o, total_o, packet = _fpc().step(fcu._rc().require_cuda(), op, surface_water_m, state.water_m, dt)
        budget = pi._resolve_core(column.cell_area_m2, op.n_layers, op.n_cells, packet)
        return FloatingPreferentialStep(surface_o, fsl._new_state(column, water_o, t_s), intake_o, total_o, dt, budget, est)
    arrays = _run(op, surface_water_m, state.water_m, dt)
    packet = pi._read_packet(np, arrays)
    budget = pi._resolve_core(column.cell_area_m2, op.n_layers, op.n_cells, packet)
    return FloatingPreferentialStep(arrays.surface_m, fsl._new_state(column, arrays.water_m, t_s), arrays.intake_m, arrays.intake_total_m, dt, budget, est)
