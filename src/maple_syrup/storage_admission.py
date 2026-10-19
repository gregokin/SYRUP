"""Explicit DRIVER-call admission helper for one windowed MAPLE transaction (QUALIFICATION CANDIDATE).

This module is NOT automatic protection: no SYRUP event, adapter or default calls it. A driver that wants the check calls
`admit_public` or `admit_paired` immediately BEFORE the corresponding `sediment_bed.apply_bed_demand` /
`apply_bed_pickup_transport_deposit` call, with the same bed, context and options. It reuses MAPLE's own planner
(`maple.surface.voxels.windowed_admission.windowed_transaction_bytes` / `admit_windowed_transaction`) on the ACTUAL bed
and context, and is bound to the exact merged MAPLE `39bcf79e` package digest (`sediment_bed.MERGED_39BCF79_MAPLE_DIGEST`);
any other source is refused by name (older trees do not provide this helper's planner contract).

What a decision is and is not
-----------------------------
* It is an ESTIMATE of the NEW native-array bytes of ONE MAPLE call (host and device separately, the maximum over the
  planner's stages) plus the driver's own reserves, compared with injected or default availability readings under a safety
  fraction (native default 0.5) by exact rational arithmetic.
* It is NOT a reservation (nothing stops another thread, process or the allocator from taking the memory afterwards) and NOT
  a whole-process bound. Explicitly excluded (`plan.excluded`): publication (`publish_bed_storage` / `apply_journal`
  buffers), the windowed commit, external transport beyond the paired callback's returned arrays, the CuPy pool's retained
  blocks and fragmentation, Numba/CUDA JIT and library workspaces, Python object overhead. The caller supplies reserves for
  those; this module invents none (`reserve_host_bytes` is REQUIRED; `reserve_device_bytes` is required for a device bed).
* PAIRED path: the demand of the deposit phase does not exist until MAPLE calls the transport callback, so its optional
  features cannot be read. The CALLER must declare `DemandFeatures` OR-ed over the pickup and the deposit demand. That
  declaration is a caller duty: it is neither inferred nor validated here from the future callback, and an understated flag
  understates the plan. The public path derives its flags from the actual demand.
* Options: SYRUP's adapters forward no `max_gather_columns`, so every real transaction runs MAPLE's native default (256).
  This helper therefore accepts only `None` or an explicit 256 and REFUSES any other value (it would plan a call that never
  happens); non-default gather support needs a separate adapter/config change. `max_journal_records` is
  `sediment_bed.resolved_journal_cap(context)` (the context cap, or MAPLE's native default), exactly what the adapters pass.
* Validation without coercion: reserves are non-Boolean integers >= 0 (`False`, floats, negatives are refused); `None` for
  `reserve_device_bytes` means 0 ONLY for a host bed. A host bed refuses a nonzero device reserve and any
  `device_available_bytes_fn` before any query. The source digest is checked before the optional native planner module is
  imported, so an older tree gets `StorageAdmissionError`, not `ModuleNotFoundError`.

CPU beds never query or import a device. For a CuPy bed the default effective-free reading is the CURRENT device's
`driver_free + pool.free_bytes()`, further limited to `pool_limit - pool.used_bytes()` when the default pool has a positive
limit (see `effective_free_device_bytes`); a bed on a device other than the current one is refused before any reading, and
no device or limit is ever changed. Readings may be injected for tests or other allocators. No
state is mutated, nothing is written or recorded beyond the returned bounded record; there is no history.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from maple_syrup.sediment_bed import (
    MERGED_39BCF79_MAPLE_DIGEST,
    BedContext,
    BedIntegrationError,
    BedState,
    bound_maple_package_digest,
    is_windowed_bed,
    pending_journal_records,
    resolved_journal_cap,
)

__all__ = ["AdmissionRecord", "DemandFeatures", "StorageAdmissionError", "admit_paired", "admit_public",
           "effective_free_device_bytes", "plan_for_bed"]


class StorageAdmissionError(BedIntegrationError):
    """Unsupported source, bed or input of the explicit admission helper (raised before any query)."""


@dataclass(frozen=True)
class DemandFeatures:
    """Optional demand features that enlarge the plan. For a paired call: the OR over pickup and deposit demands."""

    has_boundary_input: bool = False
    has_boundary_export: bool = False
    has_face_flux: bool = False

    def __post_init__(self):
        for name in ("has_boundary_input", "has_boundary_export", "has_face_flux"):
            if type(getattr(self, name)) is not bool:
                raise StorageAdmissionError(f"{name} must be a bool, got {getattr(self, name)!r}")

    @classmethod
    def from_demand(cls, demand) -> DemandFeatures:
        try:
            return cls(demand.boundary_input_by_cell_class_kg is not None,
                       demand.boundary_export_by_cell_class_kg is not None, demand.face_flux is not None)
        except AttributeError as exc:
            raise StorageAdmissionError(f"not a WaterProcessDemand: {exc}") from None


@dataclass(frozen=True)
class AdmissionRecord:
    """Bounded record of one admitted call: MAPLE's plan and decision plus the options they were made for."""

    plan: Any
    decision: Any
    path: str
    namespace: str
    max_gather_columns: int
    max_journal_records: int
    reserve_host_bytes: int
    reserve_device_bytes: int


def _require_source() -> None:
    digest = bound_maple_package_digest()
    if digest != MERGED_39BCF79_MAPLE_DIGEST:
        raise StorageAdmissionError(
            f"storage admission is bound to the merged MAPLE package {MERGED_39BCF79_MAPLE_DIGEST[:16]}...; the bound "
            f"package is {digest[:16]}... (refused)")


def _namespace_name(state: BedState) -> str:
    from maple.core.backend import array_namespace

    name = str(getattr(array_namespace(state.water.depth_m), "__name__", "")).split(".")[0]
    if name not in ("numpy", "cupy"):
        raise StorageAdmissionError(f"unsupported array namespace {name!r}; only numpy and cupy beds are planned")
    return name


def _checked_int(value, name: str, *, allow_none: bool = False):
    """A non-Boolean Python or NumPy integer >= 0, normalized to `int` (no coercion: `False`, floats, strings, arrays and
    negatives are refused by name). `None` is returned unchanged ONLY when `allow_none` (the caller decides what it means)."""
    if value is None and allow_none:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise StorageAdmissionError(f"{name} must be a non-Boolean integer, got {value!r}")
    if int(value) < 0:
        raise StorageAdmissionError(f"{name} must be >= 0, got {value!r}")
    return int(value)


def _gather_option(value) -> int:
    """This helper plans what SYRUP's adapters execute: they never forward `max_gather_columns`, so MAPLE's native default
    is always used. `None` or an explicit native default is accepted; any other value is refused (an admitted figure for a
    different cap would understate the real transaction) until adapter/config support exists."""
    from maple.surface.voxels.windowed_transfer import DEFAULT_EXCHANGE_GATHER_COLUMNS

    if value is None:
        return DEFAULT_EXCHANGE_GATHER_COLUMNS
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise StorageAdmissionError(f"max_gather_columns must be None or the native default, got {value!r}")
    if int(value) != DEFAULT_EXCHANGE_GATHER_COLUMNS:
        raise StorageAdmissionError(
            f"max_gather_columns={int(value)} is refused: the SYRUP adapters pass no such option and always execute the "
            f"native default {DEFAULT_EXCHANGE_GATHER_COLUMNS}; planning another value would not describe the real call")
    return int(value)


def _largest_overlay_bytes(store) -> int:
    """Largest physical overlay file of a deep-store version: 0 for format 1; for format 2 the maximum over the real slot
    table. Any other format, or a version that cannot report its slots, is REFUSED (never planned as 0)."""
    fmt = getattr(store, "format_version", None)
    if fmt == 1:
        return 0
    if fmt != 2:
        raise StorageAdmissionError(f"unsupported deep-store format {fmt!r}; overlay size is indeterminate (refused)")
    try:
        largest = 0  # streaming maximum: no list, no cache
        for index in range(int(store.n_chunks)):
            overlay = store.slot_files(index).overlay
            if overlay is not None:
                largest = max(largest, int(overlay.nbytes))
    except Exception as exc:  # noqa: BLE001 - re-raised as a named refusal
        raise StorageAdmissionError(f"overlay sizes of the deep-store version are indeterminate ({exc}); refused") from None
    return largest


def plan_for_bed(state: BedState, context: BedContext, *, path: str, features: DemandFeatures,
                 max_gather_columns: int | None = None):
    """MAPLE's `WindowedStepPlan` for the ACTUAL windowed bed and context. Pure; allocates no model array."""
    _require_source()  # BEFORE the optional native planner module is imported: an older tree gets a named refusal
    if not isinstance(features, DemandFeatures):
        raise StorageAdmissionError("features must be a DemandFeatures")
    if not is_windowed_bed(state):
        raise StorageAdmissionError("the admission helper plans resident-window beds only")
    gather = _gather_option(max_gather_columns)
    from maple.surface.voxels.windowed_admission import windowed_transaction_bytes

    column, g = state.voxel_column, context.geometry
    store = column.deep_store
    return windowed_transaction_bytes(
        ny=g.ny, nx=g.nx, n_classes=len(context.grain_classes.classes), n_voxels=column.n_voxels,
        window_voxels=column.window_voxels,
        packet_terms=0 if column.height_packet is None else column.height_packet.packet_terms,
        base_itemsize=column.base.dtype.itemsize, ledger_present_processes=len(state.ledger.present_processes),
        has_availability=state.sediment_availability is not None, has_boundary_input=features.has_boundary_input,
        has_boundary_export=features.has_boundary_export, has_face_flux=features.has_face_flux,
        max_gather_columns=gather, max_journal_records=resolved_journal_cap(context),
        pending_records=pending_journal_records(state), chunk_bytes_max=store.header.chunk_bytes_max,
        overlay_bytes_max=_largest_overlay_bytes(store), mapping_cache_chunks=store.mapping_cache_chunks,
        namespace=_namespace_name(state), path=path)


def effective_free_device_bytes(driver_free_bytes: int, pool_free_bytes: int, pool_limit_bytes: int,
                                pool_used_bytes: int) -> int:
    """Effective free device bytes from four numbers: `driver_free + pool_free`, limited to `pool_limit - pool_used` when the
    pool has a positive limit (0 means unlimited). Never negative."""
    for name, value in (("driver_free_bytes", driver_free_bytes), ("pool_free_bytes", pool_free_bytes),
                        ("pool_limit_bytes", pool_limit_bytes), ("pool_used_bytes", pool_used_bytes)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise StorageAdmissionError(f"{name} must be a nonnegative int, got {value!r}")
    reachable = driver_free_bytes + pool_free_bytes
    if pool_limit_bytes > 0:
        reachable = min(reachable, pool_limit_bytes - pool_used_bytes)
    return max(0, reachable)


def require_current_device(current_device_id: int, bed_device_ids: dict) -> None:
    """Refuse (before any memory reading) when a bed array's device differs from the current device. Pure numbers."""
    for name, device_id in bed_device_ids.items():
        if device_id != current_device_id:
            raise StorageAdmissionError(
                f"{name} lives on device {device_id} but the current device is {current_device_id}; select the bed's "
                "device or inject device_available_bytes_fn (the default query never switches devices)")


def _default_device_query(state: BedState) -> Callable[[], int]:
    """The default effective-free query for the CURRENT device. Refused (before any memory reading) when a bed array lives on
    another device than the current one: this helper never changes the current device or any pool limit."""
    from maple.core.backend import array_namespace

    xp = array_namespace(state.water.depth_m)
    require_current_device(int(xp.cuda.Device().id),
                           {"water.depth_m": int(state.water.depth_m.device.id),
                            "voxel_column.window_kg": int(state.voxel_column.window_kg.device.id)})

    def query() -> int:
        pool = xp.get_default_memory_pool()
        return effective_free_device_bytes(int(xp.cuda.Device().mem_info[0]), int(pool.free_bytes()),
                                           int(pool.get_limit()), int(pool.used_bytes()))

    return query


def _admit(state, context, *, path, features, reserve_host_bytes, reserve_device_bytes, safety_fraction,
           max_gather_columns, host_available_bytes_fn, device_available_bytes_fn) -> AdmissionRecord:
    _require_source()  # first: no optional native module is imported for an unsupported source
    reserve_host_bytes = _checked_int(reserve_host_bytes, "reserve_host_bytes")
    reserve_device_bytes = _checked_int(reserve_device_bytes, "reserve_device_bytes", allow_none=True)
    device = _namespace_name(state) == "cupy"
    if device and reserve_device_bytes is None:
        raise StorageAdmissionError("a device bed needs an explicit reserve_device_bytes (no default is invented)")
    if not device:
        if reserve_device_bytes:  # None and an explicit integer 0 are the only host values
            raise StorageAdmissionError("reserve_device_bytes is meaningless for a host bed (refused, not ignored)")
        if device_available_bytes_fn is not None:
            raise StorageAdmissionError("device_available_bytes_fn was given for a host bed: it would never be consulted "
                                        "(refused before any query, not ignored)")
        reserve_device_bytes = 0
    plan = plan_for_bed(state, context, path=path, features=features, max_gather_columns=max_gather_columns)
    from maple.surface.voxels.windowed_admission import admit_windowed_transaction

    if device and device_available_bytes_fn is None:
        device_available_bytes_fn = _default_device_query(state)
    decision = admit_windowed_transaction(
        plan, host_available_bytes_fn=host_available_bytes_fn,
        device_available_bytes_fn=device_available_bytes_fn if device else None, safety_fraction=safety_fraction,
        reserve_host_bytes=reserve_host_bytes, reserve_device_bytes=reserve_device_bytes)
    if not decision.admitted:
        raise MemoryError(decision.reason)
    return AdmissionRecord(plan, decision, path, plan.namespace, plan.inputs["max_gather_columns"],
                           plan.inputs["max_journal_records"], reserve_host_bytes, reserve_device_bytes)


def admit_public(state: BedState, context: BedContext, demand, *, reserve_host_bytes: int,
                 reserve_device_bytes: int | None = None, safety_fraction=0.5, max_gather_columns: int | None = None,
                 host_available_bytes_fn=None, device_available_bytes_fn=None) -> AdmissionRecord:
    """Admission before ONE `sediment_bed.apply_bed_demand` on a windowed bed; the flags come from the actual `demand`.
    Raises `MemoryError(reason)` on refusal, before any exchange."""
    return _admit(state, context, path="public", features=DemandFeatures.from_demand(demand),
                  reserve_host_bytes=reserve_host_bytes, reserve_device_bytes=reserve_device_bytes,
                  safety_fraction=safety_fraction, max_gather_columns=max_gather_columns,
                  host_available_bytes_fn=host_available_bytes_fn, device_available_bytes_fn=device_available_bytes_fn)


def admit_paired(state: BedState, context: BedContext, *, features: DemandFeatures, reserve_host_bytes: int,
                 reserve_device_bytes: int | None = None, safety_fraction=0.5, max_gather_columns: int | None = None,
                 host_available_bytes_fn=None, device_available_bytes_fn=None) -> AdmissionRecord:
    """Admission before ONE `sediment_bed.apply_bed_pickup_transport_deposit`. `features` is the CALLER'S declaration of
    the flags OR-ed over the pickup and the deposit demand (the deposit demand is not available yet; this is not inferred
    or validated). The admitted figure always covers MAPLE's allocating fallback. Raises `MemoryError` on refusal."""
    return _admit(state, context, path="paired", features=features, reserve_host_bytes=reserve_host_bytes,
                  reserve_device_bytes=reserve_device_bytes, safety_fraction=safety_fraction,
                  max_gather_columns=max_gather_columns, host_available_bytes_fn=host_available_bytes_fn,
                  device_available_bytes_fn=device_available_bytes_fn)
