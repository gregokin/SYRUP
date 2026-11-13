"""Phase 2 E2: read-only COMPACT soil-profile texture batches for explicit changed-cell indices, plus a cold canonical depth.

What this is
------------
`iter_compact_soil_profile(...)` returns a one-shot iterator of `CompactBatch` objects, each covering at most `B <= C` of the
caller's SELECTED flat cells (`cell = iy * nx + ix`, sorted unique int64 indices). A batch carries genuine compact arrays
`layer_mass_kg (L, B, K)`, `layer_fraction (L, B, K)`, `layer_total_kg (L, B)`, `coverage_m (L, B)`, the sequential
`represented_depth_m (B,)` / `active_cap_thickness_m (B,)` diagnostics of S1, the exact selected indices and the COLD
canonical depth `canonical_depth_m (B,)` (host float64). No `(L, N, K)` tensor, no whole-domain output and no second sediment
bed exist: every number is computed by the EXISTING S1 batch engines (`soil_profile_texture.reference_texture_batch`, the
Numba and CUDA runners, selected through `soil_profile_texture._runner`) on rows gathered from the actual native bed, after the
actual MAPLE validators have accepted them. The physical engines, validators and gather are reused, not copied.

Selection and validation (before any gather)
--------------------------------------------
`indices` must be a 1-D `numpy.ndarray` of dtype int64, strictly increasing (sorted unique), within `[0, N)`, and every selected
cell must be ACTIVE in the caller's `(ny, nx)` bool mask (inactive cells are excluded by omission, never zero-filled). An empty
selection is a genuine no-op: after the cheap argument checks nothing is validated, gathered, hashed or allocated and the
iterator yields nothing. Dense: MAPLE's partition validator runs on each selected batch only (honest scope: it does not prove
unselected cells). Windowed: the FULL native trusted-boundary validation of the whole store (`require_published_column`, cheap
checks, active layer, `validate_windowed_column_full`) always runs once at construction; its O(whole store) cost is explicit
and is not reduced by a small selection. An unpublished pending journal is REJECTED (`require_published_column`); this rejection
does not complete native event integration.

Canonical depth (cold witness, NOT the S1 sequential depth and NOT a hot near-rock algorithm)
---------------------------------------------------------------------------------------------
From the batch's actual gathered masses: voxel total = NumPy contiguous FP64 `sum(axis=(-2, -1))` over `(B, V, K)`; native
`_elevation_from_total_mass(total, rho, dx*dy)` = `(total / rho) / area`; active thickness = native
`compute_active_layer_actual_thickness_m` (`sum(axis=-1) / (rho * area)`); `canonical = voxel + active` -- the exact expression of
`diagnose_combined_surface_state(..., fixed_base=None).combined_elevation_m`, verified bitwise by the tests against that
diagnostic. For dense device columns the selected `(B, V, K)` rows and `(B, K)` active rows are downloaded once per batch
through MAPLE's counted `to_host` (declared bounded cold transfer, admitted in the estimate); it is never a per-hydraulic-step
full-column download. The metadata binds algorithm, NumPy version and the order id of `maple.core.backend.numpy_pairwise_order`.

Memory (metadata first; see `CompactMemoryEstimate`)
----------------------------------------------------
Every term is B-sized (or C-sized for the selection) except the host mask copy (N bytes, caller's mask metadata) and the
unchanged native validation/cache bounds. The base terms (`scratch`, `validation`, windowed `batch_input`, native cache, fixed,
metadata) are those of `soil_profile_texture.estimate_texture_memory` at the same `B`; its `(L, N, K)` output/staging and
`2N`/`8N` mask/index terms are REPLACED by the compact terms below. A defaulted batch shrinks to fit the explicit budget and the
reduction is recorded; an explicit batch is never changed. No B-, C- or N-sized array is allocated before admission (only
O(L) layer-geometry parsing and O(1) metadata; the empty selection skips even that). A reading is not a
reservation; imports, JIT, CUDA context and pool rounding are excluded; logical provenance hashes (selection digest) are
descriptive and distinct from physical content hashes.

Consumer contract
-----------------
The iterator keeps NO reference to a yielded batch and no list or history. The caller owns each `CompactBatch`; the stated peak
covers ONE live batch, so the consumer must consume/release it before requesting the next (retained batches are caller-owned and
excluded from the component's peak). Value-dependent refusals (insufficient depth under `coverage="full"`, cap thicker than the
top layer, native validation of a selected batch) raise at the offending batch during iteration; nothing is mutated anywhere
and batches already yielded are the caller's.

Not in scope: property computation, water, dirty/generation detection, Pond/lifecycle/restart/CLI, hot depth classification,
saturation of any physical claim beyond reading the native bed. Author-side: CPU/host only; the CUDA paths are defined for Root.
"""

from __future__ import annotations

import dataclasses
import hashlib
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np

from maple_syrup import soil_profile_texture as spt
from maple_syrup.soil_profile_texture import (
    COVERAGE_POLICIES,
    FLAG_INSUFFICIENT_DEPTH,
    FLAG_TOP_LAYER_THINNER_THAN_CAP,
    SoilProfileGeometryError,
    SoilProfileMemoryError,
    SoilProfileTextureError,
)

__all__ = [
    "CANONICAL_DEPTH_ALGORITHM",
    "SCHEMA",
    "CompactBatch",
    "CompactBatchIterator",
    "CompactMemoryEstimate",
    "canonical_depth_m",
    "estimate_compact_memory",
    "iter_compact_soil_profile",
]

SCHEMA = "maple_syrup.compact_soil_profile/1"
CANONICAL_DEPTH_ALGORITHM = ("canonical_cold_depth/1: voxel_total=numpy_contiguous_fp64_sum(axis=(-2,-1)); "
                             "(total/rho)/area + active_total/(rho*area) via the native diagnostic helpers; fixed_base=None")
_F8 = 8


# --- canonical depth -----------------------------------------------------------------------------------------------------
def canonical_depth_m(voxel_rows: np.ndarray, active_rows: np.ndarray, geometry: Any) -> np.ndarray:
    """Cold canonical depth `(B,)` of host rows `voxel_rows (B, V, K)` (C-contiguous float64) and `active_rows (B, K)` with the
    native arithmetic (see the module docstring). Pure host function; inputs must be EXACT `numpy.ndarray` objects (no
    subclass/masked array). Live `(B,)` float64 rows are at most 3 at any time (the voxel total is released right after the
    native voxel conversion; the arithmetic is not reassociated). It uses the actual native NumPy reduction on the host: it is
    NOT the fixed-tree GPU reduction of the hot path and is subject to none of its limits (e.g. the 32768-lane bound)."""
    from maple.core.types.active_layer import ActiveLayerState
    from maple.surface.active_layer.capacity import (
        compute_active_layer_actual_thickness_m,
    )
    from maple.surface.voxels.diagnostics import _elevation_from_total_mass

    if not (type(voxel_rows) is np.ndarray and voxel_rows.ndim == 3 and voxel_rows.dtype == np.float64
            and voxel_rows.flags.c_contiguous):
        raise SoilProfileTextureError("canonical depth needs host C-contiguous float64 voxel rows (B, V, K)")
    if not (type(active_rows) is np.ndarray and active_rows.ndim == 2 and active_rows.dtype == np.float64
            and active_rows.flags.c_contiguous and active_rows.shape[0] == voxel_rows.shape[0]
            and active_rows.shape[1] == voxel_rows.shape[2]):
        raise SoilProfileTextureError("canonical depth needs host C-contiguous float64 active rows (B, K) matching the voxel rows")
    B, K = int(active_rows.shape[0]), int(active_rows.shape[1])
    batch_geometry = dataclasses.replace(geometry, ny=1, nx=B)
    total = voxel_rows.sum(axis=(-2, -1))
    voxel_m = _elevation_from_total_mass(total, geometry.bulk_density_kg_m3, geometry.dx_m * geometry.dy_m)
    del total
    thickness = compute_active_layer_actual_thickness_m(ActiveLayerState(mass_kg=active_rows.reshape(1, B, K)), batch_geometry)
    return voxel_m + thickness.reshape(B)


# --- records -------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class CompactBatch:
    """At most `B` selected cells. Arrays live in `namespace` (the native state's own), except the host `indices` and
    `canonical_depth_m`. `indices` are the exact ORIGINAL flat cell indices (sorted, unique, int64)."""

    indices: np.ndarray  # (B,) int64 host, original flat indices
    batch_index: int
    layer_mass_kg: Any  # (L, B, K)
    layer_fraction: Any  # (L, B, K)
    layer_total_kg: Any  # (L, B)
    coverage_m: Any  # (L, B)
    represented_depth_m: Any  # (B,) S1 sequential diagnostic (NOT the canonical depth)
    active_cap_thickness_m: Any  # (B,)
    canonical_depth_m: np.ndarray  # (B,) host float64, cold canonical native-order depth
    namespace: str
    source: dict[str, Any]  # the iterator's descriptive record (shared, read-only by convention)

    @property
    def n_cells(self) -> int:
        return int(self.indices.shape[0])


@dataclass(frozen=True)
class CompactMemoryEstimate:
    """Upper-bound bytes this call allocates (caller-owned native arrays and the caller's index array excluded). Field names that
    the S1 partial-validation admission reads (`batch_cells`, `validation_bytes`, `representation`, `output_namespace`,
    `fixed_overhead_bytes`) are kept."""

    representation: str
    backend: str
    output_namespace: str
    n_cells: int
    n_selected: int
    n_layers: int
    n_classes: int
    n_voxels: int
    batch_cells: int
    output_bytes: int  # ONE compact batch of outputs in the output namespace
    output_staging_bytes: int  # host staging of those outputs (windowed with a device window)
    witness_bytes: int  # canonical-depth host rows/results (dense device: B*V*K + B*K downloaded)
    index_mask_bytes: int  # host mask copy N + host/device per-batch index temporaries
    selection_bytes: int  # sealed immutable int64 selection copy (8C, resident) + B-chunk validation scratch (4B)
    gather_bytes: int  # dense: selected row copy (B,V,K)+(B,K); windowed: native gather of one batch (base term)
    validation_bytes: int
    scratch_bytes: int
    native_cache_bytes: int
    fixed_overhead_bytes: int
    metadata_bytes: int
    reserve_bytes: int
    host_bytes: int
    device_bytes: int
    required_bytes: int
    input_bytes_excluded: int
    batch_reduced: bool = False
    metadata_only: bool = False


def estimate_compact_memory(*, representation: str, backend: str, output_namespace: str, n_cells: int, n_selected: int,
                            n_layers: int, n_classes: int, n_voxels: int, batch_cells: int, window_voxels: int = 0,
                            chunk_cells: int = 0, native_cache_bytes: int = 0, reserve_bytes: int = 0,
                            input_bytes_excluded: int = 0, batch_reduced: bool = False,
                            metadata_only: bool = False) -> CompactMemoryEstimate:
    """Pure arithmetic. Compact terms replace the S1 `(L, N, K)` output/staging and `N`-scale index terms; every other S1 term
    (scratch, validators, native gather/cache, fixed, metadata) is reused at the same `B`."""
    N = spt._checked_int(n_cells, "n_cells", minimum=1)
    C = spt._checked_int(n_selected, "n_selected")
    L, K, V = (spt._checked_int(v, nme, minimum=1) for v, nme in ((n_layers, "n_layers"), (n_classes, "n_classes"),
                                                                    (n_voxels, "n_voxels")))
    B = spt._checked_int(batch_cells, "batch_cells", minimum=1)
    base = spt.estimate_texture_memory(
        representation=representation, backend=backend, output_namespace=output_namespace, n_cells=N,
        n_active=max(C, 1), n_layers=L, n_classes=K, n_voxels=V, batch_cells=B, window_voxels=window_voxels,
        chunk_cells=chunk_cells, native_cache_bytes=native_cache_bytes, reserve_bytes=reserve_bytes,
        input_bytes_excluded=input_bytes_excluded, batch_reduced=batch_reduced, metadata_only=metadata_only)
    out = (2 * L * B * K + 2 * L * B + 2 * B) * _F8  # mass, fraction, total, coverage, depth, cap of one batch
    selection = _F8 * C + 4 * B  # sealed selection (resident for the whole iteration) + order/bounds/activity chunk scratch
    device_ns = output_namespace == "cupy"
    staging = out if (representation == "windowed" and device_ns) else 0
    indices_b = 8 * B  # the per-batch int64 index slice (host view/copy)
    if representation == "dense":
        gather = (B * V * K + B * K) * _F8  # selected rows in the state's namespace
        witness = (B * _F8) * 3 + ((B * V * K + B * K) * _F8 if device_ns else 0)  # host rows only when downloaded; results O(B)
        idx = N + indices_b + (indices_b if device_ns else 0)  # host mask copy + host slice (+ device index copy)
        if device_ns:
            device = out + gather + base.validation_bytes + base.scratch_bytes + indices_b
            host = idx + witness + selection + base.fixed_overhead_bytes + base.metadata_bytes
        else:
            device = 0
            host = out + gather + witness + idx + selection + base.validation_bytes + base.scratch_bytes + base.fixed_overhead_bytes + base.metadata_bytes
    else:
        gather = base.batch_input_bytes  # native gather of one batch (host), already B-sized
        witness = B * _F8 * 3  # canonical results; the gathered host rows ARE the gather term
        idx = N + indices_b + (indices_b if device_ns else 0)
        device = (out + indices_b) if device_ns else 0
        host = (staging if device_ns else out) + gather + witness + idx + selection + base.validation_bytes + base.scratch_bytes \
            + base.native_cache_bytes + base.fixed_overhead_bytes + base.metadata_bytes
    required = host + device + base.reserve_bytes
    return CompactMemoryEstimate(
        representation, backend, output_namespace, N, C, L, K, V, B, out, staging, witness, idx, selection, gather,
        base.validation_bytes, base.scratch_bytes, base.native_cache_bytes, base.fixed_overhead_bytes, base.metadata_bytes,
        base.reserve_bytes, host, device, required, base.input_bytes_excluded, batch_reduced, metadata_only)


def _admit(make, budget: int, batch: int, explicit: bool, reduced: bool = False) -> CompactMemoryEstimate:
    while True:
        estimate = make(batch, reduced)
        if estimate.required_bytes <= budget:
            return estimate
        if explicit or batch <= 1:
            stage = "metadata-only lower bound" if estimate.metadata_only else "final estimate"
            raise SoilProfileMemoryError(
                f"compact soil profile ({estimate.representation}, {estimate.backend}; {stage}) needs an estimated "
                f"{estimate.required_bytes} B (outputs {estimate.output_bytes}, staging {estimate.output_staging_bytes}, witness "
                f"{estimate.witness_bytes}, index/mask {estimate.index_mask_bytes}, selection {estimate.selection_bytes}, gather {estimate.gather_bytes}, validation "
                f"{estimate.validation_bytes}, scratch {estimate.scratch_bytes}, native cache {estimate.native_cache_bytes}, "
                f"fixed {estimate.fixed_overhead_bytes}, metadata {estimate.metadata_bytes}, reserve {estimate.reserve_bytes}) at "
                f"batch_cells={batch}, above the caller budget {budget} B; nothing was gathered and the budget is not raised.")
        batch = max(1, batch // 2)
        reduced = True


def _check_indices_metadata(indices: Any) -> None:
    """Cheap metadata check only (type, dtype, rank): no value scan, no copy."""
    if type(indices) is not np.ndarray or indices.dtype != np.int64 or indices.ndim != 1:
        raise SoilProfileTextureError("indices must be a 1-D numpy int64 array of selected flat cell indices")


def _seal_selection(indices: np.ndarray, n_cells: int, mask: np.ndarray, batch: int) -> tuple[np.ndarray, str]:
    """Own an immutable copy of the caller's selection (8C bytes, counted) and, in chunks of `batch` cells, validate order
    (strictly increasing, also across chunk boundaries), bounds and activity on that copy while streaming its SHA-256 digest.
    Transient per chunk: the strict-order bool and the activity bool (counted as 4B); the digest reads a contiguous view."""
    sealed = indices.copy()
    sealed.setflags(write=False)
    digest = hashlib.sha256()
    prev = -1
    for b0 in range(0, sealed.size, max(1, batch)):
        chunk = sealed[b0:b0 + batch]
        if int(chunk[0]) <= prev or (chunk.size > 1 and not bool(np.all(chunk[1:] > chunk[:-1]))):
            raise SoilProfileTextureError("indices must be strictly increasing (sorted, unique)")
        prev = int(chunk[-1])
        if int(chunk[0]) < 0 or prev >= n_cells:
            raise SoilProfileTextureError(f"indices must lie within [0, {n_cells}) (sorted unique, int64)")
        if not bool(mask[chunk].all()):
            raise SoilProfileTextureError("every selected cell must be active in active_mask (inactive cells are excluded by omission)")
        digest.update(memoryview(chunk))
    return sealed, digest.hexdigest()


# --- device guard (no switching, no implicit cross-device work) ---------------------------------------------------------
def _device_ids(arrays: list) -> set[int]:
    return {int(a.device.id) for a in arrays}


def _current_device(xp: Any) -> int:
    return int(xp.cuda.runtime.getDevice())


def _guard_device(namespace: str, xp: Any, arrays: list) -> int | None:
    """For a CuPy namespace: every participating device array (state, active layer, a device mask) must live on ONE device that
    is the CURRENT device; otherwise refuse before any copy, validator, gather or scalar read. Never switches devices.
    Returns the source device id (None on the host)."""
    if namespace != "cupy":
        return None
    ids = _device_ids(arrays)
    if len(ids) != 1:
        raise SoilProfileTextureError(f"native arrays live on different devices {sorted(ids)}; no implicit cross-device work")
    (dev,) = ids
    cur = _current_device(xp)
    if cur != dev:
        raise SoilProfileTextureError(f"native arrays live on device {dev} but the current device is {cur}; the device is never switched")
    return dev


# --- iterator ------------------------------------------------------------------------------------------------------------
class CompactBatchIterator:
    """One-shot iterator of `CompactBatch`; holds no yielded batch. `source` / `memory` are available before the first batch."""

    def __init__(self, gen: Any, source: dict[str, Any] | None, memory: CompactMemoryEstimate | None, n_selected: int) -> None:
        self._gen, self.source, self.memory, self.n_selected = gen, source, memory, n_selected

    def __iter__(self) -> CompactBatchIterator:
        return self

    def __next__(self) -> CompactBatch:
        return next(self._gen)

    def close(self) -> None:
        self._gen.close()


def iter_compact_soil_profile(
    voxel_column: Any, active_layer: Any, geometry: Any, grain_classes: Any, mass_resolution_kg: float, layers: Any, *,
    indices: Any, active_mask: Any, memory_budget_bytes: int, backend: str = "numpy", reserve_bytes: int = 0,
    batch_cells: int | None = None, coverage: str = "full",
) -> CompactBatchIterator:
    """Compact batches of the selected cells of one native bed on fixed layers (see the module docstring). All argument,
    representation, mask, selection and memory-admission checks (and, windowed, the full native validation) happen HERE,
    before the first batch; per-batch native validation, gather and compute happen lazily during iteration."""
    if backend not in spt.BACKENDS:
        raise SoilProfileTextureError(f"backend must be one of {spt.BACKENDS}, got {backend!r}")
    if type(coverage) is not str or coverage not in COVERAGE_POLICIES:
        raise SoilProfileTextureError(f"coverage must be one of {COVERAGE_POLICIES}, got {coverage!r}")
    _check_indices_metadata(indices)
    try:
        ny, nx = int(geometry.ny), int(geometry.nx)
    except (AttributeError, TypeError, ValueError) as exc:
        raise SoilProfileTextureError(f"geometry is not a MAPLE GeometrySpec: {exc}") from exc
    representation = spt._representation(voxel_column, active_layer)
    spt._check_mask_metadata(active_mask, ny, nx)
    if indices.size == 0:  # genuine no-op after the cheap metadata checks: no layers, budget, validation, gather, hash, allocation
        return CompactBatchIterator(iter(()), None, None, 0)
    layers = spt.canonical_layers(layers)
    budget = spt._checked_int(memory_budget_bytes, "memory_budget_bytes", minimum=1)
    reserve = spt._checked_int(reserve_bytes, "reserve_bytes")
    explicit = batch_cells is not None
    if explicit:
        batch_cells = spt._checked_int(batch_cells, "batch_cells", minimum=1)
    try:
        mres = float(mass_resolution_kg)
    except (TypeError, ValueError) as exc:
        raise SoilProfileTextureError(f"mass_resolution_kg must be a real number: {exc}") from exc
    if not np.isfinite(mres) or mres < 0.0:
        raise SoilProfileTextureError(f"mass_resolution_kg must be finite and >= 0, got {mass_resolution_kg!r}")
    ny, nx, dx, dy, rho, dz, cap_target = spt._geometry_numbers(geometry)
    N = ny * nx
    from maple.core.backend import (
        array_namespace,
        is_array,
        namespace_name,
        to_device,
        to_host,
    )

    active_kg = getattr(active_layer, "mass_kg", None)
    if not is_array(active_kg) or active_kg.ndim != 3 or active_kg.dtype != np.float64:
        raise SoilProfileTextureError("active_layer.mass_kg must be a float64 (ny, nx, n_classes) array")
    if tuple(active_kg.shape[:2]) != (ny, nx):
        raise SoilProfileTextureError(f"active_layer.mass_kg grid {tuple(active_kg.shape[:2])} != geometry {(ny, nx)}")
    K = int(active_kg.shape[2])
    class_ids, class_diam, class_agg, class_rho = spt._class_metadata(grain_classes, K)
    L = layers.n_layers
    denominator = rho * (dx * dy)
    C = int(indices.size)
    windowed = representation == "windowed"
    batch0 = batch_cells if explicit else max(1, min(C, spt.DEFAULT_BATCH_CELLS))

    if windowed:
        version, header, V, W, CC, cache_bytes = spt._windowed_metadata(voxel_column, K)
        xp = array_namespace(voxel_column.window_kg, voxel_column.base, active_kg)
        namespace = namespace_name(xp)
        spt._check_backend_namespace(backend, "", "windowed")
        excluded = (int(voxel_column.window_kg.nbytes) + int(voxel_column.base.nbytes) + int(voxel_column.deep_by_class_kg.nbytes)
                    + int(voxel_column.deep_total_kg.nbytes) + int(voxel_column.deep_room_kg.nbytes) + int(active_kg.nbytes))
        extra = {"window_voxels": W, "chunk_cells": CC, "native_cache_bytes": cache_bytes}
    else:
        voxels = getattr(voxel_column, "mass_kg", None)
        if not is_array(voxels) or voxels.ndim != 4 or voxels.dtype != np.float64:
            raise SoilProfileTextureError("voxel_column.mass_kg must be a float64 (ny, nx, n_voxels, n_classes) array")
        if tuple(voxels.shape[:2]) != (ny, nx) or int(voxels.shape[3]) != K:
            raise SoilProfileTextureError("voxel_column.mass_kg grid/class axes disagree with the geometry/active layer")
        if not (spt._c_contiguous(voxels) and spt._c_contiguous(active_kg)):
            raise SoilProfileTextureError("dense native arrays must be C-contiguous")
        xp = array_namespace(voxels, active_kg)
        namespace = namespace_name(xp)
        spt._check_backend_namespace(backend, namespace, "dense")
        V = int(voxels.shape[2])
        excluded = int(voxels.nbytes) + int(active_kg.nbytes)
        extra = {}

    guarded = ([voxel_column.window_kg, voxel_column.base, voxel_column.deep_by_class_kg, voxel_column.deep_total_kg,
                voxel_column.deep_room_kg, active_kg] if windowed else [voxels, active_kg])
    if namespace_name(array_namespace(active_mask)) == "cupy":
        if namespace != "cupy":  # metadata-only policy check, before any transfer
            raise SoilProfileTextureError(
                "a device-resident active_mask is not supported with a CPU-resident native source; supported masks are host "
                "bool arrays, or a device mask on the SAME device as a device-resident source (no implicit transfer)")
        guarded.append(active_mask)  # a device mask participates: same device, no implicit transfer
    device_id = _guard_device(namespace, xp, guarded)  # BEFORE admission reads, the mask copy, validators, gather, scalar reads

    def make(batch, reduced, metadata_only=False):
        return estimate_compact_memory(
            representation=representation, backend=backend, output_namespace=namespace, n_cells=N, n_selected=C,
            n_layers=L, n_classes=K, n_voxels=V, batch_cells=batch, reserve_bytes=reserve, input_bytes_excluded=excluded,
            batch_reduced=reduced, metadata_only=metadata_only, **extra)

    estimate = _admit(lambda b, r: make(b, r), budget, batch0, explicit)  # metadata only, BEFORE the mask copy / any native work
    B = estimate.batch_cells
    runner = spt._runner(backend)
    mask = spt._host_mask(active_mask, ny, nx)
    indices, indices_digest = _seal_selection(indices, N, mask, B)  # owned immutable copy, chunked validation + streamed digest
    del mask  # the host mask copy is not retained: activity was proven on the selection

    operation = "iter_compact_soil_profile(" + representation + ")"
    n_terms = spt.coverage_bound_terms(V, K, L)
    factor = spt.coverage_bound_factor(V, K, L)
    if windowed:
        from maple.surface.active_layer.validation import check_active_layer_state
        from maple.surface.voxels.windowed_inventory import (
            check_windowed_column_cheap,
            gather_full_columns,
            validate_windowed_column_full,
        )
        from maple.surface.voxels.windowed_pending import require_published_column

        try:  # native source checks, never skipped or downgraded; an unpublished journal is refused here
            require_published_column(voxel_column, operation=operation)
            check_windowed_column_cheap(voxel_column, geometry, class_ids=class_ids)
            check_active_layer_state(spt._as_active(active_kg), geometry, mres)
            validate_windowed_column_full(voxel_column, geometry, mres, class_ids=class_ids)
        except (TypeError, ValueError) as exc:
            raise spt._wrap_native(operation, exc) from exc

    import numpy

    try:  # required native order metadata: absent => typed setup/source failure, never a weakened provenance record
        from maple.core.backend import numpy_pairwise_order as _po
        order_id = str(_po.ORDER_ID)
    except (ImportError, AttributeError) as exc:
        raise SoilProfileTextureError(f"the pinned native source lacks the required numpy_pairwise_order.ORDER_ID: {exc}") from exc
    source = {
        "schema": SCHEMA, "kind": "descriptive_identity_of_inputs_read_at_build", "not_a_runtime_source_lock": True,
        "result_kind": "compact_selected_columns", "whole_grid": False, "representation": representation,
        "namespace": namespace, "device_id": device_id, "backend": backend, "coverage_policy": coverage, "native": spt._native_identity(representation),
        "grid": {"ny": ny, "nx": nx, "layout": "flat cell = iy * nx + ix"},
        "geometry": {"dx_m": dx, "dy_m": dy, "voxel_dz_m": dz, "bulk_density_kg_m3": rho,
                     "active_layer_thickness_target_m": cap_target, "cell_area_m2": dx * dy, "n_voxels": V},
        "classes": [{"class_id": i, "diameter_m": d, "is_aggregate": a, "particle_density_kg_m3": r}
                    for i, d, a, r in zip(class_ids, class_diam, class_agg, class_rho)],
        "layers": {"thickness_m": [float(t) for t in layers.thickness_m], "depth_edges_m": [float(e) for e in layers.depth_edges_m]},
        "selection": {"n_selected": C, "indices_sha256": indices_digest,
                      "provenance": "logical selection digest of the sorted unique int64 original flat indices (not a physical hash)",
                      "n_batches": (C + B - 1) // B, "batch_cells": B, "batch_reduced": estimate.batch_reduced,
                      "explicit_batch": explicit},
        "coverage_bound": {"n_terms": n_terms, "factor": factor},
        "validation": ("native: require_published_column, cheap checks, active layer, validate_windowed_column_full (whole store, "
                       "always), dense partition validator per selected batch" if windowed else
                       "native partition validator per SELECTED batch only (unselected cells are not proven)"),
        "canonical_depth": {"algorithm": CANONICAL_DEPTH_ALGORITHM, "numpy_version": numpy.__version__, "numpy_pairwise_order_id": order_id,
                            "distinct_from": "represented_depth_m (S1 sequential segment-height diagnostic)",
                            "transfer": "dense device: selected (B,V,K)+(B,K) rows downloaded per batch with counted to_host (cold)",
                            "reduction": "actual native NumPy on the host (not the fixed-tree GPU reduction; no 32768-lane bound applies)"},
        "memory": dataclasses.asdict(estimate),
    }
    if windowed:
        source["windowed"] = {"deep_store_version_id": int(version.version_id), "deep_store_manifest_digest": str(version.manifest_digest),
                              "deep_store_header_digest": header.digest(), "generation": int(voxel_column.generation),
                              "window_voxels": W, "chunk_columns": CC, "resident_floor_is_not_bedrock": True}
    source["not_claimed"] = ["hot near-rock depth classification", "property refresh", "water conservation", "native integration"]

    edges = layers.depth_edges_m
    active_flat = active_kg.reshape(N, K)
    voxels_flat = None if windowed else voxels.reshape(N, V, K)

    def generate() -> Iterator[CompactBatch]:
        for bi, b0 in enumerate(range(0, C, B)):
            idx = indices[b0:b0 + B]
            n = int(idx.shape[0])
            if windowed:
                v_host = gather_full_columns(voxel_column, idx, max_columns=B)  # fresh host (n, V, K)
                a_host = np.ascontiguousarray(to_host(active_flat[idx if namespace == "numpy" else to_device(idx, xp)]))
                spt._validate_dense_batch(a_host, v_host, geometry, mres, operation)
                result = runner(np, a_host, v_host, np.ones(n, dtype=np.bool_), edges, denominator, factor)
                depth = canonical_depth_m(v_host, a_host, geometry)
                del v_host, a_host
            else:
                idx_ns = idx if namespace == "numpy" else to_device(idx, xp)
                v_rows = voxels_flat[idx_ns]
                a_rows = active_flat[idx_ns]
                spt._validate_dense_batch(a_rows, v_rows, geometry, mres, operation)
                result = runner(xp, a_rows, v_rows, xp.ones(n, dtype=np.bool_), edges, denominator, factor)
                if namespace == "numpy":
                    depth = canonical_depth_m(np.ascontiguousarray(v_rows), np.ascontiguousarray(a_rows), geometry)
                else:  # declared bounded cold download of the SELECTED rows (counted)
                    depth = canonical_depth_m(np.ascontiguousarray(to_host(v_rows)), np.ascontiguousarray(to_host(a_rows)), geometry)
                del v_rows, a_rows, idx_ns
            mass, fraction, total, coverage_m, rep_depth, cap, flags = result
            flags_host = np.asarray(to_host(flags))
            n_cap = int(np.count_nonzero(flags_host & FLAG_TOP_LAYER_THINNER_THAN_CAP))
            if n_cap:
                raise SoilProfileGeometryError(
                    f"{n_cap} selected cell(s) in batch {bi} have an active-layer cap thicker than the top soil layer "
                    f"({float(layers.thickness_m[0])} m) beyond the declared bound; nothing was published for this batch.")
            if coverage == "full":
                n_short = int(np.count_nonzero(flags_host & FLAG_INSUFFICIENT_DEPTH))
                if n_short:
                    raise SoilProfileGeometryError(
                        f"{n_short} selected cell(s) in batch {bi} do not represent enough actual material to cover the layers "
                        f"down to {layers.bottom_depth_m} m; no material is invented (use coverage='partial_over_base' for "
                        "explicit partial coverage); nothing was published for this batch.")
            else:
                spt._admit_partial_stage(estimate)
                outputs = {"coverage_m": coverage_m, "represented_depth_m": rep_depth}
                if spt._partial_coverage_inconsistent(outputs, np.ones(n, dtype=np.bool_), layers, factor, n):
                    raise SoilProfileGeometryError(
                        f"selected cell(s) in batch {bi} have non-finite/negative depth or coverage, or coverage inconsistent "
                        "with the actual material depth beyond the declared bound; nothing was published for this batch.")
            if windowed and namespace == "cupy":  # reduced outputs placed in the resident namespace (counted)
                mass, fraction, total, coverage_m, rep_depth, cap = (to_device(a, xp) for a in (mass, fraction, total, coverage_m, rep_depth, cap))
            yield CompactBatch(indices=idx.copy(), batch_index=bi, layer_mass_kg=mass, layer_fraction=fraction, layer_total_kg=total,
                               coverage_m=coverage_m, represented_depth_m=rep_depth, active_cap_thickness_m=cap,
                               canonical_depth_m=depth, namespace=namespace, source=source)
            del mass, fraction, total, coverage_m, rep_depth, cap, result, depth

    return CompactBatchIterator(generate(), source, estimate, C)
