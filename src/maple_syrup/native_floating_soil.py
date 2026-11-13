"""Phase 2 E3: native bed -> floating-soil water owner, initialization and the staged changed-column REFRESH transaction (CPU/host; NumPy and Numba floating owners).

What this is
------------
A narrow orchestration layer that connects four ACCEPTED components without re-implementing any of them: the native MAPLE bed (Dense `VoxelColumnState`/`ActiveLayerState` or genuine resident-window
`WindowedColumn`), the E2 compact reader (`compact_soil_profile.iter_compact_soil_profile`: actual mass-overlap layer texture and the cold canonical depth of explicit cells), the pure Saxton-Rawls property
provider (`soil_texture_properties.estimate_soil_texture_properties`) and the D1/D2 floating column, zero-time reconciliation and persistent matrix history (`floating_soil_layers`,
`floating_matrix_infiltration`). The bed stays the only sediment authority: nothing here stores sediment, and the owner holds ONE water inventory.

    owner = initialize_native_floating_soil(bed, config, initial_relative_saturation=..., memory_budget_bytes=...)
    cand  = stage_native_refresh(owner, parent_bed, child_bed, mode="properties"|"depth", memory_budget_bytes=...)   # PURE: nothing is adopted
    owner2 = adopt_native_refresh(owner, cand)                                                                      # the one mutation point; owner/cand become stale

Explicit inputs (`native_floating_config`)
------------------------------------------
Fixed surface-following nominal layer thicknesses; an EXPLICIT grain-class -> (sand, silt, clay) composition mapping `(K, 3)` (rows sum to 1; NEVER guessed from a grain diameter), organic matter percent
`(L,)`, the bulk density (checked against the actual MAPLE packing: `geometry.bulk_density_kg_m3` must equal it, otherwise a typed `NativePackingError`; nothing is adjusted), explicit latent hydraulic properties
`(L,)` that a zero-mass slot RETAINS (field capacity and saturation; a zero-mass slot never receives a primary soil texture), the explicit residual content, evaporation stress/factor, constant interface
diffusivity `(L-1,)`, the lower-boundary mode, the explicit Green-Ampt matrix inputs (conductivity, suction, vegetation factor) and the initial persistent history, optional measured overrides `(L, N)` (NaN = not
measured, passed through the provider). Thin/zero coverage uses the E1 `partial_over_base` policy (a layer holding any actual material has a texture; a layer with exactly zero mass is latent).

Geometry and depth
------------------
`D` is the COLD canonical native depth of the compact reader (native NumPy order, full flat base 0; NOT the S1 sequential diagnostic). The profile bottom is open only when `D > H` (D == H is closed rock; no epsilon);
partial and zero depths are the D1 semantics. Layers keep their nominal thickness measured from the surface; the per-layer inventory is held through reassignment and the zero-time reconciliation of D1
(field-capacity cascade, saturation return to the surface, open-bottom export once) adds no time, E, T, diffusion or rain.

Binding, dirty proof, staging, adoption
---------------------------------------
There is no caller-supplied dirty list. `stage_native_refresh(owner, parent_bed, child_bed)` (1) streams the actual native rows of the ACCEPTED PARENT bed and the staged CHILD bed in cell batches of `B`
(Windowed: the mandatory native published-journal/cheap/full validation first, then the native gather), compares every active cell EXACTLY (all voxels and classes, plus the active-layer mass; a zero-net-elevation
composition exchange is dirty), and computes a cold SHA-256 SOURCE WITNESS of each side; (2) refuses unless the parent witness equals the witness recorded in the owner's current binding (so the proof is bound to the actual
accepted parent, not a label); (3) reads compact texture/depth ONLY for the dirty cells (batches released, no `(L, N, K)` tensor), computes properties and reconciles; (4) stages every result (new column, state, matrix
operator and rebound history, depth, witness) and mutates nothing. `adopt_native_refresh` checks that the owner is the CURRENT one of its lineage and that the candidate's minted parent binding is that owner's binding;
adopting twice, adopting a stale candidate, or staging against an owner that has been superseded are typed refusals, so a duplicate refresh can never double-debit the export or surface return. The witness
passes are COLD O(N V K) native reads once per refresh (counted in the admission), not a hot-GPU optimisation. The binding token is a run-local minted capability and is NOT a physical checksum.
`mode="depth"` keeps every property and updates only `D` of the dirty cells (physical-depth-only reconciliation, e.g. after actual erosion before publication); `mode="properties"` also refreshes texture-derived
properties. When the staged maps and depth equal the accepted ones nothing is reconciled (no redundant cascade at one generation).

Outputs exposed ONCE: `surface_return_m` `(N,)` (capacity water returned to the surface, an internal transfer the caller must incorporate) and `bottom_export_m` `(N,)` (open-bottom export, a loss), with the D1
receipt. Nothing claims that the returned water already flowed or that an event completed. Bounded accumulated receipts (`maxlen = 8`) plus running totals live on the owner.

Correction01 contract
---------------------
* ONLY the columns whose depth or properties actually changed are reconciled, in compact old/new sub-column batches through the accepted D1 `reconcile_floating`; every unrefreshed column (including one above
  field capacity, a valid continued state) keeps its water bit for bit, and the whole-inventory closure is checked against the FP64 bound. `NativeReconcileReceipt` is the ONE source of the accounting adoption credits.
* Every exposed array and config record is sealed (read-only); the config identity covers every override value; minted owners/candidates/bindings are registered instances that cannot be copied, deep-copied,
  pickled or rebuilt by `dataclasses.replace`; adoption does all refusable work BEFORE consuming the candidate or advancing the lineage authority.
* The witness binds the physical descriptor (grid, geometry, ORDERED grain classes, mass resolution) and canonical rows: newly added or removed EMPTY TOP CAPACITY (different voxel counts) is compared as zero
  padding and is not part of the identity; any other difference is a change.
* The pure provider is called only for slots holding actual material (no fabricated texture, no renormalisation beyond the provider's own declared summation bound).
* The scan/comparison is a COLD O(N V K) boundary pass at a refresh, NOT a post-each-Pond-step optimised path. CUDA integration is pending. The texture-derived saturated conductivity `Ks` is NOT fed back
  into the matrix law (the explicit matrix `K` is retained), so this FC/SAT refresh does not complete the Phase 2 texture-Ks feedback (that needs actual texture `Ks` unless an explicit measured override).

Correction02 contract
---------------------
* Admission is composed from the ACTUAL parent AND child shapes (`estimate_native_refresh_memory(parent_bed, child_bed, config)`; `estimate_native_initialization_memory(bed, config)` for one bed): the native full
  capacity may grow or shrink between them, so the scan rows/gather/canonical-hash/comparison scratch, each bed's own compact-reader/native validators (Windowed: deep-store validator and cache) and the persistent
  candidate are all counted BEFORE the first row is read. Row comparison needs no padding copy (`_changed_rows`); the invariant and closure checks run in bounded column chunks; fresh result arrays are sealed in
  place. A warm tracked-array (tracemalloc) peak is tested inside the estimate for Dense and Windowed at N = 1000, L = 3, B = 16; this is NOT a whole-process bound (interpreter/JIT/pool excluded; no allowance hides a
  known array).
* The minting window is thread-local and bound to ONE constructor call (no public boolean global spans user availability callbacks). `NativeRefreshRequest` on the candidate records the exact parent lineage,
  generation, witness, child witness, mode, batch size, memory requirement and descriptor for a later E4 caller. Late adoption failure stays atomic and the history is never reset.

Actual depth bookkeeping vs hydraulic geometry (E3c)
----------------------------------------------------
The published canonical depth `D` always carries the ACTUAL native value (owner `depth_m`, the column's `soil_depth` map, provenance/snapshots). Only a change of the HYDRAULIC geometry or properties may call
the zero-time cascade: with fixed nominal edges the porous slabs are `max(0, min(D, e_k+1) - e_k)` and the lower boundary is open only when the mode is open AND `D > H` (strict, no epsilon), so a pure depth change
that leaves every slab and the bottom flag bitwise equal (both depths deeper than H with unchanged properties; in an explicit CLOSED profile also `D == H` versus `D > H`) is EQUIVALENT: nothing is
reconciled, nothing exported or returned, yet a genuine new column/state/operator is published bound to the actual depth, carrying the accepted inventory bit for bit (same `W`, time, `J`, cumulative intake) with the
persistent history rebound. An OPEN profile crossing `D == H` -> `D > H` flips the bottom flag and IS reconciled; partial/zero slabs, FC/SAT changes and rock still reconcile as before. `NativeRefreshRequest`
and the candidate now distinguish `n_changed` / `changed_columns` (reconciled) from `n_depth_changed` / `depth_changed` (actual canonical depth differs). This supersedes the earlier C2 behaviour that treated
any raw depth difference as a reconcile trigger (a confirmed defect: an equivalent 1 mm change exported water from an above-field-capacity parent); the old immutable receipts stay evidence of the subset they tested.

Source-binding recipe v2 (E3b correction02)
-------------------------------------------
The physical descriptor, and through it every source witness, now binds the ACTUAL active-mask pattern (versioned digest of the mask shape and every flat cell flag, layout independent) in addition to the grid,
geometry, ordered grain classes, mass resolution and the canonical rows; v1 hashed only the active COUNT, so a different domain with the same count and equal columns could masquerade as the accepted parent.
Both backends also refuse a staged mask different from the owner's with the same typed error (CUDA: after admission, one counted O(N) mask upload and one counted 1-byte verdict, before any row, property or
water work). Witnesses of the two recipes are never equal; accepted C2 receipts/checkpoints are not rewritten and a future checkpoint must record the recipe version.

E3b contract (CUDA branch and texture-Ks mode)
-----------------------------------------------
* `backend="cuda"` configs are served by `native_floating_soil_cuda` (written WITHOUT a GPU; only host-double tests ran): same cold witness/descriptor/canonical depth/provider/changed-column rule/once-only
  adoption; the water, FC/SAT maps and layer-0 Ks stay device-resident (private, logically immutable; admitted counted detached inspection `inspect_counted`); the host holds the sealed O(N) depth and mask
  records and bounded B-tiles; the provider runs on the host per tile (explicit cold path); native device beds are read tile by tile through MAPLE's counted `to_host`; reconciliation is the accepted D1 CUDA
  reconcile over compact device sub-columns and D2 CUDA owns the operator/rebind/history. The CURRENT device is gated first; composed DEVICE and HOST admission use separate budgets, reserves and readings; host/device
  mismatches and other-device beds are typed refusals. The whole-inventory closure of the CUDA branch is the closure of the reconciled columns. The scan is a cold O(N V K) boundary read, not a per-step path;
  nothing here is a full-storm or large-domain claim.
* `matrix_conductivity_mode="texture_ks"` (explicit opt-in; `"explicit"` remains the default and is byte-identical in behaviour and identity): the Green-Ampt matrix conductivity of each cell is the accepted
  provider's CURRENT UPPER-layer (layer 0) saturated conductivity (or a measured override through the provider); a layer 0 with no material keeps the user-supplied latent Ks. Properties mode updates Ks for
  actually affected cells (including a zero-net composition exchange); depth mode holds it. Interface conductivity, diffusivity, suction and residual are NOT derived from it and no averaging exists. A Ks-only
  change builds a new operator over the same column/state (no cascade) and the persistent `J`/cumulative intake are rebound exactly (never reset).

Not in scope: Pond/wind scheduler/checkpoint/restart/every-N output, CUDA owners and device-resident beds (refused), vegetation. Author-side: CPU/host only.
"""

from __future__ import annotations

import contextlib
import hashlib
import secrets
import threading
import weakref
from collections.abc import Callable
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

from maple_syrup import compact_soil_profile as csp
from maple_syrup import floating_matrix_infiltration as fm
from maple_syrup import floating_soil_layers as fsl
from maple_syrup import soil_profile_texture as spt
from maple_syrup import soil_texture_properties as stp
from maple_syrup import subsurface as ss
from maple_syrup.conservation import volume_roundoff_bound_m3

__all__ = [
    "MODES",
    "NativeBed",
    "NativeBinding",
    "NativeFloatingConfig",
    "NativeFloatingError",
    "NativeFloatingMemoryError",
    "NativeFloatingSoil",
    "NativePackingError",
    "NativeReconcileReceipt",
    "NativeRefreshCandidate",
    "NativeRefreshRequest",
    "NativeStaleError",
    "adopt_native_refresh",
    "estimate_native_initialization_memory",
    "estimate_native_refresh_memory",
    "initialize_native_floating_soil",
    "native_floating_config",
    "retire_native_owner_to_lineage",
    "stage_native_refresh",
]

SCHEMA = "maple_syrup.native_floating_soil/1"
WITNESS_RECIPE = "native_floating_soil.witness/2"  # v2 binds the ACTIVE MASK PATTERN (v1 hashed only the active COUNT and the rows); witnesses of the two recipes are never equal
MODES = ("properties", "depth")
RECEIPT_HISTORY = 8
DEFAULT_BATCH_CELLS = 64
_F8 = 8


class NativeFloatingError(ss.SubsurfaceError):
    pass


class NativePackingError(NativeFloatingError):
    """The stated bulk density is inconsistent with the actual MAPLE packing (nothing is adjusted)."""


class NativeStaleError(NativeFloatingError):
    """A stale owner/candidate/binding (superseded parent, duplicate adoption, wrong lineage)."""


class NativeFloatingMemoryError(ss.SubsurfaceMemoryError):
    pass


# ---------------------------------------------------------------------------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class NativeBed:
    """References (never copies) to ONE native accepted/staged bed and its native descriptors: not a second sediment authority."""

    voxel_column: Any
    active_layer: Any
    geometry: Any
    grain_classes: Any
    mass_resolution_kg: float
    active_mask: np.ndarray  # (ny, nx) bool, host

    @property
    def representation(self) -> str:
        return spt._representation(self.voxel_column, self.active_layer)


def _vec(name: str, value: Any, n: int, *, low: float | None = None, high: float | None = None) -> np.ndarray:
    if type(value) is not np.ndarray or value.dtype != np.float64 or value.shape != (n,):
        raise NativeFloatingError(f"{name} must be a float64 numpy array of shape ({n},)")
    if not np.all(np.isfinite(value)) or (low is not None and np.any(value < low)) or (high is not None and np.any(value > high)):
        raise NativeFloatingError(f"{name} must be finite" + (f" and within [{low}, {high}]" if low is not None else ""))
    out = np.array(value, dtype=np.float64, copy=True)
    out.flags.writeable = False
    return out


@dataclass(frozen=True, eq=False)
class NativeFloatingConfig:
    layers: Any
    nominal_edges_m: np.ndarray
    grain_texture: np.ndarray
    organic_matter_percent: np.ndarray
    bulk_density_kg_m3: float
    latent_theta_field_capacity: np.ndarray
    latent_theta_saturation: np.ndarray
    theta_residual: float
    theta_evaporation_stress: float
    evaporation_factor: float
    diffusivity_m2_s: np.ndarray
    bottom_drainage: str
    matrix_conductivity_m_s: float | None  # explicit mode; None in texture_ks mode
    matrix_suction_m: float
    matrix_vegetation_factor: float
    initial_wetting_m: float
    initial_cumulative_intake_m: float
    overrides: Any
    backend: str
    matrix_conductivity_mode: str
    latent_saturated_conductivity_m_s: float | None
    sha256: str


def native_floating_config(*, layer_thickness_m: Any, grain_texture: Any, organic_matter_percent: Any, bulk_density_kg_m3: Any, latent_theta_field_capacity: Any, latent_theta_saturation: Any,
                           theta_residual: Any, theta_evaporation_stress: Any, evaporation_factor: Any = 1.0, diffusivity_m2_s: Any, bottom_drainage: str = "open", matrix_conductivity_m_s: Any = None,
                           matrix_suction_m: Any, matrix_vegetation_factor: Any, initial_wetting_m: Any, initial_cumulative_intake_m: Any, overrides: Any = None,
                           backend: str = "numpy", matrix_conductivity_mode: str = "explicit", latent_saturated_conductivity_m_s: Any = None) -> NativeFloatingConfig:
    """Validate and own every explicit input (nothing is defaulted or inferred). Raises `NativeFloatingError` before anything exists.

    Matrix conductivity: `matrix_conductivity_mode="explicit"` (the default; the accepted behaviour) uses the explicit scalar `matrix_conductivity_m_s`. `"texture_ks"` is the explicit opt-in feedback mode: the
    Green-Ampt matrix conductivity of each cell is the CURRENT UPPER-layer (layer 0) saturated conductivity of the accepted Saxton-Rawls provider (or a measured `saturated_conductivity_override_m_s`); a layer 0 with
    exactly zero mass keeps the USER-SUPPLIED `latent_saturated_conductivity_m_s`. No interface conductivity, diffusivity, suction or residual is derived from it and no vertical averaging exists."""
    if backend not in ("numpy", "numba", "cuda"):
        raise NativeFloatingError(f"backend must be 'numpy', 'numba' or 'cuda', got {backend!r}")
    if matrix_conductivity_mode not in ("explicit", "texture_ks"):
        raise NativeFloatingError("matrix_conductivity_mode must be 'explicit' or 'texture_ks'")
    if matrix_conductivity_mode == "explicit":
        if matrix_conductivity_m_s is None or latent_saturated_conductivity_m_s is not None:
            raise NativeFloatingError("explicit mode requires matrix_conductivity_m_s and refuses latent_saturated_conductivity_m_s")
    else:
        if matrix_conductivity_m_s is not None or latent_saturated_conductivity_m_s is None:
            raise NativeFloatingError("texture_ks mode requires latent_saturated_conductivity_m_s (explicit, for a layer 0 with no material) and refuses matrix_conductivity_m_s")
        if not (np.isfinite(float(latent_saturated_conductivity_m_s)) and float(latent_saturated_conductivity_m_s) > 0.0):
            raise NativeFloatingError("latent_saturated_conductivity_m_s must be finite and > 0")
    layers = spt.canonical_layers(layer_thickness_m)
    L = layers.n_layers
    gt = np.array(grain_texture, dtype=np.float64, copy=True)
    gt.flags.writeable = False
    if gt.ndim != 2 or gt.shape[1] != 3 or not np.all(np.isfinite(gt)) or np.any(gt < 0.0) or np.any(gt > 1.0):
        raise NativeFloatingError("grain_texture must be a finite (K, 3) array of sand/silt/clay fractions in [0, 1] (explicit composition mapping; never inferred from grain diameter)")
    if np.any(np.abs((gt[:, 0] + gt[:, 1]) + gt[:, 2] - 1.0) > stp.closure_bound()):
        raise NativeFloatingError("every grain_texture row must sum to 1 within the provider's FP64 summation bound")
    om = _vec("organic_matter_percent", np.asarray(organic_matter_percent, dtype=np.float64), L, low=0.0, high=8.0)
    lf = _vec("latent_theta_field_capacity", np.asarray(latent_theta_field_capacity, dtype=np.float64), L, low=0.0, high=1.0)
    ls = _vec("latent_theta_saturation", np.asarray(latent_theta_saturation, dtype=np.float64), L, low=0.0, high=1.0)
    diff = _vec("diffusivity_m2_s", np.asarray(diffusivity_m2_s, dtype=np.float64), max(L - 1, 0), low=0.0) if L > 1 else np.zeros(0)
    rho = float(bulk_density_kg_m3)
    if not (1000.0 <= rho <= 1800.0):
        raise NativeFloatingError(f"bulk_density_kg_m3 must lie in the provider domain [1000, 1800], got {rho!r}")
    scal = {n: float(v) for n, v in (("theta_residual", theta_residual), ("theta_evaporation_stress", theta_evaporation_stress), ("evaporation_factor", evaporation_factor),
                                      ("matrix_conductivity_m_s", 0.0 if matrix_conductivity_m_s is None else matrix_conductivity_m_s), ("matrix_suction_m", matrix_suction_m), ("matrix_vegetation_factor", matrix_vegetation_factor),
                                      ("initial_wetting_m", initial_wetting_m), ("initial_cumulative_intake_m", initial_cumulative_intake_m))}
    if not all(np.isfinite(v) for v in scal.values()):
        raise NativeFloatingError("every scalar input must be finite")
    if bottom_drainage not in ("open", "closed"):
        raise NativeFloatingError("bottom_drainage must be 'open' or 'closed'")
    ov = None
    if overrides is not None:
        if not isinstance(overrides, dict) or set(overrides) - set(stp.OVERRIDES):
            raise NativeFloatingError(f"overrides must be a dict over {stp.OVERRIDES}")
        ov = {k: np.array(v, dtype=np.float64, copy=True) for k, v in overrides.items()}
        for k, v in ov.items():
            if v.ndim != 2 or v.shape[0] != L:
                raise NativeFloatingError(f"override {k} must have shape ({L}, N)")
            v.flags.writeable = False
    h = hashlib.sha256()
    for a in (layers.depth_edges_m, gt, om, lf, ls, diff, np.array([rho, *scal.values()])):
        h.update(np.ascontiguousarray(a).tobytes())
    h.update(f"{SCHEMA}|{bottom_drainage}|{backend}".encode())
    if matrix_conductivity_mode == "texture_ks":  # the explicit-mode identity stays byte-identical to the accepted one
        h.update(f"|matrix_conductivity_mode:texture_ks:{float(latent_saturated_conductivity_m_s)!r}".encode())
    for k in sorted(ov or ()):  # EVERY override value (with name, layout, shape and dtype) is part of the identity
        h.update(f"|override:{k}:{ov[k].dtype}:{ov[k].shape}:".encode())
        h.update(np.ascontiguousarray(ov[k]).tobytes())
    edges = np.array(layers.depth_edges_m, dtype=np.float64, copy=True)
    edges.flags.writeable = False
    return NativeFloatingConfig(layers, edges, gt, om, rho, lf, ls, scal["theta_residual"], scal["theta_evaporation_stress"], scal["evaporation_factor"], diff,
                                bottom_drainage, None if matrix_conductivity_mode == "texture_ks" else scal["matrix_conductivity_m_s"], scal["matrix_suction_m"], scal["matrix_vegetation_factor"], scal["initial_wetting_m"],
                                scal["initial_cumulative_intake_m"], ov, backend, matrix_conductivity_mode,
                                None if latent_saturated_conductivity_m_s is None else float(latent_saturated_conductivity_m_s), h.hexdigest())


# ---------------------------------------------------------------------------------------------------------------------------------------------
# native reads: validation, cold rows, witness, exact comparison
# ---------------------------------------------------------------------------------------------------------------------------------------------
def _check_bed(bed: Any, config: NativeFloatingConfig | None = None) -> tuple[int, int, int, int]:
    """Metadata only: returns `(N, V, K, n_active)`. A bed's namespace must match the owner backend: a device bed needs `backend="cuda"` and a host bed a host backend (a mismatch is a typed refusal; there is no
    implicit transfer or backend fallback)."""
    if type(bed) is not NativeBed:
        raise NativeFloatingError(f"bed must be a NativeBed, got {type(bed).__name__}")
    rep = bed.representation
    ny, nx = int(bed.geometry.ny), int(bed.geometry.nx)
    mask = bed.active_mask
    if type(mask) is not np.ndarray or mask.dtype != np.bool_ or mask.shape != (ny, nx):
        raise NativeFloatingError(f"active_mask must be a host bool array of shape {(ny, nx)}")
    active = bed.active_layer.mass_kg
    from maple.core.backend import is_device_array

    device = bool(is_device_array(active))
    K = int(active.shape[2])
    if rep == "dense":
        V = int(bed.voxel_column.mass_kg.shape[2])
        main = bed.voxel_column.mass_kg
    else:
        V = int(spt._windowed_metadata(bed.voxel_column, K)[2])
        main = bed.voxel_column.window_kg
    if bool(is_device_array(main)) != device:
        raise NativeFloatingError("the native bed mixes host and device arrays; there is no implicit transfer")
    if config is not None and (config.backend == "cuda") != device:
        raise NativeFloatingError(f"a {'device' if device else 'host'}-resident native bed cannot be used with the {config.backend!r} owner backend (no implicit transfer or backend fallback)")
    if config is None and device:
        raise NativeFloatingError("device-resident native beds need an owner config with backend='cuda'")
    if config is not None:
        rho = float(bed.geometry.bulk_density_kg_m3)
        if abs(rho - config.bulk_density_kg_m3) > 1e-12 * rho:
            raise NativePackingError(f"the stated bulk density {config.bulk_density_kg_m3!r} kg/m3 is inconsistent with the actual MAPLE packing {rho!r} kg/m3; nothing is adjusted")
        if K != config.grain_texture.shape[0]:
            raise NativeFloatingError(f"grain_texture has {config.grain_texture.shape[0]} rows but the native bed has {K} grain classes")
    return ny * nx, V, K, int(mask.sum())


def _windowed_checks(bed: NativeBed, operation: str) -> None:
    from maple.surface.active_layer.validation import check_active_layer_state
    from maple.surface.voxels.windowed_inventory import (
        check_windowed_column_cheap,
        validate_windowed_column_full,
    )
    from maple.surface.voxels.windowed_pending import require_published_column

    class_ids = tuple(c.class_id for c in bed.grain_classes.classes)
    try:  # mandatory native source checks, never downgraded; an unpublished journal is refused
        require_published_column(bed.voxel_column, operation=operation)
        check_windowed_column_cheap(bed.voxel_column, bed.geometry, class_ids=class_ids)
        check_active_layer_state(spt._as_active(bed.active_layer.mass_kg), bed.geometry, bed.mass_resolution_kg)
        validate_windowed_column_full(bed.voxel_column, bed.geometry, bed.mass_resolution_kg, class_ids=class_ids)
    except (TypeError, ValueError) as exc:
        raise spt._wrap_native(operation, exc) from exc


def _rows(bed: NativeBed, idx: np.ndarray, B: int) -> tuple[np.ndarray, np.ndarray]:
    """Fresh host `(n, V, K)` voxel rows and `(n, K)` active rows of the flat cells `idx` (windowed: the native gather)."""
    from maple.core.backend import to_device, to_host

    K = int(bed.active_layer.mass_kg.shape[2])
    act = bed.active_layer.mass_kg
    if type(act) is np.ndarray:
        a = np.ascontiguousarray(act.reshape(-1, K)[idx])
    else:  # device bed: ONLY the selected tile is moved, through MAPLE's counted `to_host` (bounded cold boundary read, no whole-bed host copy)
        from maple_syrup import (
            floating_soil_layers_cuda as _fcu,  # reached only for a device bed
        )

        _cp = _fcu._rc().require_cuda()
        a = np.ascontiguousarray(to_host(act.reshape(-1, K)[to_device(idx, _cp)]))
    if bed.representation == "dense":
        vox = bed.voxel_column.mass_kg
        if type(vox) is np.ndarray:
            v = np.ascontiguousarray(vox.reshape(-1, vox.shape[2], K)[idx])
        else:
            v = np.ascontiguousarray(to_host(vox.reshape(-1, vox.shape[2], K)[to_device(idx, _cp)]))
    else:
        from maple.surface.voxels.windowed_inventory import gather_full_columns

        v = gather_full_columns(bed.voxel_column, idx, max_columns=B)
    return v, a


def _mask_digest(mask: np.ndarray) -> str:
    """Versioned digest of the ACTUAL active-mask pattern (its shape and every flat cell flag, in C order), independent of the caller's memory layout: a non-contiguous mask is copied once (host, N bytes)."""
    m = np.ascontiguousarray(mask, dtype=np.bool_)
    h = hashlib.sha256(f"{WITNESS_RECIPE}|mask|{m.shape}|".encode())
    h.update(memoryview(m.reshape(-1)).cast("B"))
    return h.hexdigest()


def _descriptor(bed: NativeBed) -> str:
    """Physical SEMANTICS of the native bed that a witness must bind: grid and geometry numbers, the ORDERED grain classes (id, diameter, aggregate flag, particle density) and the mass resolution. Two beds with
    equal masses but another class order, grid or geometry have different descriptors and are never interchangeable."""
    K = int(bed.active_layer.mass_kg.shape[2])
    ny, nx, dx, dy, rho, dz, cap = spt._geometry_numbers(bed.geometry)
    ids, diam, agg, prho = spt._class_metadata(bed.grain_classes, K)
    h = hashlib.sha256()
    h.update(f"{SCHEMA}|{WITNESS_RECIPE}|{bed.representation}|{ny}x{nx}|{dx!r}|{dy!r}|{rho!r}|{dz!r}|{cap!r}|{float(bed.mass_resolution_kg)!r}|{K}|mask:{_mask_digest(bed.active_mask)}".encode())
    for i, d, a, r in zip(ids, diam, agg, prho, strict=True):
        h.update(f"|{i}:{d!r}:{bool(a)}:{r!r}".encode())
    return h.hexdigest()


def _changed_rows(vc: np.ndarray, vp: np.ndarray) -> np.ndarray:
    """Per-cell bitwise comparison of two `(n, V, K)` row sets in the common physical depth WITHOUT padding copies: the common voxels are compared directly and the extra voxels of the larger one (empty top
    capacity) must be exactly zero bit patterns (a non-zero extra voxel is a change). A NaN or a signed zero is also a change. Temporaries are boolean (1 byte per compared value)."""
    n = vc.shape[0]
    nv = min(vc.shape[1], vp.shape[1])
    a, b = vc.view(np.uint64), vp.view(np.uint64)
    changed = np.any((a[:, :nv] != b[:, :nv]).reshape(n, -1), axis=1)
    if a.shape[1] > nv:
        changed |= np.any((a[:, nv:] != 0).reshape(n, -1), axis=1)
    elif b.shape[1] > nv:
        changed |= np.any((b[:, nv:] != 0).reshape(n, -1), axis=1)
    return changed


def _hash_canonical(hv: Any, v: np.ndarray) -> None:
    """Canonical rows: the number of voxels up to the last non-zero bit pattern, then exactly those voxels, per cell; empty top capacity is therefore not part of the identity (it may grow or shrink)."""
    nz = np.any(v.view(np.uint64) != 0, axis=2)  # (n, V)
    last = v.shape[1] - np.argmax(nz[:, ::-1], axis=1)
    last = np.where(nz.any(axis=1), last, 0).astype(np.int64)
    for i in range(v.shape[0]):  # per cell: its canonical voxel count, then exactly those voxels (one sequential stream: independent of the batch size)
        hv.update(last[i:i + 1].tobytes())
        hv.update(np.ascontiguousarray(v[i, :int(last[i])]).tobytes())


def _scan(parent: NativeBed | None, child: NativeBed, active_flat: np.ndarray, B: int) -> tuple[np.ndarray, str, str | None]:
    """ONE cold streamed pass over the active cells (a COLD O(N V K) native read, NOT a per-Pond-step path): returns `(dirty flat indices, child witness, parent witness)`. Both beds must share the
    physical descriptor (grid, geometry, ordered grain classes). A cell is dirty when ANY native value (every voxel/class compared in the common physical depth with empty top capacity padded by zeros, or the
    active-layer mass) differs bit for bit. The witness binds the descriptor and the canonical rows (empty top capacity excluded)."""
    dc = _descriptor(child)
    dp = _descriptor(parent) if parent is not None else None
    if parent is not None and dp != dc:
        raise NativeFloatingError("parent and child native beds differ in grid, geometry, grain-class order/properties or mass resolution; they are not comparable")

    def hashers(desc: str) -> tuple:
        hv, ha = hashlib.sha256(), hashlib.sha256()  # separate voxel / active streams: the witness is independent of the batch size
        for h in (hv, ha):
            h.update(f"{SCHEMA}|{desc}|{active_flat.size}".encode())
        return hv, ha

    def digest(hs: tuple) -> str:
        return hashlib.sha256(hs[0].digest() + hs[1].digest()).hexdigest()

    hc = hashers(dc)
    hp = hashers(dp) if parent is not None else None
    dirty = []
    for b0 in range(0, active_flat.size, B):
        idx = active_flat[b0:b0 + B]
        vc, ac = _rows(child, idx, B)
        _hash_canonical(hc[0], vc)
        hc[1].update(ac.tobytes())
        if parent is not None:
            vp, ap = _rows(parent, idx, B)
            _hash_canonical(hp[0], vp)
            hp[1].update(ap.tobytes())
            changed = _changed_rows(vc, vp)  # exact BITWISE comparison of every native value
            changed |= np.any(ac.view(np.uint64) != ap.view(np.uint64), axis=1)
            dirty.append(idx[changed])
            del vp, ap
        del vc, ac
    out = np.concatenate(dirty) if dirty else np.zeros(0, dtype=np.int64)
    return out.astype(np.int64, copy=False), digest(hc), None if hp is None else digest(hp)


# ---------------------------------------------------------------------------------------------------------------------------------------------
# memory composition (metadata only, BEFORE any scratch)
# ---------------------------------------------------------------------------------------------------------------------------------------------
def _is_device_bed(bed: NativeBed) -> bool:
    from maple.core.backend import is_device_array

    return bool(is_device_array(bed.active_layer.mass_kg))


def _e2_backend(bed: NativeBed) -> str:
    """The E2 compact-reader backend selected by the GENUINE native representation and namespace: a Dense device bed uses the E2 CUDA compute; a Windowed bed (host or device window) is gathered and reduced
    on the host by MAPLE's own I/O ('numpy') and the compact outputs are placed in the window's namespace by E2; a host Dense bed keeps the accepted CPU 'numpy' path. The E2 representation/backend guards are
    never weakened or bypassed."""
    return "cuda" if (bed.representation == "dense" and _is_device_bed(bed)) else "numpy"


def _shape_info(bed: NativeBed, config: NativeFloatingConfig) -> tuple[int, int, int, int]:
    return _check_bed(bed, config)


def _native_phase(bed: NativeBed, config: NativeFloatingConfig, B: int, C: int, K: int, V: int, N: int) -> Any:
    """The compact-reader/native-validation requirement of ONE bed at ITS OWN shape (a windowed bed adds its deep-store validator, gather and cache terms)."""
    extra = {}
    if bed.representation == "windowed":
        _, _, _, W, CC, cache_bytes = spt._windowed_metadata(bed.voxel_column, K)
        extra = {"window_voxels": W, "chunk_cells": CC, "native_cache_bytes": cache_bytes}
    return _native_estimate(bed, config, B, C, K, V, N, extra)


def _native_estimate(bed: NativeBed, config: NativeFloatingConfig, B: int, C: int, K: int, V: int, N: int, extra: dict) -> Any:
    return csp.estimate_compact_memory(representation=bed.representation, backend=_e2_backend(bed), output_namespace="cupy" if _is_device_bed(bed) else "numpy", n_cells=N, n_selected=C,
                                       n_layers=config.layers.n_layers, n_classes=K, n_voxels=V, batch_cells=B, **extra)


def _estimate(parent: NativeBed | None, child: NativeBed, config: NativeFloatingConfig, batch_cells: int, reserve_bytes: int, n_selected: int | None = None) -> dict:
    N, Vc, K, C = _shape_info(child, config)
    if n_selected is not None:  # a SELECTED refresh reads only these columns (default None keeps the historical all-active model)
        C = fsl._count(n_selected, "n_selected")
    Vp = Vc
    if parent is not None:
        Np, Vp, Kp, Cp = _shape_info(parent, config)
        if (Np, Kp, Cp) != (N, K, C):
            raise NativeFloatingError("parent and child beds must share the grid, class count and active mask")
    L = config.layers.n_layers
    B = max(1, min(fsl._count(batch_cells, "batch_cells", minimum=1), max(C, 1)))
    Vmax = max(Vc, Vp)
    # ---- cold scan, per batch of B cells, from the ACTUAL parent AND child shapes: the gathered/copied row sets (windowed: the native gather needs a second set of the same size while it assembles),
    # the canonical-hash non-zero masks, and the comparison booleans over the common depth plus the extra voxels of the larger bed; plus the sealed O(C)/O(N) index/mask arrays
    rows_c = (B * Vc * K + B * K) * _F8
    rows_p = (B * Vp * K + B * K) * _F8 if parent is not None else 0
    gather_extra = (B * Vc * K * _F8 if child.representation == "windowed" else 0) + (B * Vp * K * _F8 if parent is not None and parent.representation == "windowed" else 0)
    nz = (B * Vc * K + B * Vc) + ((B * Vp * K + B * Vp) if parent is not None else 0)  # bool: `view != 0` and its any(axis=2)
    cmp_ = (B * Vmax * K + B * K * _F8) if parent is not None else 0  # bool comparisons over the common depth / extra voxels, plus the uint64-active comparison booleans
    scan_index = 8 * C + 8 * C + N  # active_flat, dirty list, mask view bound
    scan = rows_c + rows_p + gather_extra + nz + cmp_ + scan_index
    layouts = {k: "cell" for k in stp.INPUTS}
    layouts.update({k: "cell" for k in (config.overrides or {})})
    est_child = _native_phase(child, config, B, C, K, Vc, N)
    est_parent = _native_phase(parent, config, B, C, K, Vp, N) if parent is not None else None
    native_child = est_child.required_bytes
    native_parent = est_parent.required_bytes if est_parent is not None else 0
    native_device = max(est_child.device_bytes, est_parent.device_bytes if est_parent is not None else 0)  # device part of the compact reader (a device bed), counted separately by the CUDA estimate
    prov = stp.estimate_property_memory(L, B, layouts=layouts, backend="numpy").main_required_bytes
    maps = 2 * L * N * _F8  # new theta_fc / theta_sat arrays (the accepted maps stay live inside the owner)
    depth = N * _F8 + 8 * C + 8 * C + N  # new D, dirty and changed index arrays, is_changed mask
    lay = {"theta_residual": "scalar", "theta_field_capacity": "cell", "theta_saturation": "cell", "interface": "shared"}
    est = fsl.estimate_floating_memory(L, N, layouts=lay, diffusion_mode="constant", backend=config.backend)
    est_b = fsl.estimate_floating_memory(L, B, layouts=lay, diffusion_mode="constant", backend=config.backend)
    # new whole column (retained) + one compact old/new sub-column pair with its state and reconcile step (B-sized) + the whole-width water scatter and the new state (2 x L N) + return/export (2 x N)
    recon = est.retained_bytes + 2 * est_b.retained_bytes + est_b.step_bytes + 2 * L * N * _F8 + 2 * N * _F8
    matrix = 7 * N * _F8  # new layer-0 Ks record + operator law arrays (<= 3 N) and the rebound history (2 N)
    chunk_checks = 2 * L * B * _F8 + L * B  # bounded column-chunk copies of the invariant/closure checks
    phases = {"scan": scan, "parent_native": native_parent, "child_native_and_properties": native_child + prov, "persistent_candidate": maps + depth + recon + matrix + chunk_checks}
    total = phases["persistent_candidate"] + max(phases["scan"], phases["parent_native"], phases["child_native_and_properties"]) + fsl._count(reserve_bytes, "reserve_bytes")
    return {"batch_cells": B, "n_cells": N, "n_active": C, "parent_voxels": Vp if parent is not None else None, "child_voxels": Vc, "compact_bytes": native_child, "native_device_bytes": native_device, "native_child_device_bytes": est_child.device_bytes, "native_parent_device_bytes": 0 if est_parent is None else est_parent.device_bytes,
            "provider_bytes": prov, "terms": phases,
            "required_bytes": total, "scope": "modelled host payload from the ACTUAL parent and child shapes; native input arrays, interpreter/JIT and pool rounding are excluded (no allowance hides a known array); a reading is not a reservation"}


def estimate_native_refresh_memory(parent_bed: NativeBed, child_bed: NativeBed, config: NativeFloatingConfig, *, batch_cells: int = DEFAULT_BATCH_CELLS, reserve_bytes: int = 0) -> dict:
    """Composed host byte model of one REFRESH from the actual accepted PARENT and staged CHILD beds (both shapes are required inputs: the native full capacity may differ between them): the cold scan (rows of
    both beds, gather, canonical-hash and comparison scratch), each bed's compact reader/native validators (Windowed: deep-store validator and cache), the property provider at `(L, B)`, the new maps/depth, the
    compact reconciliation, the whole-width scatter/state and the rebound history. Peak = persistent candidate + the largest SIMULTANEOUS transient phase; every term is listed. Metadata only."""
    return _estimate(parent_bed, child_bed, config, batch_cells, reserve_bytes)


def estimate_native_initialization_memory(bed: NativeBed, config: NativeFloatingConfig, *, batch_cells: int = DEFAULT_BATCH_CELLS, reserve_bytes: int = 0) -> dict:
    """The same composition for an initialization (one bed, no parent)."""
    return _estimate(None, bed, config, batch_cells, reserve_bytes)


def _admit(model: dict, budget: Any, reader: Callable[[], int] | None, what: str) -> None:
    need = model["required_bytes"]
    if budget is not None and need > fsl._count(budget, "memory_budget_bytes"):
        raise NativeFloatingMemoryError(f"{what} needs an estimated {need} B (terms {model['terms']}) but the declared memory_budget_bytes is {budget}; nothing was allocated")
    avail = (reader or fsl.sp.default_host_available_bytes)()
    if isinstance(avail, (bool, np.bool_)) or not isinstance(avail, (int, np.integer)) or int(avail) < 0:
        raise NativeFloatingError(f"the available-bytes reader must return a non-Boolean integer >= 0, got {avail!r}")
    if need > int(avail):
        raise NativeFloatingMemoryError(f"{what} needs an estimated {need} B but only {int(avail)} B are reported available; nothing was allocated")


# ---------------------------------------------------------------------------------------------------------------------------------------------
# texture -> properties for compact batches
# ---------------------------------------------------------------------------------------------------------------------------------------------
def _provider_backend(config: NativeFloatingConfig) -> str:
    """The accepted provider runs on the host for a CUDA owner too (an explicit bounded COLD tile path: the composed host budget counts the tile and the transfers); outputs are uploaded per tile."""
    return "numpy" if config.backend == "cuda" else config.backend


def _batch_properties(batch: Any, config: NativeFloatingConfig, K: int, provider_bytes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """`(theta_field_capacity (L, B), theta_saturation (L, B), layer-0 Ks (B,) or None)` of one compact batch (Ks only in the explicit `texture_ks` mode: the provider's saturated conductivity of the UPPER layer,
    the latent value for a layer 0 holding no material). The pure provider is called ONLY for slots holding actual material (per layer, on exactly those cells); a zero-mass slot
    receives the explicit latent values and no texture is fabricated, estimated or passed for it (a rock-only profile calls nothing). The class fractions are mapped by the explicit `(K, 3)` matrix and handed
    to the provider UNCHANGED: the provider's own declared MAPLE summation bound is the only closure contract (no renormalisation)."""
    L, B = config.layers.n_layers, batch.n_cells
    from maple.core.backend import to_host

    frac = np.asarray(to_host(batch.layer_fraction), dtype=np.float64)  # a device batch tile is moved through the counted `to_host` (bounded B, cold)
    total = np.asarray(to_host(batch.layer_total_kg), dtype=np.float64)
    tex = np.einsum("lbk,kc->lbc", frac, config.grain_texture)
    fc = np.empty((L, B))
    sat = np.empty((L, B))
    fc[:] = config.latent_theta_field_capacity[:, None]
    sat[:] = config.latent_theta_saturation[:, None]
    ks0 = None
    if config.matrix_conductivity_mode == "texture_ks":
        ks0 = np.full(B, config.latent_saturated_conductivity_m_s)
    for layer in range(L):
        sel = np.flatnonzero(total[layer] > 0.0)
        if sel.size == 0:
            continue
        m = int(sel.size)
        cols = [np.ascontiguousarray(tex[layer:layer + 1, sel, c]) for c in range(3)]
        om = np.ascontiguousarray(np.full((1, m), config.organic_matter_percent[layer]))
        rho = np.ascontiguousarray(np.full((1, m), config.bulk_density_kg_m3))
        kw = {}
        if config.overrides:
            kw = {k: np.ascontiguousarray(v[layer:layer + 1, batch.indices[sel]]) for k, v in config.overrides.items()}
        res = stp.estimate_soil_texture_properties(sand_fraction=cols[0], silt_fraction=cols[1], clay_fraction=cols[2], organic_matter_percent=om, bulk_density_kg_m3=rho, n_layers=1, n_cells=m,
                                                   backend=_provider_backend(config), memory_budget_bytes=provider_bytes, **kw)  # a (1, m) call: never larger than the composed (L, B) provider term
        fc[layer, sel] = res.array("theta_field_capacity_6kpa")[0]
        sat[layer, sel] = res.array("theta_saturation")[0]
        if layer == 0 and ks0 is not None:
            ks0[sel] = res.array("saturated_conductivity_m_s")[0]
    return fc, sat, ks0


def _read_texture(child: NativeBed, config: NativeFloatingConfig, indices: np.ndarray, fc: np.ndarray | None, sat: np.ndarray | None, depth: np.ndarray, *, model: dict, batch_cells: int,
                  update_properties: bool, ks: np.ndarray | None = None) -> None:
    """Fill `fc`/`sat` (L, N) columns and `depth` (N,) for the selected cells, batch by batch, releasing each compact batch."""
    if indices.size == 0:
        return
    K = int(child.active_layer.mass_kg.shape[2])
    it = csp.iter_compact_soil_profile(child.voxel_column, child.active_layer, child.geometry, child.grain_classes, child.mass_resolution_kg, config.layers, indices=indices,
                                       active_mask=child.active_mask, memory_budget_bytes=model["compact_bytes"], batch_cells=max(1, min(batch_cells, int(indices.size))), backend=_e2_backend(child),
                                       coverage="partial_over_base")
    for batch in it:
        depth[batch.indices] = batch.canonical_depth_m
        if update_properties:
            f, s, k0 = _batch_properties(batch, config, K, model["provider_bytes"])
            fc[:, batch.indices] = f
            sat[:, batch.indices] = s
            if ks is not None:
                ks[batch.indices] = k0
        del batch


# ---------------------------------------------------------------------------------------------------------------------------------------------
# owner, bindings, candidate
# ---------------------------------------------------------------------------------------------------------------------------------------------
_MINT = threading.local()  # thread-local: a parallel thread can never see another thread's minting window
_MINTED_INSTANCES: weakref.WeakSet = weakref.WeakSet()  # every genuinely minted binding/owner/candidate; copies and direct constructions are never members


@contextlib.contextmanager
def _minting():
    """A minting window bound to ONE local constructor call (never to a whole public function, so user availability callbacks and re-entrant calls run OUTSIDE it)."""
    _MINT.ok = True
    try:
        yield
    finally:
        _MINT.ok = False


def _no_direct(what: str) -> None:
    if not getattr(_MINT, "ok", False):
        raise NativeFloatingError(f"a {what} is minted only by initialize_native_floating_soil / stage_native_refresh / adopt_native_refresh")


class _Capability:
    """Mixin of the minted capability records: registered in the process-wide minted set at construction; never copyable, deep-copyable or serialisable (an ordinary copy would otherwise mint a second
    genuine branch of the same authority)."""

    __slots__ = ()

    def __copy__(self):
        raise NativeFloatingError(f"a {type(self).__name__} is a minted capability and cannot be copied")

    def __deepcopy__(self, memo):
        raise NativeFloatingError(f"a {type(self).__name__} is a minted capability and cannot be copied")

    def __reduce_ex__(self, protocol):
        raise NativeFloatingError(f"a {type(self).__name__} is a minted capability and cannot be serialised")


def _require_minted(obj: Any, what: str) -> None:
    if obj not in _MINTED_INSTANCES:
        raise NativeStaleError(f"the {what} is not a genuinely minted instance of this process (a copy or reconstruction does not carry the authority)")


def _seal(a: np.ndarray) -> np.ndarray:
    out = np.array(a, copy=True)
    out.flags.writeable = False
    return out


@dataclass(frozen=True, eq=False)
class NativeBinding(_Capability):
    """A minted generation capability: the lineage, the generation and the cold native SOURCE WITNESS (which binds the physical bed descriptor) the owner was accepted against. `token` is a run-local capability,
    NOT a physical checksum."""

    lineage: str
    generation: int
    source_witness: str
    token: str

    def __post_init__(self) -> None:
        _no_direct("NativeBinding")
        _MINTED_INSTANCES.add(self)


class _Authority:
    """The shared mutable cell of one lineage: only the binding stored here is the current accepted parent."""

    __slots__ = ("current",)

    def __init__(self) -> None:
        self.current: NativeBinding | None = None


@dataclass(frozen=True)
class NativeReconcileReceipt:
    """Aggregate receipt of the zero-time reconciliation of ONLY the changed columns (in compact batches), with the whole-inventory closure checked separately."""

    n_changed_columns: int
    n_batches: int
    storage_before_m3: float  # changed columns only
    storage_after_m3: float
    surface_return_m3: float
    bottom_export_m3: float
    residual_m3: float
    bound_m3: float
    moved_percolation_m3: float
    whole_before_m3: float  # every active column
    whole_after_m3: float
    whole_residual_m3: float
    whole_bound_m3: float
    old_column_sha256: str
    new_column_sha256: str


@dataclass(frozen=True, eq=False)
class NativeFloatingSoil(_Capability):
    """The accepted water owner: ONE inventory (`state`), the floating column, the matrix operator and persistent history, the canonical depth and the binding. Immutable; superseded by adoption."""

    config: NativeFloatingConfig
    column: Any
    state: Any
    matrix_op: Any
    matrix_history: Any
    depth_m: np.ndarray
    ks_m_s: Any  # (N,) layer-0 saturated conductivity record (texture_ks mode) or None
    binding: NativeBinding
    receipts: tuple
    totals: Any
    _authority: Any

    def __post_init__(self) -> None:
        _no_direct("NativeFloatingSoil")
        _MINTED_INSTANCES.add(self)

    @property
    def generation(self) -> int:
        return self.binding.generation

    @property
    def is_current(self) -> bool:
        return self in _MINTED_INSTANCES and self._authority.current is self.binding


@dataclass(frozen=True)
class NativeRefreshRequest:
    """The exact-parent request this candidate answers (for a later Pond/E4 caller): the parent lineage/generation/witness it was staged against, the child witness, mode, batch size, the composed memory
    requirement, the physical descriptor and the proven dirty/changed counts. Descriptive; the authority is the minted parent binding."""

    lineage: str
    parent_generation: int
    parent_witness: str
    child_witness: str
    mode: str
    batch_cells: int
    memory_required_bytes: int
    descriptor: str
    n_dirty: int
    n_changed: int  # columns RECONCILED (porous geometry / bottom condition / FC-SAT actually changed)
    n_depth_changed: int = 0  # columns whose actual canonical depth differs (includes hydraulically equivalent bookkeeping changes)


@dataclass(frozen=True, eq=False)
class NativeRefreshCandidate(_Capability):
    """A STAGED, unadopted refresh: every result exists and every exposed array is a sealed (read-only) copy; nothing is mutated. `surface_return_m` and `bottom_export_m` are exposed here ONCE for the
    caller and are consistent with `receipt` (the only source of the accounting that adoption credits); `dirty_indices` is the proven changed active set, `changed_columns` those whose depth or properties
    actually changed and were reconciled."""

    parent_binding: NativeBinding
    child_witness: str
    mode: str
    dirty_indices: np.ndarray
    changed_columns: np.ndarray
    depth_changed: np.ndarray
    column: Any
    state: Any
    matrix_op: Any
    matrix_history: Any
    depth_m: np.ndarray
    ks_m_s: Any
    surface_return_m: np.ndarray
    bottom_export_m: np.ndarray
    receipt: Any
    reconciled: bool
    memory: Any
    request: Any
    _authority: Any
    _consumed: list

    def __post_init__(self) -> None:
        _no_direct("NativeRefreshCandidate")
        _MINTED_INSTANCES.add(self)

    @property
    def n_dirty(self) -> int:
        return int(self.dirty_indices.size)


def _quota_args(config: NativeFloatingConfig, memory_budget_bytes: Any, reserve_bytes: Any, available_bytes_fn: Any, host_memory_budget_bytes: Any, host_reserve_bytes: Any,
                host_available_bytes_fn: Any) -> dict:
    if config.backend != "cuda" and (host_memory_budget_bytes is not None or host_reserve_bytes or host_available_bytes_fn is not None):
        raise NativeFloatingError("host_memory_budget_bytes / host_reserve_bytes / host_available_bytes_fn apply only to the cuda backend")
    return {"memory_budget_bytes": memory_budget_bytes, "reserve_bytes": reserve_bytes, "available_bytes_fn": available_bytes_fn, "host_memory_budget_bytes": host_memory_budget_bytes,
            "host_reserve_bytes": host_reserve_bytes, "host_available_bytes_fn": host_available_bytes_fn}


def _flat_active(bed: NativeBed) -> np.ndarray:
    return np.flatnonzero(bed.active_mask.reshape(-1)).astype(np.int64)


def _area(bed: NativeBed) -> float:
    return float(bed.geometry.dx_m) * float(bed.geometry.dy_m)


def _prepare(config: NativeFloatingConfig, fc: np.ndarray, sat: np.ndarray, depth: np.ndarray, mask_flat: np.ndarray, area: float) -> Any:
    return fsl.prepare_floating_column(config.nominal_edges_m, theta_residual=config.theta_residual, theta_field_capacity=np.ascontiguousarray(fc), theta_saturation=np.ascontiguousarray(sat),
                                       diffusivity_m2_s=config.diffusivity_m2_s if config.layers.n_layers > 1 else 0.0, theta_evaporation_stress=config.theta_evaporation_stress,
                                       evaporation_factor=config.evaporation_factor, diffusion_mode="constant", bottom_drainage=config.bottom_drainage, soil_depth_m=np.ascontiguousarray(depth),
                                       active_mask=np.ascontiguousarray(mask_flat), cell_area_m2=area, backend=config.backend)


def _build_column(config: NativeFloatingConfig, fc: np.ndarray, sat: np.ndarray, depth: np.ndarray, bed: NativeBed) -> Any:
    return _prepare(config, fc, sat, depth, bed.active_mask.reshape(-1), _area(bed))


def _fresh_op(config: NativeFloatingConfig, column: Any, ks: np.ndarray | None = None) -> Any:
    cond = ks if config.matrix_conductivity_mode == "texture_ks" else config.matrix_conductivity_m_s
    return fm.prepare_floating_matrix(column, conductivity_m_s=cond, suction_m=config.matrix_suction_m, vegetation_factor=config.matrix_vegetation_factor)


def _reconcile_changed(config: NativeFloatingConfig, old_maps: dict, old_depth: np.ndarray, old_water: np.ndarray, new_fc: np.ndarray, new_sat: np.ndarray, new_depth: np.ndarray,
                       changed: np.ndarray, area: float, B: int, t_s: float) -> tuple:
    """Zero-time reconciliation of ONLY the columns whose depth or properties actually changed, in compact batches of `B`: for each batch a compact OLD and NEW floating sub-column pair is prepared and the
    accepted D1 `reconcile_floating` (the one physical body) is applied to the old water of exactly those columns; results are scattered into whole-column arrays. Every unrefreshed column keeps its water
    bit for bit (it is never passed through a cascade, even when it lies above field capacity: a valid continued state). Returns `(new water (L, N), surface_return (N,), bottom_export (N,), receipt dict)`."""
    _L, N = old_water.shape
    water = np.array(old_water, copy=True)
    surf = np.zeros(N)
    exp_ = np.zeros(N)
    before = after = s_m3 = e_m3 = resid = bound = moved = 0.0
    nb = 0
    ones = None
    for b0 in range(0, changed.size, B):
        idx = changed[b0:b0 + B]
        n = int(idx.size)
        ones = np.ones(n, dtype=np.bool_)
        old_sub = _prepare(config, old_maps["theta_field_capacity"][:, idx], old_maps["theta_saturation"][:, idx], old_depth[idx], ones, area)
        new_sub = _prepare(config, new_fc[:, idx], new_sat[:, idx], new_depth[idx], ones, area)
        old_state = fsl.floating_continued_state(old_sub, np.ascontiguousarray(old_water[:, idx]), t_s)
        cand = fsl.reconcile_floating(old_sub, old_state, new_sub)
        water[:, idx] = np.asarray(cand.state.water_m)
        surf[idx] = np.asarray(cand.surface_return_m)
        exp_[idx] = np.asarray(cand.bottom_export_m)
        r = cand.receipt
        before += r.storage_before_m3
        after += r.storage_after_m3
        s_m3 += r.surface_return_m3
        e_m3 += r.bottom_export_m3
        resid += r.residual_m3
        bound += r.bound_m3
        moved += r.moved_percolation_m3
        nb += 1
        del old_sub, new_sub, old_state, cand
    return water, surf, exp_, {"n_changed_columns": int(changed.size), "n_batches": nb, "storage_before_m3": before, "storage_after_m3": after, "surface_return_m3": s_m3, "bottom_export_m3": e_m3,
                               "residual_m3": resid, "bound_m3": bound, "moved_percolation_m3": moved}


def initialize_native_floating_soil(bed: NativeBed, config: NativeFloatingConfig, *, initial_relative_saturation: Any, memory_budget_bytes: Any = None, reserve_bytes: Any = 0,
                                    available_bytes_fn: Callable[[], int] | None = None, batch_cells: int = DEFAULT_BATCH_CELLS, host_memory_budget_bytes: Any = None,
                                    host_reserve_bytes: Any = 0, host_available_bytes_fn: Callable[[], int] | None = None) -> NativeFloatingSoil:
    """New-production initialization of ALL active cells from the actual native bed. `initial_relative_saturation` in `[0, 1)` (strictly unsaturated, the D1 initial-state policy) sets
    `W = r * theta_sat * h` of every porous slot (0 on rock slots; inactive placeholders 0). Order: metadata/packing checks -> composed admission -> (Windowed: native validation) -> cold witness scan ->
    compact texture/depth of every active cell in released batches -> properties -> column/state/matrix -> mint. Nothing is retained on failure. `backend="cuda"`: see `native_floating_soil_cuda` (the
    `host_*` quota arguments apply only there)."""
    quota = _quota_args(config, memory_budget_bytes, reserve_bytes, available_bytes_fn, host_memory_budget_bytes, host_reserve_bytes, host_available_bytes_fn)
    if config.backend == "cuda":
        from maple_syrup import native_floating_soil_cuda as ncuda

        return ncuda.initialize_cuda(bed, config, initial_relative_saturation=initial_relative_saturation, quota=quota, batch_cells=batch_cells)
    N, _V, _K, _C = _check_bed(bed, config)
    r = float(initial_relative_saturation)
    if not (0.0 <= r < 1.0):
        raise NativeFloatingError("initial_relative_saturation must lie in [0, 1) (strictly unsaturated initialization)")
    model = estimate_native_initialization_memory(bed, config, batch_cells=batch_cells, reserve_bytes=fsl._count(reserve_bytes, "reserve_bytes"))
    _admit(model, memory_budget_bytes, available_bytes_fn, "native floating initialization")
    if bed.representation == "windowed":
        _windowed_checks(bed, "initialize_native_floating_soil")
    active_flat = _flat_active(bed)
    B = model["batch_cells"]
    _, witness, _ = _scan(None, bed, active_flat, B)
    L = config.layers.n_layers
    fc, sat, depth = (np.zeros((L, N)), np.zeros((L, N)), np.zeros(N))
    fc[:] = config.latent_theta_field_capacity[:, None]
    sat[:] = config.latent_theta_saturation[:, None]
    ks = np.full(N, config.latent_saturated_conductivity_m_s) if config.matrix_conductivity_mode == "texture_ks" else None
    _read_texture(bed, config, active_flat, fc, sat, depth, model=model, batch_cells=B, update_properties=True, ks=ks)
    column = _build_column(config, fc, sat, depth, bed)
    h = np.stack([fsl.porous_thickness(column, k) for k in range(L)])
    water = r * sat * h
    water[:, ~bed.active_mask.reshape(-1)] = 0.0
    state = fsl.floating_initial_state(column, np.ascontiguousarray(water), 0.0)
    op = _fresh_op(config, column, ks)
    if ks is not None:
        ks.flags.writeable = False
    hist = fm.floating_matrix_history(op, wetting_m=config.initial_wetting_m, cumulative_intake_m=config.initial_cumulative_intake_m)
    authority = _Authority()
    depth.flags.writeable = False  # the owner takes ownership of this fresh array
    with _minting():
        binding = NativeBinding(secrets.token_hex(8), 0, witness, secrets.token_hex(16))
        owner = NativeFloatingSoil(config, column, state, op, hist, depth, ks, binding, (), MappingProxyType({"refreshes": 0, "cells_refreshed": 0, "surface_return_m3": 0.0, "bottom_export_m3": 0.0}),
                                   authority)
    authority.current = binding
    return owner


def _inventory_closure(water: np.ndarray, old_water: np.ndarray, mask: np.ndarray, changed: np.ndarray, B: int, area: float, acc: dict, surf: np.ndarray, exp_: np.ndarray) -> tuple[float, float, float, float]:
    """The whole-inventory closure of one changed-column refresh, factored unchanged from `stage_native_refresh` (shared with the event-boundary refresh): unrefreshed columns bit for bit (bounded chunks), the active-column
    storage before/after, the FP64 residual against the explicit surface return + bottom export and the exposed-array consistency. Returns `(w_before_m3, w_after_m3, residual_m3, bound_m3)` or raises `NativeFloatingError`."""
    _L, N = old_water.shape
    is_changed = np.zeros(N, dtype=bool)
    is_changed[changed] = True
    w_before = w_after = 0.0
    for c0 in range(0, N, B):
        sl = slice(c0, min(c0 + B, N))
        keep = ~is_changed[sl]
        if not np.array_equal(water[:, sl][:, keep], old_water[:, sl][:, keep]):
            raise NativeFloatingError("internal: an unrefreshed column's water changed")
        w_before += float(old_water[:, sl][:, mask[sl]].sum())
        w_after += float(water[:, sl][:, mask[sl]].sum())
    w_before *= area
    w_after *= area
    whole_resid = (w_after - w_before) + (acc["surface_return_m3"] + acc["bottom_export_m3"])
    whole_bound = volume_roundoff_bound_m3(4 * int(old_water.size) + 4, max(abs(w_before), abs(w_after), abs(acc["surface_return_m3"]), abs(acc["bottom_export_m3"])))
    if not (np.isfinite(whole_resid) and abs(whole_resid) <= whole_bound):
        raise NativeFloatingError(f"whole-inventory closure violated: residual {whole_resid!r} m3 exceeds the FP64 bound {whole_bound!r} m3; nothing is staged")
    if abs(float(surf.sum()) * area - acc["surface_return_m3"]) > whole_bound or abs(float(exp_.sum()) * area - acc["bottom_export_m3"]) > whole_bound:
        raise NativeFloatingError("internal: exposed return/export arrays disagree with the receipt")
    return w_before, w_after, whole_resid, whole_bound


def stage_native_refresh(owner: NativeFloatingSoil, parent_bed: NativeBed, child_bed: NativeBed, *, mode: str = "properties", memory_budget_bytes: Any = None, reserve_bytes: Any = 0,
                         available_bytes_fn: Callable[[], int] | None = None, batch_cells: int = DEFAULT_BATCH_CELLS, host_memory_budget_bytes: Any = None, host_reserve_bytes: Any = 0,
                         host_available_bytes_fn: Callable[[], int] | None = None) -> NativeRefreshCandidate:
    """PURE staging of one changed-column refresh (module docstring). Refuses a stale/copied owner, a parent bed that is not the owner's accepted source (witness mismatch, which also binds the physical
    descriptor), incomparable beds, an inconsistent packing and any provider, admission or later-batch failure; on every failure nothing was mutated. Only the columns whose depth or properties actually
    changed are reconciled."""
    if type(owner) is not NativeFloatingSoil:
        raise NativeFloatingError(f"owner must be a NativeFloatingSoil, got {type(owner).__name__}")
    _require_minted(owner, "owner")
    if not owner.is_current:
        raise NativeStaleError("the owner has been superseded by an adoption; stage against the current owner")
    if mode not in MODES:
        raise NativeFloatingError(f"mode must be one of {MODES}, got {mode!r}")
    cfg = owner.config
    quota = _quota_args(cfg, memory_budget_bytes, reserve_bytes, available_bytes_fn, host_memory_budget_bytes, host_reserve_bytes, host_available_bytes_fn)
    if cfg.backend == "cuda":
        from maple_syrup import native_floating_soil_cuda as ncuda

        return ncuda.stage_cuda(owner, parent_bed, child_bed, mode=mode, quota=quota, batch_cells=batch_cells)
    for bed in (parent_bed, child_bed):
        N, _V, _K, _C = _check_bed(bed, cfg)
    if (parent_bed.representation, parent_bed.geometry.ny, parent_bed.geometry.nx) != (child_bed.representation, child_bed.geometry.ny, child_bed.geometry.nx) \
            or not np.array_equal(parent_bed.active_mask, child_bed.active_mask) or not np.array_equal(child_bed.active_mask.reshape(-1), owner.column._maps["active"]):
        raise NativeFloatingError("parent and child beds must share the representation, grid and the owner's active mask")
    model = estimate_native_refresh_memory(parent_bed, child_bed, cfg, batch_cells=batch_cells, reserve_bytes=fsl._count(reserve_bytes, "reserve_bytes"))
    _admit(model, memory_budget_bytes, available_bytes_fn, "native floating refresh")
    if parent_bed.representation == "windowed":
        _windowed_checks(parent_bed, "stage_native_refresh(parent)")
        _windowed_checks(child_bed, "stage_native_refresh(child)")
    B = model["batch_cells"]
    active_flat = _flat_active(child_bed)
    dirty, child_witness, parent_witness = _scan(parent_bed, child_bed, active_flat, B)
    if parent_witness != owner.binding.source_witness:
        raise NativeStaleError("the parent bed is not the native source this owner was accepted against (cold witness mismatch, including the physical descriptor); the dirty set would not be proven")
    maps = owner.column._maps
    fc, sat, depth = (np.array(maps["theta_field_capacity"], copy=True), np.array(maps["theta_saturation"], copy=True), np.array(owner.depth_m, copy=True))
    ks = None if owner.ks_m_s is None else np.array(owner.ks_m_s, copy=True)  # depth mode HOLDS Ks as a property; properties mode updates the dirty cells
    _read_texture(child_bed, cfg, dirty, fc, sat, depth, model=model, batch_cells=B, update_properties=mode == "properties", ks=ks if mode == "properties" else None)
    changed, depth_changed = _changed_columns(owner, dirty, fc, sat, depth, B)  # only the dirty cells can differ; chunked, no (L, N) temporaries
    ks_changed = ks is not None and bool(np.any(ks[dirty] != owner.ks_m_s[dirty]))
    request = NativeRefreshRequest(owner.binding.lineage, owner.binding.generation, owner.binding.source_witness, child_witness, mode, int(B), int(model["required_bytes"]),
                                   _descriptor(child_bed), int(dirty.size), int(changed.size), int(depth_changed.size))
    dirty.flags.writeable = False
    changed.flags.writeable = False
    depth_changed.flags.writeable = False
    if changed.size == 0:
        zeros = np.zeros(N)
        zeros.flags.writeable = False
        op, hist, ks_out = owner.matrix_op, owner.matrix_history, owner.ks_m_s
        column, state, depth_out = owner.column, owner.state, owner.depth_m
        if ks is not None:
            ks.flags.writeable = False
        if depth_changed.size:
            # Only ACTUAL-depth bookkeeping changed (hydraulically equivalent geometry): publish the actual canonical depth in a genuine new column/state/operator bound to it, with the water bit for bit
            # the accepted inventory (same W, time, history values; nothing exported or returned; no cascade, no physics step). The accepted parent is untouched until adoption.
            depth.flags.writeable = False
            column = _build_column(cfg, np.asarray(maps["theta_field_capacity"]), np.asarray(maps["theta_saturation"]), depth, child_bed)
            state = fsl.floating_continued_state(column, np.array(owner.state.water_m, copy=True), float(owner.state.t_s))
            depth_out = depth
        if depth_changed.size or ks_changed:  # a new operator is needed when the column identity or the matrix conductivity changed; the persistent history is rebound exactly (no reset)
            op = _fresh_op(cfg, column, ks)
            hist = fm.rebind_floating_matrix_history(owner.matrix_op, owner.matrix_history, op).history
            ks_out = ks if ks_changed else owner.ks_m_s
        with _minting():
            return NativeRefreshCandidate(owner.binding, child_witness, mode, dirty, changed, depth_changed, column, state, op, hist, depth_out, ks_out, zeros, zeros, None, False,
                                          model, request, owner._authority, [False])
    old_water = np.asarray(owner.state.water_m)
    area = _area(child_bed)
    water, surf, exp_, acc = _reconcile_changed(cfg, maps, owner.depth_m, old_water, fc, sat, depth, changed, area, B, float(owner.state.t_s))
    new_column = _build_column(cfg, fc, sat, depth, child_bed)
    mask = np.asarray(new_column._maps["active"])
    new_state = fsl.floating_continued_state(new_column, np.ascontiguousarray(water), float(owner.state.t_s))
    w_before, w_after, whole_resid, whole_bound = _inventory_closure(water, old_water, mask, changed, B, area, acc, surf, exp_)
    if ks is not None:
        ks.flags.writeable = False
    new_op = _fresh_op(cfg, new_column, ks)
    rb = fm.rebind_floating_matrix_history(owner.matrix_op, owner.matrix_history, new_op)
    receipt = NativeReconcileReceipt(whole_before_m3=w_before, whole_after_m3=w_after, whole_residual_m3=whole_resid, whole_bound_m3=whole_bound, old_column_sha256=owner.column.sha256,
                                     new_column_sha256=new_column.sha256, **acc)
    for fresh in (depth, surf, exp_):  # the candidate takes ownership of these freshly built arrays: sealed in place (no second copy)
        fresh.flags.writeable = False
    with _minting():
        return NativeRefreshCandidate(owner.binding, child_witness, mode, dirty, changed, depth_changed, new_column, new_state, new_op, rb.history, depth, ks, surf, exp_, receipt, True, model, request,
                                      owner._authority, [False])


def _geometry_changed(config: NativeFloatingConfig, d_old: np.ndarray, d_new: np.ndarray) -> np.ndarray:
    """Per column: does the HYDRAULIC geometry differ between two actual canonical depths? With fixed nominal edges the D1 porous slab of layer k is `max(0, min(D, e_{k+1}) - e_k)` and the lower boundary is open
    only when the configured mode is open AND `D > H` (strict; `D == H` is closed rock). Two depths with bitwise-equal slabs and equal bottom flags are hydraulically EQUIVALENT (e.g. both deeper than H with
    unchanged properties; or, in a closed profile, `D == H` versus `D > H`): only bookkeeping differs. No epsilon, no tolerance."""
    e = config.nominal_edges_m
    H = float(e[-1])
    differ = np.zeros(d_new.shape, dtype=bool)
    for k in range(config.layers.n_layers):
        h_old = np.maximum(np.minimum(d_old, e[k + 1]) - e[k], 0.0)
        h_new = np.maximum(np.minimum(d_new, e[k + 1]) - e[k], 0.0)
        differ |= h_old != h_new
    if config.bottom_drainage == "open":
        differ |= (d_old > H) != (d_new > H)
    return differ


def _changed_columns(owner: Any, dirty: np.ndarray, fc: np.ndarray, sat: np.ndarray, depth: np.ndarray, B: int) -> tuple[np.ndarray, np.ndarray]:
    """`(reconcile set, raw-depth-changed set)` over the dirty cells, chunked. The RECONCILE set holds only the columns whose porous geometry/bottom condition (`_geometry_changed`) or FC/SAT properties actually
    changed: only those may call the zero-time cascade. The raw-depth set holds every column whose ACTUAL canonical depth differs (pure bookkeeping included): the published depth record and the column
    parameters always carry the actual depth."""
    return _changed_columns_parts(owner.config, owner.column._maps, owner.depth_m, dirty, fc, sat, depth, B)


def _changed_columns_parts(config: NativeFloatingConfig, maps: dict, old_depth: np.ndarray, dirty: np.ndarray, fc: np.ndarray, sat: np.ndarray, depth: np.ndarray, B: int) -> tuple[np.ndarray, np.ndarray]:
    """The body of `_changed_columns` from plain parts (the accepted held `maps`, the held/current canonical depth `old_depth` and the candidate fc/sat/depth): `(reconcile set, raw-depth-changed set)` with the SAME semantics (Ks is not a trigger)."""
    out, raw = [], []
    for b0 in range(0, dirty.size, B):
        d = dirty[b0:b0 + B]
        moved = depth[d] != old_depth[d]
        ch = _geometry_changed(config, old_depth[d], depth[d]) | np.any(fc[:, d] != maps["theta_field_capacity"][:, d], axis=0) | np.any(sat[:, d] != maps["theta_saturation"][:, d], axis=0)
        out.append(d[ch])
        raw.append(d[moved])
    cat = lambda parts: np.concatenate(parts).astype(np.int64, copy=False) if parts else np.zeros(0, dtype=np.int64)
    return cat(out), cat(raw)


def adopt_native_refresh(owner: NativeFloatingSoil, candidate: NativeRefreshCandidate) -> NativeFloatingSoil:
    """The one mutation point. Requires genuinely minted, exact-instance owner and candidate (a copy or reconstruction is refused), the owner to be the CURRENT one of its lineage, the candidate's minted parent
    binding to be that owner's binding and the candidate unconsumed. ALL refusable work (totals, receipts, the new binding and owner) is done first; only then is the candidate consumed and the lineage
    authority advanced, so a late failure leaves the old authority and the candidate retryable. Afterwards the old owner and the candidate are stale."""
    if type(owner) is not NativeFloatingSoil or type(candidate) is not NativeRefreshCandidate:
        raise NativeFloatingError("adopt_native_refresh needs a NativeFloatingSoil and a NativeRefreshCandidate")
    _require_minted(owner, "owner")
    _require_minted(candidate, "candidate")
    _require_minted(candidate.parent_binding, "parent binding")
    if candidate._authority is not owner._authority or candidate.parent_binding is not owner.binding:
        raise NativeStaleError("the candidate was staged against a different parent binding or lineage")
    if not owner.is_current:
        raise NativeStaleError("the owner has already been superseded")
    if candidate._consumed[0]:
        raise NativeStaleError("the candidate was already adopted")
    totals = dict(owner.totals)
    totals["refreshes"] += 1
    totals["cells_refreshed"] += candidate.n_dirty
    receipts = owner.receipts
    if candidate.receipt is not None:
        totals["surface_return_m3"] += candidate.receipt.surface_return_m3  # the receipt is the ONE source of the credited accounting
        totals["bottom_export_m3"] += candidate.receipt.bottom_export_m3
        receipts = (receipts + (candidate.receipt,))[-RECEIPT_HISTORY:]
    with _minting():
        binding = NativeBinding(owner.binding.lineage, owner.binding.generation + 1, candidate.child_witness, secrets.token_hex(16))
        new = NativeFloatingSoil(owner.config, candidate.column, candidate.state, candidate.matrix_op, candidate.matrix_history, candidate.depth_m, candidate.ks_m_s, binding, receipts, MappingProxyType(totals),
                                 owner._authority)
    candidate._consumed[0] = True
    owner._authority.current = binding
    return new


def retire_native_owner_to_lineage(owner: NativeFloatingSoil) -> None:
    """G3: the ONE public, legitimate release of a genuine CURRENT owner into a daily lineage (`floating_daily.start_floating_daily_lineage`, called LAST after every refusable construction). Requires an exact, genuinely
    minted, current `NativeFloatingSoil`; afterwards `owner.is_current` is False for good (the lineage authority is cleared), so the owner can never again be staged/adopted/handed to a Pond vertical against its stale
    initial water `W0`: the daily lineage's state is the only current water authority. It changes no array, mints nothing and is not reversible."""
    if type(owner) is not NativeFloatingSoil:
        raise NativeFloatingError("retire_native_owner_to_lineage needs a NativeFloatingSoil")
    _require_minted(owner, "owner")
    if not owner.is_current:
        raise NativeStaleError("the owner has already been superseded or released")
    owner._authority.current = None


def read_canonical_depth(bed: NativeBed, config: NativeFloatingConfig, indices: np.ndarray, scratch: np.ndarray, *, batch_cells: int = DEFAULT_BATCH_CELLS, available_bytes_fn: Callable[[], int] | None = None,
                         reserve_bytes: int = 0) -> np.ndarray:
    """HOT touched-column canonical depth: the SAME native compact reader as the cold refresh (native NumPy order, full flat base 0: `D = voxel.sum / (rho A) + active.sum / (rho A)` of the actual bed), restricted to
    the flat cells `indices` and filled into the caller's persistent `(N,)` float64 `scratch` (written ONLY at `indices`). Admitted from the actual bed shape BEFORE any row is read; bounded work: `batch_cells`
    compact columns at a time. Returns `scratch[indices]` as a fresh array. No texture, property, hash or whole-bed pass; the bed must be the native bed whose sediment is the ONE authority."""
    if type(scratch) is not np.ndarray or scratch.dtype != np.float64 or scratch.ndim != 1:
        raise NativeFloatingError("scratch must be a host float64 (N,) array")
    idx = np.asarray(indices)
    if idx.ndim != 1 or idx.dtype.kind not in "iu":
        raise NativeFloatingError("indices must be a 1-D integer array of flat cells")
    if idx.size == 0:
        return np.zeros(0)
    N, _V, _K, _C = _check_bed(bed, config)
    if scratch.shape != (N,) or int(idx.min()) < 0 or int(idx.max()) >= N or not bool(bed.active_mask.reshape(-1)[idx].all()):
        raise NativeFloatingError("indices must be distinct-or-repeated active flat cells of the bed and scratch must be (N,)")
    model = _estimate(None, bed, config, batch_cells, fsl._count(reserve_bytes, "reserve_bytes"))
    if available_bytes_fn is None:
        raise NativeFloatingError("read_canonical_depth needs an explicit available_bytes_fn (no unlimited reader)")
    hot = {"required_bytes": int(model["compact_bytes"]) + fsl._count(reserve_bytes, "reserve_bytes"), "terms": {"compact_reader": int(model["compact_bytes"])}}  # ONLY the compact reader: no property provider, no cold candidate
    _admit(hot, None, available_bytes_fn, "touched-column canonical depth")
    sel = np.unique(idx).astype(np.int64, copy=False)
    _read_texture(bed, config, sel, None, None, scratch, model=model, batch_cells=int(model["batch_cells"]), update_properties=False)
    return np.array(scratch[idx], copy=True)


def native_source_witness(bed: NativeBed, config: NativeFloatingConfig, *, batch_cells: int = DEFAULT_BATCH_CELLS, available_bytes_fn: Callable[[], int] | None = None, reserve_bytes: int = 0) -> str:
    """ONE cold streamed source witness of an actual native bed (the SAME validated `_scan` recipe v2 that accepted an owner: descriptor incl. the active-mask PATTERN, grid/geometry/ordered grain classes/rho/area/mass
    resolution, and the canonical rows). For an event-initialization handoff only (a COLD O(N V K) read, never a per-step path). Admitted from the bed's own shape (the scan term of the cold estimate) BEFORE any row is read;
    an explicit available-bytes reader is required."""
    if available_bytes_fn is None:
        raise NativeFloatingError("native_source_witness needs an explicit available_bytes_fn (no unlimited reader)")
    N, _V, _K, _C = _check_bed(bed, config)
    model = _estimate(None, bed, config, batch_cells, fsl._count(reserve_bytes, "reserve_bytes"))
    scan = {"required_bytes": int(model["terms"]["scan"]) + fsl._count(reserve_bytes, "reserve_bytes"), "terms": {"scan": int(model["terms"]["scan"])}}
    _admit(scan, None, available_bytes_fn, "native source witness")
    return _scan(None, bed, _flat_active(bed), int(model["batch_cells"]))[1]
