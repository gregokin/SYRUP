"""Phase 1 soil-profile texture: a read-only reduction of the actual MAPLE bed to user soil layers.

What this module does
---------------------
For every active cell it maps the ACTUAL native sediment column -- the separate active-layer cap
(`ActiveLayerState.mass_kg`, already extracted from the voxels) followed by the occupied subactive voxels
(`VoxelColumnState.mass_kg`, voxel 0 deepest) -- onto the caller's fixed soil layers and returns, per layer,
per cell and per grain class, the overlap-weighted class MASS (kg) and the resulting class FRACTIONS. Grain
classes keep their native order and metadata; nothing is relabelled as clay/silt/sand (an aggregate class is an
aggregate class, not a primary texture). No hydraulic property is inferred: the user supplies those explicitly in
Phase 1 (user decision 2026-11-10); a declared texture-to-property method is Phase 2 work.

Declared physics / geometry (one definition, used by all backends)
------------------------------------------------------------------
Depth `z` is measured downward from the actual material surface of the cell. The material segments are, in
order: the active cap, then the occupied voxels from the highest index downward. Each segment's height is its
ACTUAL mass divided by the native packing: `h_s = (sum_k M_sk) / (bulk_density_kg_m3 * dx_m * dy_m)` -- the same
formula as MAPLE's `compute_active_layer_actual_thickness_m` -- never the nominal `voxel_dz_m`, so empty voxel
capacity is not soil and a partially filled top voxel has its actual height. A segment of exactly zero mass is
skipped. Segment `s` occupies `[top_s, top_s + h_s)` with `top_0 = 0` and `top_{s+1} = top_s + h_s` (a sequential
fold). Soil layer `l` occupies `[e_l, e_{l+1})` with `e_0 = 0`. The overlap is
`o_ls = max(0, min(top_s + h_s, e_{l+1}) - max(top_s, e_l))` and the layer receives
`M_lk += (o_ls / h_s) * M_sk` -- a fraction of the segment's ACTUAL class masses (a 0.02 m top voxel overlapping a
layer by 0.01 m contributes half of its actual class masses). Accumulation is sequential over segments in the
order above. Fractions are `M_lk / sum_k M_lk` where that sum is positive, else 0.

Class totals (`sum_k`) are this module's DECLARED sequential fold `((c0 + c1) + c2) + ...` -- the same declared
reduction MAPLE's own `windowed_inventory.sequential_class_fold` uses -- so NumPy, Numba and CUDA evaluate the same
IEEE operations in the same order. It is NOT claimed to be bitwise MAPLE's canonical library `mass_kg.sum(axis=-1)`
(used by `compute_active_layer_actual_thickness_m` and the dense validator): NumPy's pairwise reduction groups
8 or more terms differently, so for `n_classes >= 8` a segment height here may differ from the native diagnostic
by summation roundoff (bounded by MAPLE's `summation_error_bound_kg(n_classes, total)`); for fewer classes the two
are the same arithmetic on this NumPy (recorded by `tests/soil_profile_texture/test_backends.py`, a probe, not a
universal claim). The NumPy reference is the definition; bitwise agreement of the compiled backends is tested.

Refusals (clear errors before any output is published)
------------------------------------------------------
* `SoilProfileGeometryError` when an ACTIVE cell does not represent enough material to cover every layer, or
  when the top soil layer is thinner than the cell's actual active cap (the cap is counted once, inside the top
  layer). "Enough" is decided with a DECLARED, operation-derived FP64 bound -- not a user allowance:
  `bound_c = summation_error_bound_kg(n_terms, 1.0) * max(represented_depth_c, bottom_edge)` where
  `summation_error_bound_kg` is MAPLE's own summation contract (`4 * n_terms * eps`) and `n_terms =
  (n_voxels + 1) * (n_classes + 5) + n_layers + 2` counts every floating-point operation that can enter a
  layer's covered length (per segment: `n_classes - 1` class adds, one division, one top add, and per layer one
  max, one min, one subtraction, one accumulation; plus the sequential layer edges and the final comparison).
  Every operand of those operations is at most the cell's represented depth or the deepest layer edge, so the
  scale is their maximum. A deficit above that bound is a genuinely missing layer and is refused; nothing is
  invented below the bed, the profile is never truncated, and a partially covered layer is never renormalised as
  full. There is no user-adjustable allowance.
* Inactive cells (mask False) get zeros, coverage 0 and no refusal: no fictitious profile is constructed.
* Native validity is checked ONCE at build with MAPLE's own validators (see "Native APIs"); later water steps
  must not re-scan whole columns.

Layer geometry contract
-----------------------
`SoilLayerGeometry` canonicalises itself on construction (fresh read-only float64 copies, finite positive
thicknesses, finite strictly advancing sequential edges; supplied edges must equal the recomputed ones bitwise), so
the direct constructor, `layer_geometry` and `dataclasses.replace` all enforce the same contract. The public entry
point additionally REBUILDS the geometry from a fresh copy of the thicknesses and re-checks the supplied edges, so a
caller re-enabling writeability on the arrays and mutating them cannot make an accepted texture carry a stale or
inconsistent layer identity: the returned texture owns its own canonical copy.

Native APIs used (read-only)
----------------------------
* `maple_syrup.sediment_bed.is_windowed_bed` -- the strict class-identity binding: exactly the dense
  `VoxelColumnState` or exactly the genuine `WindowedVoxelColumnState`; subclasses/duck types raise
  `BedIntegrationError` (propagated). A `BedState` with only the two bed fields set is used as the probe.
* Dense: `maple.surface.active_layer.validation.check_active_layer_voxel_partition` per batch (shape/dtype/
  finite/nonnegative/overfill/packing/partition rules), on a `(1, B, ...)`-shaped view with a geometry whose
  `(ny, nx)` is replaced by `(1, B)` -- the validator reads no other geometry field for the horizontal shape.
* Windowed: `require_published_column` (an unpublished `pending_deep` journal is refused; nothing is published
  here), `check_windowed_column_cheap`, `check_active_layer_state`, and ALWAYS MAPLE's trusted-boundary streamed
  validator `validate_windowed_column_full` (dense validator per chunk, deep metadata bitwise against the bound
  version). There is no cheaper mode: a budget too small for the native validation refuses the build, it never
  degrades the checks. Columns are read with `gather_full_columns` in bounded batches of ACTIVE cells only
  (version below `base`, resident window inside, zero above -- the native overlay rule, never a stale deep read,
  never `windowed_to_dense`). Every gathered batch is additionally validated with the dense partition validator.

Backends, residency and memory
------------------------------
`backend` is the COMPUTE backend: "numpy" (reference, vectorised over the cells of a batch), "numba" (compiled
serial cell loop, `soil_profile_texture_numba`), "cuda" (one thread per cell, `soil_profile_texture_cuda`). Dense
inputs must already live where the backend computes (host for numpy/numba, CuPy for cuda); a mismatch is refused,
nothing is moved implicitly. Windowed reads are host I/O by construction, so windowed computes on the host
("numpy"/"numba" only) and the REDUCED outputs are moved to the window's namespace once at the end.

Memory admission happens in two steps, both BEFORE any array of this call is allocated: a metadata-only
conservative lower bound (every cell assumed active) from shapes and headers alone, then the final estimate once
the active count is known. `estimate_texture_memory` lists every component: mandatory outputs, host staging for a
device-resident windowed state, mask/flag/index copies (including the device mask copy and per-batch index/mask
temporaries), the native validators' own temporaries (cheap window checks over the whole resident window, the
streamed full validator's chunk buffers incl. up to two materialised overlay chunks, the dense partition validator
per batch), the native gather's advanced-index and overlay temporaries (two column batches, two window-row
batches), compute scratch, the deep store's own declared mapping-cache bound, a fixed declared allowance for the
streamed identity hashing and this call's small Python objects, a declared per-class / per-layer allowance for
the O(K) + O(L) metadata records (a many-class native bed is an admissible input), and the caller's reserve. Batch
buffers are released before the next gather/kernel call (one batch alive at a time; the estimate counts one). EXCLUDED and
reported separately (`input_bytes_excluded`): the caller's own native arrays (dense column and active layer, or
resident window, base, deep metadata and active layer). Not counted: the deep store's files on disk, CuPy
memory-pool retention policy, and one-time process costs (module imports, Numba compilation, CUDA kernel load).
No whole-process bound is promised. The suite checks with `tracemalloc` that a warm host build's peak stays below
`required_bytes`.

Scope
-----
Phase 1 layers are fixed within a run; nothing here follows erosion/deposition (Phase 2 remapping), stores water,
or binds a runtime source lock / combined restart. The returned `source` record is a DESCRIPTIVE identity of what
was read (native module file hashes, geometry, classes, layers, windowed version/digest) for later coupling
records, not a qualification claim.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "BACKENDS",
    "COVERAGE_POLICIES",
    "DEFAULT_BATCH_CELLS",
    "FLAG_INSUFFICIENT_DEPTH",
    "FLAG_TOP_LAYER_THINNER_THAN_CAP",
    "SCHEMA",
    "SoilLayerGeometry",
    "SoilProfileBackendUnavailableError",
    "SoilProfileGeometryError",
    "SoilProfileMemoryError",
    "SoilProfileTexture",
    "SoilProfileTextureError",
    "TextureMemoryEstimate",
    "build_soil_profile_texture",
    "canonical_layers",
    "coverage_bound_factor",
    "coverage_bound_terms",
    "estimate_texture_memory",
    "identity_modules",
    "layer_geometry",
    "partial_stage_bytes",
    "reference_texture_batch",
    "soil_profile_texture_from_bed",
]

SCHEMA = "maple_syrup.soil_profile_texture.v2"
BACKENDS = ("numpy", "numba", "cuda")
DEFAULT_BATCH_CELLS = 4096  # equals MAPLE's DEFAULT_MAX_GATHER_COLUMNS; never raised silently
FLAG_INSUFFICIENT_DEPTH = 1
FLAG_TOP_LAYER_THINNER_THAN_CAP = 2
COVERAGE_POLICIES = ("full", "partial_over_base")
COVERAGE_POLICY_FULL, COVERAGE_POLICY_PARTIAL = COVERAGE_POLICIES
# Working stage of the opt-in coverage validation (per chunk of B cells, in the outputs' namespace): at most 6 live float64
# (B,) vectors (tolerance, expected-coverage chain, difference chain) and 8 live bool (B,) vectors (finite/sign/compare/OR
# chain), plus the B-byte host flag copy. Counted by `partial_stage_bytes`, admitted by `_admit_partial_stage`.
PARTIAL_STAGE_F64_VECTORS = 6
PARTIAL_STAGE_BOOL_VECTORS = 8
COVERAGE_BOUND_FUNCTION = "maple.surface.voxels._numerics.summation_error_bound_kg"
IDENTITY_HASH_CHUNK_BYTES = 8192  # streamed native-module hashing: one read buffer alive per read
IDENTITY_STREAM_BUFFERS = 2  # the current read and the one it replaces (released after `digest.update`)
# Declared constant for this call's small Python objects and temporaries (geometry replace, exception text, the
# source record, dataclass instances, index scalars): a stated allowance, not a measurement, counted on the host.
INTERPRETER_OVERHEAD_BYTES = 32 * 1024
# Declared per-class / per-layer allowances for the O(K) + O(L) Python metadata this call builds (class id strings,
# diameter/density floats, the source record's per-class dicts and per-layer lists, the result tuples). Measured
# ~240 B per class on CPython 3.12 (review `root_wide_class_memory.json`); declared at about twice that.
CLASS_METADATA_BYTES = 512
LAYER_METADATA_BYTES = 256
_F8 = 8
_MAX_REPORTED_CELLS = 8


class SoilProfileTextureError(ValueError):
    """Invalid argument, unsupported native state or failed native validation; nothing was published."""


class SoilProfileGeometryError(SoilProfileTextureError):
    """An active cell cannot represent the requested layers (insufficient material depth, or top layer thinner
    than the actual active cap). Nothing is invented or truncated."""


class SoilProfileMemoryError(MemoryError):
    """The explicit caller budget is below the estimated need; nothing was allocated."""


class SoilProfileBackendUnavailableError(RuntimeError):
    """The requested compute backend (Numba / CuPy + CUDA device) is not usable; there is no silent fallback."""


# --- layers -----------------------------------------------------------------------------------------------------------
def _canonical_thickness(thickness_m: Any) -> np.ndarray:
    if isinstance(thickness_m, (str, bytes)):
        raise SoilProfileTextureError("thickness_m must be a 1-D sequence of positive layer thicknesses (m)")
    try:
        thickness = np.array(thickness_m, dtype=np.float64, copy=True)  # always a fresh private array
    except (TypeError, ValueError) as exc:
        raise SoilProfileTextureError(f"thickness_m is not numeric: {exc}") from exc
    if thickness.ndim != 1 or thickness.shape[0] < 1:
        raise SoilProfileTextureError(f"thickness_m must be 1-D with at least one layer, got shape {thickness.shape}")
    if not np.all(np.isfinite(thickness)) or not np.all(thickness > 0.0):
        raise SoilProfileTextureError("every layer thickness must be finite and > 0")
    return thickness


def _sequential_edges(thickness: np.ndarray) -> np.ndarray:
    """`e_0 = 0, e_{i+1} = e_i + t_i` (the declared sequential fold); finite and STRICTLY advancing or refused."""
    edges = np.empty(thickness.shape[0] + 1, dtype=np.float64)
    edges[0] = 0.0
    for i in range(thickness.shape[0]):
        with np.errstate(over="ignore"):  # an overflow is refused explicitly just below, not warned about
            edges[i + 1] = edges[i] + thickness[i]
        if not np.isfinite(edges[i + 1]):
            raise SoilProfileTextureError(
                f"layer edges overflow FP64: the sequential depth edge after layer {i} is not finite")
        if not edges[i + 1] > edges[i]:
            raise SoilProfileTextureError(
                f"layer {i} (thickness {thickness[i]!r} m) does not advance the sequential depth edge "
                f"{edges[i]!r} m in FP64; the layer would have zero represented thickness -- refused")
    return edges


@dataclass(frozen=True)
class SoilLayerGeometry:
    """Fixed Phase 1 soil layers: positive finite thicknesses (m), top layer first; `depth_edges_m[0] == 0` is the
    actual material surface. Canonicalised on construction: a fresh read-only float64 copy of the thicknesses
    (validated) and the edges DERIVED by the declared sequential fold (finite, strictly advancing). The edges are
    an `init=False` field: they cannot be supplied to the constructor or to `dataclasses.replace` (a `TypeError`),
    so no caller-authored edge vector can ever be recorded; `replace(..., thickness_m=...)` recomputes them.
    `canonical_layers` (the public boundary) additionally re-checks an instance's current edges against a fresh
    recomputation, so arrays made writeable and mutated afterwards are detected and refused."""

    thickness_m: np.ndarray  # (L,)
    depth_edges_m: np.ndarray = field(init=False, repr=False, compare=False)  # (L + 1,), derived

    def __post_init__(self) -> None:
        thickness = _canonical_thickness(self.thickness_m)
        edges = _sequential_edges(thickness)
        thickness.flags.writeable = False
        edges.flags.writeable = False
        object.__setattr__(self, "thickness_m", thickness)
        object.__setattr__(self, "depth_edges_m", edges)

    @property
    def n_layers(self) -> int:
        return int(self.thickness_m.shape[0])

    @property
    def bottom_depth_m(self) -> float:
        return float(self.depth_edges_m[-1])


def layer_geometry(thickness_m: Any) -> SoilLayerGeometry:
    """Validate layer thicknesses (1-D, >= 1 entry, finite, > 0) and form the sequential depth edges."""
    return SoilLayerGeometry(thickness_m=thickness_m)


def canonical_layers(layers: Any) -> SoilLayerGeometry:
    """A FRESH canonical `SoilLayerGeometry` for the public boundary: thicknesses copied and revalidated, edges
    recomputed, and -- for a `SoilLayerGeometry` input -- its current `depth_edges_m` re-checked bitwise against
    the fresh recomputation (a record whose arrays were made writeable and mutated is refused). The caller's arrays
    are never retained."""
    if isinstance(layers, SoilLayerGeometry):
        fresh = SoilLayerGeometry(thickness_m=np.array(layers.thickness_m, dtype=np.float64, copy=True))
        try:
            given = np.array(layers.depth_edges_m, dtype=np.float64, copy=True)
        except (TypeError, ValueError) as exc:
            raise SoilProfileTextureError(f"depth_edges_m is not numeric: {exc}") from exc
        if given.shape != fresh.depth_edges_m.shape or given.tobytes() != fresh.depth_edges_m.tobytes():
            raise SoilProfileTextureError(
                "inconsistent layer record: depth_edges_m is not the sequential fold of thickness_m "
                f"(given {given.tolist()}, recomputed {fresh.depth_edges_m.tolist()}); the record was altered after "
                "construction -- refused")
        return fresh
    return SoilLayerGeometry(thickness_m=layers)


# --- coverage bound (declared, operation-derived) ------------------------------------------------------------------------
def coverage_bound_terms(n_voxels: int, n_classes: int, n_layers: int) -> int:
    """Number of FP64 operations that can enter one layer's covered length of one cell (see module docstring):
    `(n_voxels + 1) * (n_classes + 5) + n_layers + 2`."""
    V = _checked_int(n_voxels, "n_voxels", minimum=1)
    K = _checked_int(n_classes, "n_classes", minimum=1)
    L = _checked_int(n_layers, "n_layers", minimum=1)
    return (V + 1) * (K + 5) + L + 2


def coverage_bound_factor(n_voxels: int, n_classes: int, n_layers: int) -> float:
    """MAPLE's `summation_error_bound_kg(n_terms, 1.0)`: the dimensionless factor multiplied, per cell, by
    `max(represented_depth, bottom_edge)` to give the covered-length bound in metres."""
    from maple.surface.voxels._numerics import summation_error_bound_kg

    return float(summation_error_bound_kg(coverage_bound_terms(n_voxels, n_classes, n_layers), 1.0))


# --- memory -----------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class TextureMemoryEstimate:
    """Upper-bound bytes THIS call allocates (caller-owned native inputs excluded, reported separately).
    `required_bytes = host_bytes + device_bytes + reserve_bytes` is compared with the caller's budget."""

    representation: str
    backend: str
    output_namespace: str
    n_cells: int
    n_active: int  # the count used for the estimate (the cell count itself in the metadata-only lower bound)
    n_layers: int
    n_classes: int
    n_voxels: int
    batch_cells: int
    output_bytes: int  # mandatory outputs in the output namespace
    output_staging_bytes: int  # host staging of the outputs (windowed with a device window only)
    mask_index_bytes: int  # host mask copy, flags, active indices, device mask copy, per-batch mask/index temporaries
    batch_input_bytes: int  # native gather of ONE batch: two column batches, two window-row batches, index/active rows
    validation_bytes: int  # native validators' temporaries: cheap window checks, full streamed validator, per-batch dense
    scratch_bytes: int  # compute scratch of one batch (numpy vectors; batch outputs; cuda edges)
    native_cache_bytes: int  # the deep store's declared mapping-cache bound (`mapped_bytes_bound`), host
    fixed_overhead_bytes: int  # streamed identity-hash chunks + the declared interpreter allowance, host
    metadata_bytes: int  # declared O(n_classes) + O(n_layers) Python metadata (class/layer records, result tuples), host
    reserve_bytes: int
    host_bytes: int
    device_bytes: int
    required_bytes: int
    input_bytes_excluded: int  # the caller's own native arrays (dense column + active layer, or resident window set)
    batch_reduced: bool = False
    metadata_only: bool = False  # True for the pre-admission lower bound (every cell assumed active)


def _checked_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise SoilProfileTextureError(f"{name} must be a non-Boolean integer, got {value!r}")
    if int(value) < minimum:
        raise SoilProfileTextureError(f"{name} must be >= {minimum}, got {int(value)}")
    return int(value)


def _dense_validator_bytes(B: int, V: int, K: int) -> int:
    """MAPLE's `check_active_layer_voxel_partition` on a `(1, B, V, K)` batch: voxel totals, local scale and its two
    `maximum` temporaries, the tolerance difference/product pair, the `full`/`empty`/`partial` masks, the int64 code
    and its diff, the model-resolution mask, the partial count, the active-layer flags/totals/tolerance vectors and
    the subactive/elevation vectors -- counted as 11 float64 `(B, V)` arrays, 6 `(B, V)` bytes, 2 `(B, K)` bytes
    and 12 `(B,)` float64."""
    return 11 * _F8 * B * V + 6 * B * V + 2 * B * K + 12 * _F8 * B


def estimate_texture_memory(
    *,
    representation: str,
    backend: str,
    output_namespace: str,
    n_cells: int,
    n_active: int,
    n_layers: int,
    n_classes: int,
    n_voxels: int,
    batch_cells: int,
    window_voxels: int = 0,
    chunk_cells: int = 0,
    native_cache_bytes: int = 0,
    reserve_bytes: int = 0,
    input_bytes_excluded: int = 0,
    batch_reduced: bool = False,
    metadata_only: bool = False,
) -> TextureMemoryEstimate:
    """Pure arithmetic. `batch_cells` is the number of cells processed per batch (dense: a view of the resident
    column; windowed: a fresh native gather). Windowed estimates require `window_voxels`, the deep-store
    `chunk_cells` (the native full validator's unit) and the store's `native_cache_bytes`."""
    if representation not in ("dense", "windowed"):
        raise SoilProfileTextureError(f"representation must be 'dense' or 'windowed', got {representation!r}")
    if backend not in BACKENDS:
        raise SoilProfileTextureError(f"backend must be one of {BACKENDS}, got {backend!r}")
    if output_namespace not in ("numpy", "cupy"):
        raise SoilProfileTextureError(f"output_namespace must be 'numpy' or 'cupy', got {output_namespace!r}")
    N = _checked_int(n_cells, "n_cells", minimum=1)
    n_active = _checked_int(n_active, "n_active")
    L = _checked_int(n_layers, "n_layers", minimum=1)
    K = _checked_int(n_classes, "n_classes", minimum=1)
    V = _checked_int(n_voxels, "n_voxels", minimum=1)
    B = _checked_int(batch_cells, "batch_cells", minimum=1)
    W = _checked_int(window_voxels, "window_voxels")
    C = _checked_int(chunk_cells, "chunk_cells")
    cache = _checked_int(native_cache_bytes, "native_cache_bytes")
    reserve = _checked_int(reserve_bytes, "reserve_bytes")
    excluded = _checked_int(input_bytes_excluded, "input_bytes_excluded")
    if representation == "windowed" and (W < 1 or C < 1):
        raise SoilProfileTextureError("windowed estimates require window_voxels >= 1 and chunk_cells >= 1")
    device_output = output_namespace == "cupy"

    # Mandatory outputs: mass + fraction (L, N, K), total + coverage (L, N), depth + cap (N).
    outputs = (2 * L * N * K + 2 * L * N + 2 * N) * _F8
    # Per-batch kernel outputs before placement: (L, B, K) x2, (L, B) x2, (B,) x2, flags (B,).
    batch_outputs = (2 * L * B * K + 2 * L * B + 2 * B) * _F8 + B
    if backend == "numpy":
        # heights/tops/bottoms/overlap/mask vectors (B,) x ~12, class totals (B, V), (B, K) products x3
        scratch = (12 * B + B * V + 3 * B * K) * _F8 + batch_outputs
    elif backend == "numba":
        scratch = batch_outputs + 8 * B  # the kernel keeps scalars per cell; contiguity checks allocate nothing
    else:
        scratch = batch_outputs + (L + 1) * _F8  # edges upload; no per-cell scratch
    # Identity hashing streams the modules ONE AT A TIME (the module count does not change the peak).
    fixed = IDENTITY_STREAM_BUFFERS * IDENTITY_HASH_CHUNK_BYTES + INTERPRETER_OVERHEAD_BYTES
    # Class/layer metadata scales with K and L (a many-class native bed is an admissible input).
    metadata = CLASS_METADATA_BYTES * K + LAYER_METADATA_BYTES * L
    # Host mask copy + flags (N bytes each) always; the per-batch uint8 mask (B).
    mask_index_host = 2 * N
    mask_index_device = 0
    staging = 0
    batch_input = 0
    host = 0
    device = 0
    if representation == "dense":
        validation = _dense_validator_bytes(B, V, K)
        if backend == "cuda":
            mask_index_device = N + B  # device mask copy and the per-batch uint8 view copy
            device = outputs + validation + scratch + mask_index_device
            host = mask_index_host + fixed + metadata
        else:
            mask_index_host += B
            host = outputs + validation + scratch + mask_index_host + fixed + metadata
    else:
        # Active flat indices (int64, N worst case), per-batch idx (B int64) and its device copy when the window is
        # on the device.
        mask_index_host += 8 * N + 8 * B
        if device_output:
            mask_index_device = 8 * B
        # Native gather of ONE batch: `gather_columns` output (B, V, K) + the advanced-index chunk temporary (<= B, V,
        # K) + overlay row temporaries (counted inside the second column batch); window rows advanced-indexed in the
        # window's namespace (B, W, K) + the host copy; base rows x2; chunk_of/sel/local/unique index vectors; the
        # overlay's `above` mask (B, V) and `rows`; the active rows (B, K) + their gather temporary.
        batch_input = 2 * _F8 * B * V * K + 2 * _F8 * B * W * K + B * V + 2 * _F8 * B * K + 7 * _F8 * B
        # Native cheap checks over the WHOLE resident set: one window-size bool (isfinite / negative), deep flags,
        # slot-0 totals and base masks; `check_active_layer_state`: two (N, K) bools and ~6 (N,) float64.
        cheap = N * W * K + 2 * N * K + 3 * _F8 * N + 2 * N + 2 * N * K + 6 * _F8 * N
        # Native full validator per chunk: `full` (C, V, K) + up to two materialised overlay chunk buffers (bounded
        # two-buffer overlap of `iter_chunks`), host window rows (C, W, K) + device temporary, base rows, the overlay
        # `above` mask, the dense validator on the chunk, the recomputed metadata (window copy (C, W, K), prefix
        # (C, K), totals (C, V), room/total (C,)) and the bitwise comparison copies (C, K) x2.
        full = (3 * _F8 * C * V * K + 2 * _F8 * C * W * K + 2 * _F8 * C + C * V + _dense_validator_bytes(C, V, K)
                + _F8 * C * W * K + 3 * _F8 * C * K + _F8 * C * V + 2 * _F8 * C)
        validation = cheap + full + _dense_validator_bytes(B, V, K)
        host = outputs + mask_index_host + batch_input + validation + scratch + cache + fixed + metadata
        if device_output:
            staging = outputs
            device = outputs + mask_index_device + _F8 * B * W * K  # outputs, idx copy, window-row gather temporary
            host = staging + mask_index_host + batch_input + validation + scratch + cache + fixed + metadata
    required = host + device + reserve
    return TextureMemoryEstimate(
        representation, backend, output_namespace, N, n_active, L, K, V, B, outputs, staging,
        mask_index_host + mask_index_device, batch_input, validation, scratch, cache, fixed, metadata, reserve, host,
        device, required, excluded, batch_reduced, metadata_only,
    )


# --- result -----------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class SoilProfileTexture:
    """Read-only Phase 1 texture of one native bed on fixed layers. Arrays live in `namespace` (the native state's
    own namespace); cells are flattened in C order (`cell = iy * nx + ix`), the same `(L, N)` layout as
    `maple_syrup.subsurface`. Grain classes keep MAPLE's order (`class_ids`). `layers` is this texture's own
    canonical copy (never the caller's object)."""

    layer_mass_kg: Any  # (L, N, K) overlap-weighted actual class mass of each layer
    layer_fraction: Any  # (L, N, K) mass fractions (0 where the layer holds no mass / cell inactive)
    layer_total_kg: Any  # (L, N) sequential class fold of layer_mass_kg
    coverage_m: Any  # (L, N) material length covering the layer (== thickness within the declared bound when accepted)
    represented_depth_m: Any  # (N,) active cap + occupied voxel heights (0 for inactive cells)
    active_cap_thickness_m: Any  # (N,) actual active-layer thickness (0 for inactive cells)
    active_mask: np.ndarray  # (N,) host bool
    layers: SoilLayerGeometry
    ny: int
    nx: int
    class_ids: tuple[str, ...]
    class_diameter_m: tuple[float, ...]
    class_is_aggregate: tuple[bool, ...]
    class_particle_density_kg_m3: tuple[float, ...]
    cell_area_m2: float
    bulk_density_kg_m3: float
    coverage_bound_terms: int  # n_terms of the declared bound
    coverage_bound_factor: float  # summation_error_bound_kg(n_terms, 1.0); bound_c = factor * max(depth_c, bottom edge)
    namespace: str
    backend: str
    representation: str
    memory: TextureMemoryEstimate
    source: dict[str, Any]
    coverage_policy: str = COVERAGE_POLICY_FULL  # "full" (historical strict) | "partial_over_base" (explicit opt-in)

    @property
    def n_layers(self) -> int:
        return self.layers.n_layers

    @property
    def n_cells(self) -> int:
        return int(self.active_mask.shape[0])

    @property
    def n_classes(self) -> int:
        return len(self.class_ids)

    def source_json(self) -> str:
        return json.dumps(self.source, sort_keys=True, indent=2)


# --- the NumPy reference kernel (definition of the arithmetic) ---------------------------------------------------------
def _fold_classes(a: Any) -> Any:
    """`((c0 + c1) + c2) + ...` over the trailing class axis; no `+ 0.0` prefix (bitwise what the loops do)."""
    t = a[..., 0]
    for k in range(1, a.shape[-1]):
        t = t + a[..., k]
    return t


def reference_texture_batch(xp: Any, active_kg: Any, voxels_kg: Any, mask: Any, edges_m: np.ndarray,
                            denominator_kg_m: float, bound_factor: float) -> tuple[Any, ...]:
    """The declared arithmetic, vectorised over the cells of ONE batch in namespace `xp`.

    `active_kg (B, K)`, `voxels_kg (B, V, K)` (voxel 0 deepest), `mask (B,) bool`, `edges_m (L + 1,)` host,
    `denominator_kg_m = bulk_density * dx * dy`, `bound_factor` the dimensionless coverage bound factor. Returns
    `(mass (L, B, K), fraction (L, B, K), total (L, B), coverage (L, B), depth (B,), cap (B,), flags (B,) uint8)`.
    Inactive cells produce zeros and flag 0."""
    B, V, K = (int(s) for s in voxels_kg.shape)
    L = int(edges_m.shape[0]) - 1
    edges = [float(e) for e in edges_m]
    mask = mask.astype(np.bool_)
    total_active = _fold_classes(active_kg)  # (B,)
    h_active = total_active / denominator_kg_m
    total_voxel = _fold_classes(voxels_kg)  # (B, V)
    h_voxel = total_voxel / denominator_kg_m
    mass = xp.zeros((L, B, K), dtype=np.float64)
    coverage = xp.zeros((L, B), dtype=np.float64)
    z = xp.zeros((B,), dtype=np.float64)
    for s in range(V + 1):
        if s == 0:
            h, m = h_active, active_kg
        else:
            v = V - s
            h, m = h_voxel[:, v], voxels_kg[:, v, :]
        positive = h > 0.0
        h_safe = xp.where(positive, h, 1.0)
        top = z
        bottom = z + h
        for l in range(L):
            lo = xp.maximum(top, edges[l])
            hi = xp.minimum(bottom, edges[l + 1])
            o = hi - lo
            valid = positive & (o > 0.0) & mask
            w = xp.where(valid, o / h_safe, 0.0)
            mass[l] = mass[l] + w[:, None] * m
            coverage[l] = coverage[l] + xp.where(valid, o, 0.0)
        z = xp.where(positive, bottom, z)
    depth = xp.where(mask, z, 0.0)
    cap = xp.where(mask, h_active, 0.0)
    total = _fold_classes(mass)  # (L, B): sequential fold over K
    has_mass = total > 0.0
    total_safe = xp.where(has_mass, total, 1.0)
    fraction = xp.where(has_mass[:, :, None], mass / total_safe[:, :, None], 0.0)
    # Declared bound: factor * max(represented depth, deepest edge), per cell.
    tol = float(bound_factor) * xp.maximum(z, edges[L])
    flags = xp.zeros((B,), dtype=np.uint8)
    for l in range(L):
        thickness = edges[l + 1] - edges[l]
        short = mask & ((thickness - coverage[l]) > tol)
        flags = flags | xp.where(short, FLAG_INSUFFICIENT_DEPTH, 0).astype(np.uint8)
    thick0 = edges[1] - edges[0]
    flags = flags | xp.where(mask & ((h_active - thick0) > tol), FLAG_TOP_LAYER_THINNER_THAN_CAP, 0).astype(np.uint8)
    return mass, fraction, total, coverage, depth, cap, flags


# --- native binding and validation helpers -----------------------------------------------------------------------------
def _representation(voxel_column: Any, active_layer: Any) -> str:
    """'dense' or 'windowed' through SYRUP's strict class-identity check (BedIntegrationError propagates)."""
    from maple_syrup.sediment_bed import BedState, is_windowed_bed

    probe = BedState(voxel_column=voxel_column, active_layer=active_layer, water=None, ledger=None,
                     committed_topography=None, sediment_availability=None)  # binding probe only
    return "windowed" if is_windowed_bed(probe) else "dense"


def _class_metadata(grain_classes: Any, n_classes: int) -> tuple[tuple, tuple, tuple, tuple]:
    classes = getattr(grain_classes, "classes", None)
    if not isinstance(classes, tuple) or not classes:
        raise SoilProfileTextureError("grain_classes must be a MAPLE GrainClassSet with a non-empty `classes` tuple")
    ids, diameters, aggregates, densities = [], [], [], []
    for c in classes:
        for name in ("class_id", "diameter_m", "is_aggregate", "particle_density_kg_m3"):
            if not hasattr(c, name):
                raise SoilProfileTextureError(f"grain class {c!r} lacks {name}")
        ids.append(str(c.class_id))
        diameters.append(float(c.diameter_m))
        aggregates.append(bool(c.is_aggregate))
        densities.append(float(c.particle_density_kg_m3))
    if len(ids) != n_classes:
        raise SoilProfileTextureError(
            f"grain_classes declares {len(ids)} class(es) but the native bed carries {n_classes}; the class axis is "
            "bound to MAPLE's class order and count")
    if len(set(ids)) != len(ids):
        raise SoilProfileTextureError("grain class ids must be distinct")
    return tuple(ids), tuple(diameters), tuple(aggregates), tuple(densities)


def _geometry_numbers(geometry: Any) -> tuple[int, int, float, float, float, float, float]:
    try:
        ny, nx = int(geometry.ny), int(geometry.nx)
        dx, dy = float(geometry.dx_m), float(geometry.dy_m)
        rho, dz, cap = float(geometry.bulk_density_kg_m3), float(geometry.voxel_dz_m), float(geometry.active_layer_thickness_m)
    except (AttributeError, TypeError, ValueError) as exc:
        raise SoilProfileTextureError(f"geometry is not a MAPLE GeometrySpec: {exc}") from exc
    for name, value in (("dx_m", dx), ("dy_m", dy), ("bulk_density_kg_m3", rho), ("voxel_dz_m", dz),
                        ("active_layer_thickness_m", cap)):
        if not np.isfinite(value) or value <= 0.0:
            raise SoilProfileTextureError(f"geometry.{name} must be finite and > 0, got {value!r}")
    if ny < 1 or nx < 1:
        raise SoilProfileTextureError(f"geometry grid must be >= 1 x 1, got {(ny, nx)}")
    return ny, nx, dx, dy, rho, dz, cap


def _check_mask_metadata(active_mask: Any, ny: int, nx: int) -> None:
    """Shape/dtype contract of the mask WITHOUT copying it (used before memory admission)."""
    from maple.core.backend import is_array

    if not is_array(active_mask):
        raise SoilProfileTextureError(f"active_mask must be a (ny, nx) bool array, got {type(active_mask).__name__}")
    if active_mask.dtype != np.bool_:
        raise SoilProfileTextureError(f"active_mask must be bool, got {active_mask.dtype}")
    if tuple(active_mask.shape) != (ny, nx):
        raise SoilProfileTextureError(f"active_mask must have shape {(ny, nx)}, got {tuple(active_mask.shape)}")


def _host_mask(active_mask: Any, ny: int, nx: int) -> np.ndarray:
    """The host (N,) bool copy of the mask (allocates N bytes; counted in `mask_index_bytes`)."""
    from maple.core.backend import to_host

    return np.ascontiguousarray(to_host(active_mask)).reshape(ny * nx).copy()


def _check_backend_namespace(backend: str, namespace: str, representation: str) -> None:
    if representation == "windowed":
        if backend == "cuda":
            raise SoilProfileTextureError(
                "backend 'cuda' is not a windowed compute path: windowed columns are gathered on the host by MAPLE's "
                "own I/O, reduced there ('numpy' or 'numba') and the reduced outputs are moved to the window's "
                "namespace once; select 'numpy' or 'numba'")
        return
    if backend == "cuda" and namespace != "cupy":
        raise SoilProfileTextureError(
            "backend 'cuda' requires the dense native state to be resident on CuPy; the host arrays are not moved "
            "implicitly (select 'numpy'/'numba', or move the state at a documented boundary)")
    if backend != "cuda" and namespace != "numpy":
        raise SoilProfileTextureError(
            f"backend {backend!r} computes on the host but the dense native state is resident on {namespace}; "
            "select backend 'cuda' or bring the state to the host at a documented boundary")


def _wrap_native(operation: str, exc: Exception) -> SoilProfileTextureError:
    return SoilProfileTextureError(f"{operation}: native validation refused the state: {exc}")


def _validate_dense_batch(active_rows: Any, voxel_rows: Any, geometry: Any, mass_resolution_kg: float,
                          operation: str) -> None:
    """MAPLE's own partition validator on one batch (`(1, B, ...)` views; geometry grid replaced by `(1, B)`)."""
    from maple.core.types.active_layer import ActiveLayerState
    from maple.core.types.voxel import VoxelColumnState
    from maple.surface.active_layer.validation import check_active_layer_voxel_partition

    B = int(voxel_rows.shape[0])
    batch_geometry = dataclasses.replace(geometry, ny=1, nx=B)
    try:
        check_active_layer_voxel_partition(
            ActiveLayerState(mass_kg=active_rows.reshape(1, B, active_rows.shape[-1])),
            VoxelColumnState(mass_kg=voxel_rows.reshape(1, B, voxel_rows.shape[1], voxel_rows.shape[2])),
            batch_geometry, mass_resolution_kg)
    except ValueError as exc:
        raise _wrap_native(operation, exc) from exc


def _file_sha256(path: Path) -> str:
    """Streamed hash: at most `IDENTITY_HASH_CHUNK_BYTES` of file content alive at once (counted in the estimate)."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(IDENTITY_HASH_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


#: Native modules whose contracts the DENSE path actually reads (types, formula, validators, numerics). All are
#: mandatory in every supported MAPLE; none is the optional windowed storage.
DENSE_IDENTITY_MODULES = (
    "maple.core.types.voxel",
    "maple.core.types.active_layer",
    "maple.surface.active_layer.capacity",
    "maple.surface.active_layer.validation",
    "maple.surface.voxels.diagnostics",
    "maple.surface.voxels._numerics",
)
#: Additionally read by the WINDOWED path only (optional storage; absent from a genuine Dense-only MAPLE).
WINDOWED_IDENTITY_MODULES = (
    "maple.surface.voxels.windowed_inventory",
    "maple.surface.voxels.windowed_pending",
    "maple.surface.voxels.deep_store",
)


def identity_modules(representation: str) -> tuple[str, ...]:
    """The native modules hashed into the descriptive identity of `representation` ('dense' | 'windowed')."""
    if representation == "dense":
        return DENSE_IDENTITY_MODULES
    if representation == "windowed":
        return DENSE_IDENTITY_MODULES + WINDOWED_IDENTITY_MODULES
    raise SoilProfileTextureError(f"representation must be 'dense' or 'windowed', got {representation!r}")


def _native_identity(representation: str) -> dict[str, Any]:
    """Which native source answered: the maple package path/version and the streamed hashes of exactly the
    modules the SELECTED representation reads. A genuine Dense bed never imports the optional windowed modules
    (a Dense-only MAPLE has none); a genuine windowed bed hashes them because it used them. This is a
    descriptive record of what was read (no lock, no qualification, no rebinding of existing receipts): a module
    that the selected path already imported and used cannot be missing here, so no import is guarded or
    swallowed."""
    import importlib

    import maple

    def entry(name: str) -> dict[str, str]:
        module = importlib.import_module(name)  # already imported by the selected path; a failure is real
        path = Path(module.__file__).resolve()
        return {"path": str(path), "sha256": _file_sha256(path)}

    modules = identity_modules(representation)
    return {
        "maple_package": str(Path(maple.__file__).resolve().parent),
        "maple_version": getattr(maple, "__version__", None),
        "representation": representation,
        "modules": {name: entry(name) for name in modules},
        "hashing": f"streamed, {IDENTITY_HASH_CHUNK_BYTES} B per read, one module at a time",
    }


def _report_cells(flags: np.ndarray, bit: int, mask: np.ndarray, nx: int) -> tuple[int, list[tuple[int, int]]]:
    hit = np.flatnonzero((flags & bit) != 0)
    hit = hit[mask[hit]]
    cells = [(int(c // nx), int(c % nx)) for c in hit[:_MAX_REPORTED_CELLS]]
    return int(hit.shape[0]), cells


# --- public entry points ------------------------------------------------------------------------------------------------
def soil_profile_texture_from_bed(state: Any, context: Any, layers: Any, **options: Any) -> SoilProfileTexture:
    """`build_soil_profile_texture` on a SYRUP `BedState`/`BedContext` pair (only the bed fields and the geometry/
    classes/resolution are read; the ledger, water, commit and availability objects are not required)."""
    return build_soil_profile_texture(state.voxel_column, state.active_layer, context.geometry, context.grain_classes,
                                      context.mass_resolution_kg, layers, **options)


def build_soil_profile_texture(
    voxel_column: Any,
    active_layer: Any,
    geometry: Any,
    grain_classes: Any,
    mass_resolution_kg: float,
    layers: Any,
    *,
    active_mask: Any,
    memory_budget_bytes: int,
    backend: str = "numpy",
    reserve_bytes: int = 0,
    batch_cells: int | None = None,
    coverage: str = COVERAGE_POLICY_FULL,
) -> SoilProfileTexture:
    """Read the actual native bed once and return its Phase 1 layer texture (see the module docstring).

    `layers`: a `SoilLayerGeometry` or a 1-D sequence of layer thicknesses (m), top first; always re-canonicalised.
    `active_mask`: `(ny, nx)` bool (host or the state's namespace). `memory_budget_bytes`: explicit cap on host +
    device bytes this call may allocate, `reserve_bytes` included, checked from metadata alone before any copy.
    `batch_cells`: cells per batch (None: `DEFAULT_BATCH_CELLS`, reduced to fit the budget and recorded; an explicit
    value is never changed). Windowed states always receive MAPLE's full trusted-boundary validation.
    `coverage`: `"full"` (default, historical) refuses an active cell whose actual material does not cover every layer.
    `"partial_over_base"` (explicit opt-in, Phase 2 E1) instead returns the available material: layers (or the lower
    part of one layer) below the bottom of the native voxel column (the flat full-grid base) carry exactly zero mass
    (`layer_total_kg == 0`; zero fractions mean "no material", not an estimate) and `coverage_m` shorter than the layer
    thickness; nothing is invented. Every other refusal is unchanged.
    """
    if backend not in BACKENDS:
        raise SoilProfileTextureError(f"backend must be one of {BACKENDS}, got {backend!r}")
    if type(coverage) is not str or coverage not in COVERAGE_POLICIES:
        raise SoilProfileTextureError(f"coverage must be one of {COVERAGE_POLICIES}, got {coverage!r}")
    partial = coverage == COVERAGE_POLICY_PARTIAL
    layers = canonical_layers(layers)
    budget = _checked_int(memory_budget_bytes, "memory_budget_bytes", minimum=1)
    reserve = _checked_int(reserve_bytes, "reserve_bytes")
    explicit_batch = batch_cells is not None
    if explicit_batch:
        batch_cells = _checked_int(batch_cells, "batch_cells", minimum=1)
    try:
        mres = float(mass_resolution_kg)
    except (TypeError, ValueError) as exc:
        raise SoilProfileTextureError(f"mass_resolution_kg must be a real number: {exc}") from exc
    if not np.isfinite(mres) or mres < 0.0:
        raise SoilProfileTextureError(f"mass_resolution_kg must be finite and >= 0, got {mass_resolution_kg!r}")

    ny, nx, dx, dy, rho, dz, cap_target = _geometry_numbers(geometry)
    representation = _representation(voxel_column, active_layer)  # BedIntegrationError propagates
    _check_mask_metadata(active_mask, ny, nx)  # no copy yet
    n_cells = ny * nx
    denominator = rho * (dx * dy)  # MAPLE: bulk_density_kg_m3 * cell_area_m2 (cell_area_m2 = dx_m * dy_m)

    from maple.core.backend import is_array, to_host

    active_kg = getattr(active_layer, "mass_kg", None)
    if not is_array(active_kg) or active_kg.ndim != 3 or active_kg.dtype != np.float64:
        raise SoilProfileTextureError("active_layer.mass_kg must be a float64 (ny, nx, n_classes) array")
    if tuple(active_kg.shape[:2]) != (ny, nx):
        raise SoilProfileTextureError(f"active_layer.mass_kg grid {tuple(active_kg.shape[:2])} != geometry {(ny, nx)}")
    n_classes = int(active_kg.shape[2])
    class_ids, class_diam, class_agg, class_rho = _class_metadata(grain_classes, n_classes)

    admission = {"budget": budget, "reserve": reserve, "batch_cells": batch_cells, "explicit_batch": explicit_batch}
    if representation == "dense":
        result = _build_dense(voxel_column, active_kg, active_mask, geometry, mres, layers, backend, admission,
                              ny, nx, n_classes, denominator)
    else:
        result = _build_windowed(voxel_column, active_kg, active_mask, geometry, mres, layers, backend, admission,
                                 ny, nx, n_classes, denominator, class_ids)
    outputs, flags, mask, estimate, namespace, n_terms, factor, source_extra = result
    n_active = int(mask.sum())

    # Refusals: evaluated on the host flags of ACTIVE cells, before any output is published.
    n_short, short_cells = _report_cells(flags, FLAG_INSUFFICIENT_DEPTH, mask, nx)
    if partial:  # the opt-in interprets the shortfall as physical base; it must still agree with the actual material
        _admit_partial_stage(estimate)
        n_inconsistent = _partial_coverage_inconsistent(outputs, mask, layers, factor, estimate.batch_cells)
        if n_inconsistent:
            raise SoilProfileGeometryError(
                f"{n_inconsistent} active cell(s) have non-finite/negative depth or coverage, or layer coverage that "
                "disagrees with the actual represented material depth beyond the declared FP64 bound (partial_over_base "
                "requires coverage_l == clip(depth - top_l, 0, thickness_l)); nothing was published.")
        n_short_partial = n_short
        n_short = 0
    n_cap, cap_cells = _report_cells(flags, FLAG_TOP_LAYER_THINNER_THAN_CAP, mask, nx)
    if n_cap:
        raise SoilProfileGeometryError(
            f"{n_cap} active cell(s) have an actual active-layer cap thicker than the top soil layer "
            f"({float(layers.thickness_m[0])} m) by more than the declared FP64 bound; the cap is counted once inside "
            f"the top layer and is not split. First cells (iy, ix): {cap_cells}. Choose a thicker top layer; nothing "
            "was published.")
    if n_short:
        depth_host = to_host(outputs["represented_depth_m"])
        active_depths = depth_host[mask] if n_active else np.zeros(0)
        min_depth = float(active_depths.min()) if active_depths.size else 0.0
        raise SoilProfileGeometryError(
            f"{n_short} active cell(s) do not represent enough actual material to cover the requested layers down to "
            f"{layers.bottom_depth_m} m (minimum represented depth over active cells {min_depth} m; declared FP64 "
            f"bound factor {factor} x max(depth, bottom edge), {n_terms} operations). First cells (iy, ix): "
            f"{short_cells}. No material is invented and the profile is not truncated; reduce the layer depths, "
            "deactivate those cells, or supply a deeper native column.")

    source = {
        "schema": SCHEMA,
        "kind": "descriptive_identity_of_inputs_read_at_build",
        "not_a_runtime_source_lock": True,
        "not_a_restart_codec": True,
        "representation": representation,
        "namespace": namespace,
        "backend": backend,
        "native": _native_identity(representation),
        "geometry": {"ny": ny, "nx": nx, "dx_m": dx, "dy_m": dy, "voxel_dz_m": dz, "bulk_density_kg_m3": rho,
                     "active_layer_thickness_target_m": cap_target, "cell_area_m2": dx * dy,
                     "n_voxels": estimate.n_voxels},
        "classes": [{"class_id": i, "diameter_m": d, "is_aggregate": a, "particle_density_kg_m3": r}
                    for i, d, a, r in zip(class_ids, class_diam, class_agg, class_rho)],
        "layers": {"thickness_m": [float(t) for t in layers.thickness_m],
                   "depth_edges_m": [float(e) for e in layers.depth_edges_m],
                   "datum": "actual material surface of each cell, depth positive downward",
                   "canonical": "fresh copy, sequential edges, validated at the public boundary"},
        "coverage_bound": {"function": COVERAGE_BOUND_FUNCTION, "n_terms": n_terms, "factor": factor,
                           "scale_rule": "max(represented_depth_cell, deepest_layer_edge)",
                           "applies_to": ["layer coverage deficit", "active cap above top layer"]},
        "class_total_arithmetic": "declared sequential class fold ((c0 + c1) + c2) + ...; not claimed bitwise equal "
                                  "to MAPLE's library sum(axis=-1) for n_classes >= 8",
        "cells": {"n_cells": n_cells, "n_active": n_active, "layout": "(L, N, K), N = iy * nx + ix"},
        "units": {"layer_mass_kg": "kg", "layer_fraction": "mass fraction (dimensionless)",
                  "coverage_m": "m", "represented_depth_m": "m", "active_cap_thickness_m": "m"},
        "memory": dataclasses.asdict(estimate),
        "mass_resolution_kg": mres,
    }
    source.update(source_extra)
    if partial:
        source["coverage_policy"] = {
            "policy": COVERAGE_POLICY_PARTIAL, "version": "maple_syrup.soil_profile_texture.partial_coverage/1",
            "interpretation": "layer parts below the bottom of the native voxel column (flat full-grid base, elevation 0) "
                              "hold exactly zero mass; texture there is undefined (zero fractions), not estimated",
            "validation_stage": {"chunk_cells": estimate.batch_cells,
                                 "model_bytes_namespace_host": list(partial_stage_bytes(estimate.batch_cells)),
                                 "host_transfer": "one bool flag copy of chunk_cells per chunk (counted to_host)"},
            "n_active_cells_short": int(n_short_partial),
            "represented_depth_m_meaning": "existing sequential segment-height diagnostic; NOT the canonical hydraulic "
                                           "bedrock-depth witness",
            "not_claimed": ["hydraulic depth classification", "property refresh", "water conservation",
                            "pond/restart integration"]}
    return SoilProfileTexture(
        layer_mass_kg=outputs["layer_mass_kg"], layer_fraction=outputs["layer_fraction"],
        layer_total_kg=outputs["layer_total_kg"], coverage_m=outputs["coverage_m"],
        represented_depth_m=outputs["represented_depth_m"], active_cap_thickness_m=outputs["active_cap_thickness_m"],
        active_mask=mask, layers=layers, ny=ny, nx=nx, class_ids=class_ids, class_diameter_m=class_diam,
        class_is_aggregate=class_agg, class_particle_density_kg_m3=class_rho, cell_area_m2=dx * dy,
        bulk_density_kg_m3=rho, coverage_bound_terms=n_terms, coverage_bound_factor=factor, namespace=namespace,
        backend=backend, representation=representation, memory=estimate, source=source, coverage_policy=coverage,
    )


def partial_stage_bytes(batch: int) -> tuple[int, int]:
    """`(namespace_bytes, host_bytes)` of the opt-in coverage validation stage for chunks of `batch` cells: the live
    float64/bool expression temporaries in the outputs' namespace, and the host copy of the `batch` bool flags. No
    full-domain, no `L x N`, no mask upload, no library reduction workspace."""
    B = _checked_int(batch, "batch", minimum=1)
    return PARTIAL_STAGE_F64_VECTORS * _F8 * B + PARTIAL_STAGE_BOOL_VECTORS * B, B


def _admit_partial_stage(estimate: TextureMemoryEstimate) -> None:
    """The stage runs AFTER every batch kernel/validator buffer is released, so it may reuse the working set the
    estimate already admitted for the batch phase: its namespace temporaries must fit inside the counted `validation_bytes`
    (the native partition validator of one batch, >= 184 B per cell) and its host flags inside the counted host scratch.
    This is checked, not assumed; the default `full` estimate is unchanged. Refused (nothing published) otherwise."""
    ns_bytes, host_bytes = partial_stage_bytes(estimate.batch_cells)
    host_reusable = estimate.validation_bytes if (estimate.output_namespace == "numpy" or estimate.representation == "windowed") \
        else estimate.fixed_overhead_bytes  # dense CUDA keeps the validator on the device; its host side holds the flags in the declared allowance
    if ns_bytes > estimate.validation_bytes or host_bytes > host_reusable:
        raise SoilProfileMemoryError(
            f"partial-coverage validation stage needs {ns_bytes} B (namespace) + {host_bytes} B (host flags) at "
            f"batch_cells={estimate.batch_cells}, above the batch working set already admitted "
            f"({estimate.validation_bytes} B validation, {host_reusable} B host); nothing was published.")


def _partial_coverage_inconsistent(outputs: dict, mask: np.ndarray, layers: Any, factor: float, batch: int) -> int:
    """Count ACTIVE cells with invalid or inconsistent coverage: non-finite or negative `represented_depth_m`, a
    non-finite or negative `coverage_m`, or `coverage_m[l]` differing from `clip(depth - top_l, 0, thickness_l)` by more than
    the declared operation-derived bound `factor * max(depth, bottom edge)` (the strict check's bound). Bad flags are formed
    per chunk of `batch` cells in the outputs' namespace, only those `batch` bool flags are copied to the host with MAPLE's
    counted `to_host`, and the HOST mask slice selects/counts the active cells: no second full-domain mask upload, no
    library reduction workspace, no `L x N` tensor. Fails closed: NaN/inf are flagged before any comparison."""
    from maple.core.backend import array_namespace, to_host

    xp = array_namespace(outputs["coverage_m"])
    edges = layers.depth_edges_m
    L, N = layers.n_layers, int(mask.shape[0])
    step = max(1, int(batch))
    bad = 0
    for b0 in range(0, N, step):
        sl = slice(b0, min(N, b0 + step))
        depth = outputs["represented_depth_m"][sl]
        wrong = ~xp.isfinite(depth) | (depth < 0.0)
        tol = factor * xp.maximum(depth, float(edges[L]))
        for l in range(L):
            cov = outputs["coverage_m"][l, sl]
            expected = xp.minimum(xp.maximum(depth - float(edges[l]), 0.0), float(edges[l + 1] - edges[l]))
            wrong = wrong | ~xp.isfinite(cov) | (cov < 0.0) | (xp.abs(cov - expected) > tol)
        host_flags = np.asarray(to_host(wrong))  # the ONE host flag buffer (fresh: a device download or the CPU `wrong` itself)
        np.logical_and(host_flags, mask[sl], out=host_flags)  # mask combined IN PLACE: no second host chunk, `mask` untouched
        bad += int(np.count_nonzero(host_flags))
    return bad


# --- batch runner selection ---------------------------------------------------------------------------------------------
def _runner(backend: str):
    """`runner(xp, active_rows, voxel_rows, mask_rows, edges, denominator, bound_factor) -> 7-tuple`."""
    if backend == "numpy":
        return reference_texture_batch
    if backend == "numba":
        from maple_syrup.soil_profile_texture_numba import prepare_runner

        return prepare_runner()
    from maple_syrup.soil_profile_texture_cuda import prepare_runner

    return prepare_runner()


def _admit(make_estimate, budget: int, batch: int, explicit: bool, reduced: bool = False) -> TextureMemoryEstimate:
    """Reduce a DEFAULTED batch until the estimate fits; refuse when even one cell does not fit or when an explicit
    batch does not fit. `reduced` carries an earlier (lower-bound) reduction into the record. Nothing is allocated
    before this returns."""
    while True:
        estimate = make_estimate(batch, reduced)
        if estimate.required_bytes <= budget:
            return estimate
        if explicit or batch <= 1:
            stage = "metadata-only lower bound" if estimate.metadata_only else "final estimate"
            raise SoilProfileMemoryError(
                f"soil profile texture ({estimate.representation}, {estimate.backend}; {stage}) needs an estimated "
                f"{estimate.required_bytes} B (outputs {estimate.output_bytes} B, staging {estimate.output_staging_bytes} B, "
                f"mask/index {estimate.mask_index_bytes} B, batch inputs {estimate.batch_input_bytes} B, validation "
                f"{estimate.validation_bytes} B, scratch {estimate.scratch_bytes} B, native cache "
                f"{estimate.native_cache_bytes} B, fixed overhead {estimate.fixed_overhead_bytes} B, class/layer metadata "
                f"{estimate.metadata_bytes} B, reserve "
                f"{estimate.reserve_bytes} B) at batch_cells={batch}, above the "
                f"caller budget {budget} B; nothing was allocated and the budget is not raised. The caller's own native "
                f"arrays ({estimate.input_bytes_excluded} B) are not counted.")
        batch = max(1, batch // 2)
        reduced = True


def _allocate_outputs(xp: Any, L: int, N: int, K: int) -> dict[str, Any]:
    return {
        "layer_mass_kg": xp.zeros((L, N, K), dtype=np.float64),
        "layer_fraction": xp.zeros((L, N, K), dtype=np.float64),
        "layer_total_kg": xp.zeros((L, N), dtype=np.float64),
        "coverage_m": xp.zeros((L, N), dtype=np.float64),
        "represented_depth_m": xp.zeros((N,), dtype=np.float64),
        "active_cap_thickness_m": xp.zeros((N,), dtype=np.float64),
    }


def _place(outputs: dict[str, Any], result: tuple[Any, ...], where: Any) -> None:
    mass, fraction, total, coverage, depth, cap, _ = result
    outputs["layer_mass_kg"][:, where, :] = mass
    outputs["layer_fraction"][:, where, :] = fraction
    outputs["layer_total_kg"][:, where] = total
    outputs["coverage_m"][:, where] = coverage
    outputs["represented_depth_m"][where] = depth
    outputs["active_cap_thickness_m"][where] = cap


def _c_contiguous(array: Any) -> bool:
    flags = getattr(array, "flags", None)
    return bool(flags is not None and flags.c_contiguous)


def _default_batch(explicit: bool, batch_cells: int | None, n: int) -> int:
    return batch_cells if explicit else max(1, min(n, DEFAULT_BATCH_CELLS))


def _build_dense(voxel_column, active_kg, active_mask, geometry, mres, layers, backend, admission, ny, nx,
                 n_classes, denominator):
    from maple.core.backend import (
        MixedArrayNamespaceError,
        array_namespace,
        is_array,
        namespace_name,
        to_host,
    )

    voxels = getattr(voxel_column, "mass_kg", None)
    if not is_array(voxels) or voxels.ndim != 4 or voxels.dtype != np.float64:
        raise SoilProfileTextureError("voxel_column.mass_kg must be a float64 (ny, nx, n_voxels, n_classes) array")
    if tuple(voxels.shape[:2]) != (ny, nx):
        raise SoilProfileTextureError(f"voxel_column.mass_kg grid {tuple(voxels.shape[:2])} != geometry {(ny, nx)}")
    if int(voxels.shape[3]) != n_classes:
        raise SoilProfileTextureError(
            f"voxel_column has {int(voxels.shape[3])} grain classes but active_layer has {n_classes}")
    if not (_c_contiguous(voxels) and _c_contiguous(active_kg)):
        raise SoilProfileTextureError("dense native arrays must be C-contiguous (batch views are taken without copies)")
    try:
        xp = array_namespace(voxels, active_kg)
    except MixedArrayNamespaceError as exc:
        raise SoilProfileTextureError(f"voxel column and active layer live in different namespaces: {exc}") from exc
    namespace = namespace_name(xp)
    _check_backend_namespace(backend, namespace, "dense")
    N, V, K, L = ny * nx, int(voxels.shape[2]), n_classes, layers.n_layers
    excluded = int(voxels.nbytes) + int(active_kg.nbytes)
    budget, reserve = admission["budget"], admission["reserve"]
    explicit, batch0 = admission["explicit_batch"], _default_batch(admission["explicit_batch"], admission["batch_cells"], N)

    def make(n_active, metadata_only):
        def inner(batch, reduced):
            return estimate_texture_memory(representation="dense", backend=backend, output_namespace=namespace,
                                           n_cells=N, n_active=n_active, n_layers=L, n_classes=K, n_voxels=V,
                                           batch_cells=batch, reserve_bytes=reserve, input_bytes_excluded=excluded,
                                           batch_reduced=reduced, metadata_only=metadata_only)
        return inner

    # 1. Metadata-only lower bound (every cell assumed active) BEFORE the mask copy or any native check.
    lower = _admit(make(N, True), budget, batch0, explicit)
    runner = _runner(backend)  # backend availability is a setup failure, raised before any allocation
    mask = _host_mask(active_mask, ny, nx)  # N bytes, counted
    # 2. Final estimate (dense batches cover every cell, so the batch is unchanged; the record is the final one).
    estimate = _admit(make(int(mask.sum()), False), budget, lower.batch_cells, explicit, lower.batch_reduced)
    n_terms = coverage_bound_terms(V, K, L)
    factor = coverage_bound_factor(V, K, L)
    B = estimate.batch_cells
    outputs = _allocate_outputs(xp, L, N, K)
    flags = np.zeros((N,), dtype=np.uint8)
    voxels_flat = voxels.reshape(N, V, K)
    active_flat = active_kg.reshape(N, K)
    mask_dev = xp.asarray(mask)  # the same host array for NumPy; one counted copy on the device
    edges = layers.depth_edges_m
    for b0 in range(0, N, B):
        b1 = min(N, b0 + B)
        v_rows, a_rows = voxels_flat[b0:b1], active_flat[b0:b1]
        _validate_dense_batch(a_rows, v_rows, geometry, mres, "build_soil_profile_texture(dense)")
        result = runner(xp, a_rows, v_rows, mask_dev[b0:b1], edges, denominator, factor)
        _place(outputs, result, slice(b0, b1))
        flags[b0:b1] = to_host(result[6])  # one small counted read per batch on the device
        del result, v_rows, a_rows  # release this batch's buffers before the next validation/kernel allocation
    extra = {"dense": {"n_voxels": V, "n_batches": (N + B - 1) // B, "validation": "native partition validator per batch"}}
    return outputs, flags, mask, estimate, namespace, n_terms, factor, extra


def _windowed_metadata(voxel_column, n_classes):
    """Header/version metadata of a windowed column WITHOUT any array work (used before admission). The native
    cheap validator re-checks every binding afterwards."""
    from maple.surface.voxels.deep_store import DeepStoreVersion

    version = getattr(voxel_column, "deep_store", None)
    if not isinstance(version, DeepStoreVersion):
        raise SoilProfileTextureError("windowed column is not bound to a DeepStoreVersion handle")
    header = version.header
    window = getattr(voxel_column, "window_kg", None)
    if window is None or getattr(window, "ndim", 0) != 4:
        raise SoilProfileTextureError("windowed column lacks a 4-D window_kg array")
    V, W, C = int(header.n_voxels), int(window.shape[2]), int(header.chunk_columns)
    if int(voxel_column.n_voxels) != V or int(header.n_classes) != n_classes:
        raise SoilProfileTextureError("windowed column metadata disagrees with the bound deep-store header or the active layer")
    return version, header, V, W, C, int(version.mapped_bytes_bound)


def _build_windowed(voxel_column, active_kg, active_mask, geometry, mres, layers, backend, admission, ny, nx,
                    n_classes, denominator, class_ids):
    from maple.core.backend import (
        MixedArrayNamespaceError,
        array_namespace,
        namespace_name,
        to_device,
        to_host,
    )
    from maple.surface.active_layer.validation import check_active_layer_state
    from maple.surface.voxels.windowed_inventory import (
        check_windowed_column_cheap,
        gather_full_columns,
        validate_windowed_column_full,
    )
    from maple.surface.voxels.windowed_pending import require_published_column

    _check_backend_namespace(backend, "", "windowed")
    operation = "build_soil_profile_texture(windowed)"
    version, header, V, W, C, cache_bytes = _windowed_metadata(voxel_column, n_classes)
    try:
        xp = array_namespace(voxel_column.window_kg, voxel_column.base, active_kg)
    except MixedArrayNamespaceError as exc:
        raise SoilProfileTextureError(f"window, base and active layer live in different namespaces: {exc}") from exc
    namespace = namespace_name(xp)
    N, K, L = ny * nx, n_classes, layers.n_layers
    excluded = (int(voxel_column.window_kg.nbytes) + int(voxel_column.base.nbytes)
                + int(voxel_column.deep_by_class_kg.nbytes) + int(voxel_column.deep_total_kg.nbytes)
                + int(voxel_column.deep_room_kg.nbytes) + int(active_kg.nbytes))
    budget, reserve = admission["budget"], admission["reserve"]
    explicit = admission["explicit_batch"]

    def make(n_active, metadata_only):
        def inner(batch, reduced):
            return estimate_texture_memory(representation="windowed", backend=backend, output_namespace=namespace,
                                           n_cells=N, n_active=n_active, n_layers=L, n_classes=K, n_voxels=V,
                                           batch_cells=batch, window_voxels=W, chunk_cells=C,
                                           native_cache_bytes=cache_bytes, reserve_bytes=reserve,
                                           input_bytes_excluded=excluded, batch_reduced=reduced,
                                           metadata_only=metadata_only)
        return inner

    # 1. Metadata-only lower bound (every cell assumed active) BEFORE the mask copy, the index vector or any native
    #    check (the native cheap/full validators allocate their own temporaries, which are in this estimate).
    lower = _admit(make(N, True), budget, _default_batch(explicit, admission["batch_cells"], N), explicit)
    runner = _runner(backend)
    # 2. Native source checks (never skipped, never downgraded): published, cheap bindings, active layer, FULL.
    try:
        require_published_column(voxel_column, operation=operation)  # refuses an unpublished journal; publishes nothing
        check_windowed_column_cheap(voxel_column, geometry, class_ids=class_ids)
        check_active_layer_state(_as_active(active_kg), geometry, mres)
        validate_windowed_column_full(voxel_column, geometry, mres, class_ids=class_ids)
    except (TypeError, ValueError) as exc:
        raise _wrap_native(operation, exc) from exc
    mask = _host_mask(active_mask, ny, nx)  # N bytes, counted
    active_idx = np.flatnonzero(mask).astype(np.int64)  # <= 8 N bytes, counted
    n_active = int(active_idx.shape[0])
    # 3. Final estimate with the actual active count (batch never above the lower bound's admitted batch).
    batch_final = lower.batch_cells if explicit else max(1, min(lower.batch_cells, n_active))
    estimate = _admit(make(n_active, False), budget, batch_final, explicit, lower.batch_reduced)
    n_terms = coverage_bound_terms(V, K, L)
    factor = coverage_bound_factor(V, K, L)
    B = estimate.batch_cells
    host_outputs = _allocate_outputs(np, L, N, K)
    flags = np.zeros((N,), dtype=np.uint8)
    edges = layers.depth_edges_m
    active_flat = active_kg.reshape(N, K)
    n_batches = 0
    for b0 in range(0, n_active, B):
        idx = active_idx[b0:min(n_active, b0 + B)]
        full = gather_full_columns(voxel_column, idx, max_columns=B)  # fresh host (B, V, K); refuses pending itself
        a_rows = np.ascontiguousarray(to_host(active_flat[xp.asarray(idx)]))  # (B, K) host copy
        _validate_dense_batch(a_rows, full, geometry, mres, operation)
        result = runner(np, a_rows, full, np.ones(idx.shape[0], dtype=np.bool_), edges, denominator, factor)
        _place(host_outputs, result, idx)
        flags[idx] = result[6]
        n_batches += 1
        del full, a_rows, result, idx  # one batch alive at a time: released before the next gather allocates
    if namespace == "cupy":
        outputs = {name: to_device(array, xp) for name, array in host_outputs.items()}  # one counted H2D each
    else:
        outputs = host_outputs
    extra = {"windowed": {
        "n_voxels": V, "window_voxels": W, "n_batches": n_batches, "batch_cells": B,
        "validation": "native: require_published_column, check_windowed_column_cheap, check_active_layer_state, "
                      "validate_windowed_column_full (always), dense partition validator per gathered batch",
        "deep_store_version_id": int(version.version_id),
        "deep_store_manifest_digest": str(version.manifest_digest), "deep_store_header_digest": header.digest(),
        "deep_store_dir": str(version.store_dir), "chunk_columns": C, "mapping_cache_bound_bytes": cache_bytes,
        "generation": int(voxel_column.generation),
    }}
    return outputs, flags, mask, estimate, namespace, n_terms, factor, extra


def _as_active(active_kg: Any):
    from maple.core.types.active_layer import ActiveLayerState

    return ActiveLayerState(mass_kg=active_kg)
