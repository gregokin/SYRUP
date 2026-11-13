"""Phase 2 S2-A.1: explicit per-cell / per-layer soil PARAMETERS, preparation only (`parameter_component_only`).

This module validates caller-supplied soil layer inputs for `N` cells, derives the minimal set of retained maps and owns detached, backend-resident copies of them. It contains NO hydrology: no
step, kernel, runner, state evolution, infiltration, evaporation, transpiration, routing, storm or checkpoint, and it chooses no moving frame, texture-to-property method, field-capacity pressure,
conductivity averaging rule, wetting-history rule or water fate. The caller states the geometry (`depth_edges_m`, per cell or common), the retention thresholds (`theta_residual`,
`theta_field_capacity`, `theta_saturation`), the existing diffusion inputs and the shared scalars; `theta_residual` is a withdrawal floor, NOT a minimum storage, so dry (zero) water states stay
allowed and nothing here inspects or invents water. A prepared `SpatialSoilColumn` is a separate class from `subsurface.SoilLayers`/`SoilColumn` (no subclass): every existing entry point
refuses it at its first line (typed `SubsurfaceError`), and `prepare_spatial_column` refuses the foreign `SoilLayers`.

Correction01 contracts: the spec is NOT trusted at preparation time -- `prepare_spatial_column` rebuilds it from the LIVE arrays through the same factory and refuses (typed, before any admission/allocation) any stale or edited
metadata (shape, dtype, layout, bytes, options, extra/missing/forged records, hand-built or replaced specs); a one-layer scalar interface input must still be finite and >= 0; public layout/estimate metadata is deeply read-only and the
owner can only be created inside `prepare_spatial_column`; the NumPy path never probes CuPy (CuPy is discovered only for a cupy-typed array or an explicit cuda request, and an unusable CuPy is a typed setup failure).

Inputs (explicit arrays, exactly the backend's namespace, float64, C-contiguous; real host scalars allowed where noted; NO implicit conversion or transfer)
-----------------------------------------------------------------------------------------------------------------------------------------------
`L >= 1` layers, `N >= 1` cells (flat C order `cell = iy * nx + ix`, as everywhere in SYRUP). Per input array the LAYOUT is derived from its shape and recorded:
  depth_edges_m           `(L+1,)` SHARED | `(L+1, N)` CELL   (each column strictly increasing from exactly 0: depth below the local surface; thickness may differ by cell)
  theta_residual,
  theta_field_capacity,
  theta_saturation        real scalar (SCALAR) | `(L,)` SHARED | `(L, N)` CELL
  diffusivity_m2_s        (mode "constant") | interface_conductivity_m_s (mode "tietjen_donor_depth"): real scalar | `(L-1,)` | `(L-1, N)`; for `L = 1` a scalar or an EMPTY `(0,)` / `(0, N)` array
  diffusion_coefficient d (donor mode), theta_evaporation_stress, evaporation_factor, gravity_drainage, diffusion_mode, bottom_drainage: shared host scalars/strings, exactly the meanings of
  `subsurface.soil_layers` (no new mode, no averaging, no provider). Every cell is validated (inactive cells must carry valid placeholders; the active mask is stored, never used to skip validation).

Validation = the per-element rules of `soil_layers`, evaluated per cell with the SAME expression order: finite edges, `edges[0] == 0`, strictly increasing with finite positive widths; finite thetas with `0 <= res <= fc <= sat <= 1`
and `res < sat`; interface input finite and `>= 0`; `stress > theta_residual[0]` in every cell; then every derived quantity must be finite and representable (inverse thickness, centres, centre separation `> 0`,
pair inverse thickness, conductance, exchange rate = conductance x pair inverse, storage depths `theta * dz`, `saturation depth > 0`, capacity tolerance `coef * sat_depth` finite and resolvable `sat + tol > sat`, `coef =
volume_roundoff_bound_m3(LAYER_OPERATION_TERMS, 1.0)`). Input checks come first (flags 1 edges, 2 non-finite theta, 4 theta relations, 8 interface input, 32 stress); the derived check (16) is evaluated only for cells that
passed every input check, so the flag a cell reports does not depend on the implementation. A width that overflows is reported as an edge failure (`soil_layers` refuses the same set of inputs, with a derived-value message).

Retained maps (the ONLY arrays kept; everything else is transient and freed)
------------------------------------------------------------------------------
  theta_residual, theta_field_capacity, theta_saturation  detached copies, input layout kept (a SCALAR becomes a 1-element array); the top-layer residual theta0 is row 0 of theta_residual (no extra map);
  dz                      `(L,)` | `(L, N)`   = `edges[k+1] - edges[k]`  (layout of the edges);
  pair_inverse_thickness  `(L-1,)` | `(L-1, N)`   = `1/dz_k + 1/dz_{k+1}`  (layout of the edges);
  interface_conductance_down  `(L-1,)` | `(L-1, N)`  constant mode: `D / separation` (the "up" conductance ALIASES the same array, as in `SoilLayers`); donor mode: `((d K) dz_k) / separation`, plus
  interface_conductance_up    donor mode only: `((d K) dz_{k+1}) / separation`  (layout = join of the edge and interface-input layouts: CELL if either is CELL).
No edges, centres, D/K arrays, depth-form thresholds (`theta * dz`) or capacity-tolerance maps are retained: a later compiled reader recomputes `theta * dz` and `coef * saturation depth` with the same ordered scalar
operations (qualified later). For `L = 1` there are no interface maps (they are absent, not fake zeros). Retained doubles per cell, fully CELL: constant `4L + 2(L-1)`, donor `4L + 3(L-1)`; SHARED/SCALAR arrays cost `8 * size`
independent of `N`. The active mask is copied (N bool). Prepared maps are owned: the NumPy arrays are read-only; CUDA arrays are private to the column; no public view is exposed (see `inspect`, `summary`).

Backends and residency
----------------------
"numpy" and "numba" keep everything on the host (the numba backend needs Numba importable; the PREPARATION is the same NumPy derivation for both, and S2-A.3 additionally compiles the spatial variant of the shared hydrology kernel of `subsurface_numba` at preparation, outside any step). "cuda" keeps the inputs, the retained maps and the work on the
CURRENT device: ONE RawKernel (one thread per cell, loops over the layers, the same expression order as the NumPy reference, compiled with `--fmad=false --prec-div=true --prec-sqrt=true --ftz=false`) validates and derives and
writes the retained maps directly; failures are reported through device atomics into `DIAG_WORDS = 4` `uint64` words (flag OR, flagged-cell count, `min(cell << 8 | flags)` = first flagged cell with its flags, active-cell
count) read back with ONE 32-byte transfer. No CuPy reduction, no mask/index/flag array, no library workspace and no full-size download exists; the host receives only the 32-byte packet and the identity chunks below. The
derived NumPy-vs-CUDA values use the same IEEE operations; this slice does not claim byte equality of the compiled kernel with NumPy (the host-compiler parity test checks the arithmetic of the kernel source against the
NumPy reference; Root qualifies the device). Deterministic identity (below) hashes the INPUTS, so CPU/GPU roundoff in derived maps never changes the identity.

Identity (cold, streamed, chunk-size invariant)
------------------------------------------------
`parameters_sha256` = SHA-256 over the schema tag, the scalar options, then each input array (name, layout, shape) followed by its bytes row by row in C order, streamed in pieces of `hash_chunk_bytes` (NumPy: views, no copy;
CUDA: each piece is copied to ONE preallocated host buffer of that size, counted in the host requirement). The `mask_sha256` hashes the mask the same way. The identity is of the explicit INPUTS (not of the derived or retained maps),
its schema string differs from `SoilLayers.identity()`, and it is chunk-size invariant (tested). The record also carries the module source hash; no existing verifier, capsule or receipt is touched.

Memory (modelled, not measured resident memory; see `estimate_spatial_memory`)
------------------------------------------------------------------------------
Counted BEFORE any allocation, host and device separately, against explicit caller quotas and truthful, injectable available-bytes readings; nothing is raised silently. Retained maps and mask copy; the NumPy chunk work
(`REF_VEC_F64` float64, `REF_VEC_BOOL` bool and `REF_VEC_U8` uint8 `(chunk_columns,)` work vectors, checked against `tracemalloc` warm); the host identity buffer (`hash_chunk_bytes`) on the CUDA side; the 32-byte packet; a
declared small Python-object allowance. The caller's input arrays are EXCLUDED from the requirement and reported with their actual bytes (`input_bytes`). Excluded from the model: NVRTC/module load, the CUDA context, allocator pool
policy, interpreter/JIT state and whole-process memory. A reading is not a reservation.
"""

from __future__ import annotations

import hashlib
import math
import os
import contextlib
import threading
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable

import numpy as np

from maple_syrup import subsurface as ss

__all__ = [
    "SPATIAL_ROW_TEMP_DOUBLES",
    "STATE_COMPONENT_TAG",
    "SpatialOperationMemory",
    "SpatialSoilState",
    "SpatialStep",
    "estimate_spatial_operation_memory",
    "spatial_column_state",
    "spatial_moisture",
    "spatial_step",
    "spatial_step_with_recharge",
    "spatial_storage_m3",
    "spatial_uniform_state",
    "COMPONENT_TAG",
    "DIAG_WORDS",
    "LAYOUT_CELL",
    "LAYOUT_SCALAR",
    "LAYOUT_SHARED",
    "MAX_CUDA_CELLS",
    "SCHEMA",
    "SpatialLayerBackendUnavailableError",
    "SpatialLayerError",
    "SpatialLayerMemoryError",
    "SpatialLayerSpec",
    "SpatialMemoryEstimate",
    "SpatialSoilColumn",
    "estimate_spatial_memory",
    "prepare_spatial_column",
    "spatial_layer_spec",
]

COMPONENT_TAG = "parameter_component_only"
SCHEMA = "maple_syrup.spatial_soil_layers/1"
LAYOUT_SCALAR, LAYOUT_SHARED, LAYOUT_CELL = "scalar", "shared", "cell"
FLAG_EDGES, FLAG_THETA, FLAG_REL, FLAG_IFACE, FLAG_DERIVED, FLAG_STRESS = 1, 2, 4, 8, 16, 32
_FLAG_TEXT = {FLAG_EDGES: "depth edges are not finite, do not start at exactly 0, or are not strictly increasing with finite positive widths",
              FLAG_THETA: "a theta input is not finite", FLAG_REL: "theta values violate 0 <= residual <= field capacity <= saturation <= 1 with residual < saturation",
              FLAG_IFACE: "the interface input (diffusivity / conductivity) is not finite or is negative",
              FLAG_DERIVED: "a derived quantity is not finite/representable or the capacity is not resolvable (layers too thin or inputs too extreme)",
              FLAG_STRESS: "theta_evaporation_stress is not above the layer-0 residual"}
DIAG_WORDS = 4
PACKET_BYTES = DIAG_WORDS * 8
AUX_BYTES = 4 * 8  # the one 4-element device auxiliary array of the CUDA path
MAX_CUDA_CELLS = min(2**56 - 1, (2**31 - 1) * 128)  # (cell << 8 | flags) fits 64 bits and the grid fits the launch limit (128 threads per block)
CUDA_BLOCK = 128
REF_VEC_F64, REF_VEC_BOOL, REF_VEC_U8 = 24, 16, 4  # live (chunk,) work vectors of the NumPy chunk derivation (counted from the code; checked by tracemalloc in the tests)
PY_ALLOWANCE_BYTES = 65536  # small Python objects, exception text, dict/tuple records
DEFAULT_CHUNK_COLUMNS = 65536
DEFAULT_HASH_CHUNK_BYTES = 1 << 20
COMPILE_OPTIONS = ("--fmad=false", "--prec-div=true", "--prec-sqrt=true", "--ftz=false")


class SpatialLayerError(ss.SubsurfaceError):
    """Invalid explicit input (metadata or value) or an unrepresentable derived parameter. Nothing was published."""

    def __init__(self, message: str, *, bit: int | None = None, n_flagged: int = 0, first_cell: int | None = None):
        super().__init__(message)
        self.bit, self.n_flagged, self.first_cell = bit, int(n_flagged), first_cell


class SpatialLayerMemoryError(ss.SubsurfaceMemoryError):
    """The modelled requirement plus reserve exceeds the declared quota or the reported availability of one side. Nothing was allocated."""


class SpatialLayerBackendUnavailableError(ss.SubsurfaceBackendUnavailableError):
    """The requested backend (Numba, CuPy/CUDA) or a host/device memory reading is unavailable. There is no fallback."""


# --- small helpers ---------------------------------------------------------------------------------------------------------------------
def _count(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or int(value) < minimum:
        raise SpatialLayerError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return int(value)


def _real(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise SpatialLayerError(f"{name} must be a real host scalar, got {type(value).__name__}")
    out = float(value)
    if not math.isfinite(out):
        raise SpatialLayerError(f"{name} must be finite, got {value!r}")
    return out


def _is_real_scalar(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, (bool, np.bool_))


def default_host_available_bytes() -> int:
    """Available physical host memory (`sysconf`). A reading, not a reservation; a platform without it is a setup error (inject a reader)."""
    try:
        pages, page = os.sysconf("SC_AVPHYS_PAGES"), os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, ValueError, OSError) as exc:
        raise SpatialLayerBackendUnavailableError(f"the host cannot report available memory ({type(exc).__name__}: {exc}); pass a host reader") from exc
    if pages < 0 or page <= 0:
        raise SpatialLayerBackendUnavailableError("the host reported an invalid available-memory figure; pass a host reader")
    return int(pages) * int(page)


def _cuda_module():
    from maple_syrup import soil_water_remap_cuda as rc  # read-only reuse of the S1 device probes (require_cuda, device id, device available-bytes reading)

    return rc


def _require_cuda():
    try:
        return _cuda_module().require_cuda()
    except Exception as exc:  # noqa: BLE001
        if isinstance(exc, SpatialLayerError):
            raise
        raise SpatialLayerBackendUnavailableError(str(exc)) from exc


def _cupy_ndarray_type(what: str):
    """Discover CuPy ONLY because a non-NumPy array was supplied (a NumPy spec/preparation never touches CuPy or the GPU). A CuPy that is not usable is a typed setup failure, not a fallback."""
    try:
        from maple.core.backend import cupy_import_error, cupy_module

        cp = cupy_module()
        reason = None if cp is not None else (cupy_import_error() or "not installed")
    except Exception as exc:  # noqa: BLE001 - reported as a typed setup failure below (no fallback)
        raise SpatialLayerBackendUnavailableError(f"{what} is not a numpy.ndarray and CuPy could not be probed ({type(exc).__name__}: {exc}); there is no fallback") from exc
    if cp is None:
        raise SpatialLayerBackendUnavailableError(f"{what} is not a numpy.ndarray and CuPy is not usable ({reason}); there is no fallback")
    return cp.ndarray


def _freeze(value: Any) -> Any:
    """Deep, read-only view of nested plain mappings/lists (public metadata must stay canonical)."""
    if isinstance(value, dict):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    return value


# --- the explicit input record --------------------------------------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class _Input:
    name: str
    kind: str  # scalar | shared | cell
    array: Any  # the CALLER's array (no copy) or None for a host scalar
    scalar: float | None
    nbytes: int  # actual bytes of the caller's array (0 for a host scalar)
    rows: int  # leading extent (0 for an empty interface input)


@dataclass(frozen=True, eq=False)
class SpatialLayerSpec:
    """Metadata-checked, UNPREPARED description of the explicit inputs. Holds references to the caller's arrays (no copy, no value read); `prepare_spatial_column` validates, derives and owns copies.
    Not a `SoilLayers`; hydrology is unsupported."""

    n_layers: int
    n_cells: int
    namespace: str  # "numpy" | "cupy"
    inputs: Any  # a read-only mapping name -> _Input (revalidated from the LIVE arrays at every preparation)
    diffusion_mode: str
    bottom_drainage: str
    gravity_drainage: bool
    diffusion_coefficient: float | None
    theta_evaporation_stress: float
    evaporation_factor: float
    component_tag: str = COMPONENT_TAG

    def layouts(self) -> dict[str, str]:
        return {k: v.kind for k, v in self.inputs.items()}

    def input_bytes(self) -> dict[str, int]:
        return {k: v.nbytes for k, v in self.inputs.items()}


def _array_kind(name: str, value: Any, shared_shape: tuple, cell_shape: tuple, *, scalar_ok: bool, ns_type: Any) -> _Input:
    if _is_real_scalar(value):
        if not scalar_ok:
            raise SpatialLayerError(f"{name} must be an array of shape {shared_shape} or {cell_shape}, a scalar is not accepted")
        return _Input(name, LAYOUT_SCALAR, None, float(value), 0, 1)
    if isinstance(value, (bool, np.bool_)):
        raise SpatialLayerError(f"{name} must be numeric, got a Boolean")
    if type(value) is not np.ndarray and (ns_type is None or type(value) is not ns_type):
        raise SpatialLayerError(f"{name} must be exactly a numpy.ndarray or cupy.ndarray (no subclass, no masked array, no list), got {type(value).__module__}.{type(value).__name__}")
    if value.dtype != np.float64:
        raise SpatialLayerError(f"{name} must be float64, got {value.dtype}")
    if not value.flags.c_contiguous:
        raise SpatialLayerError(f"{name} must be C-contiguous")
    shape = tuple(int(n) for n in value.shape)
    if shape == tuple(shared_shape):
        return _Input(name, LAYOUT_SHARED, value, None, int(value.nbytes), int(shape[0]))
    if shape == tuple(cell_shape):
        return _Input(name, LAYOUT_CELL, value, None, int(value.nbytes), int(shape[0]))
    raise SpatialLayerError(f"{name} must have shape {shared_shape} (common) or {cell_shape} (per cell){' or be a real scalar' if scalar_ok else ''}, got {shape}")


def spatial_layer_spec(depth_edges_m: Any, *, theta_residual: Any, theta_field_capacity: Any, theta_saturation: Any, diffusivity_m2_s: Any = None,
                       interface_conductivity_m_s: Any = None, diffusion_coefficient: Any = None, theta_evaporation_stress: Any, evaporation_factor: Any = 1.0,
                       gravity_drainage: bool = True, diffusion_mode: str = "constant", bottom_drainage: str = "open", n_cells: Any) -> SpatialLayerSpec:
    """Check the METADATA of the explicit inputs (types, dtypes, shapes, modes, scalars) without reading an element or copying anything; returns an unprepared `SpatialLayerSpec`.
    Mode rules are those of `soil_layers`: "constant" requires `diffusivity_m2_s` and refuses the donor inputs; "tietjen_donor_depth" requires both `interface_conductivity_m_s` and
    `diffusion_coefficient` and refuses `diffusivity_m2_s`."""
    N = _count(n_cells, "n_cells", minimum=1)
    if _is_real_scalar(depth_edges_m) or isinstance(depth_edges_m, (bool, np.bool_)):
        raise SpatialLayerError("depth_edges_m must be an array of shape (L+1,) or (L+1, N)")
    ns_type = None  # NumPy arrays select the NumPy namespace directly: CuPy is not probed
    if type(depth_edges_m) is not np.ndarray:
        if type(depth_edges_m).__module__.split(".")[0] != "cupy":  # a list, a masked array, ... is simply not a supported array: no probing of CuPy or the GPU
            raise SpatialLayerError(f"depth_edges_m must be exactly a numpy.ndarray or cupy.ndarray, got {type(depth_edges_m).__module__}.{type(depth_edges_m).__name__}")
        ns_type = _cupy_ndarray_type("depth_edges_m")  # a CuPy-typed array was supplied: discover CuPy; an unusable CuPy is a typed setup failure
        if type(depth_edges_m) is not ns_type:
            raise SpatialLayerError(f"depth_edges_m must be exactly a numpy.ndarray or cupy.ndarray, got {type(depth_edges_m).__module__}.{type(depth_edges_m).__name__}")
    if depth_edges_m.ndim not in (1, 2) or depth_edges_m.shape[0] < 2:
        raise SpatialLayerError(f"depth_edges_m must have shape (L+1,) or (L+1, N) with L >= 1, got {tuple(depth_edges_m.shape)}")
    L = int(depth_edges_m.shape[0]) - 1
    namespace = "numpy" if type(depth_edges_m) is np.ndarray else "cupy"
    try:  # the accepted validators, re-raised as this module's typed error
        mode = ss._check_diffusion_mode(diffusion_mode)
        bottom = ss._check_bottom_drainage(bottom_drainage)
    except ss.SubsurfaceError as exc:
        raise SpatialLayerError(str(exc)) from None
    if type(gravity_drainage) is not bool:
        raise SpatialLayerError(f"gravity_drainage must be a bool, got {gravity_drainage!r}")
    inputs = {"depth_edges_m": _array_kind("depth_edges_m", depth_edges_m, (L + 1,), (L + 1, N), scalar_ok=False, ns_type=ns_type)}
    for key, value in (("theta_residual", theta_residual), ("theta_field_capacity", theta_field_capacity), ("theta_saturation", theta_saturation)):
        inputs[key] = _array_kind(key, value, (L,), (L, N), scalar_ok=True, ns_type=ns_type)
    if mode == "constant":
        if diffusivity_m2_s is None:
            raise SpatialLayerError("diffusion_mode='constant' requires diffusivity_m2_s")
        if interface_conductivity_m_s is not None or diffusion_coefficient is not None:
            raise SpatialLayerError("diffusion_mode='constant' refuses interface_conductivity_m_s / diffusion_coefficient (no silent reinterpretation)")
        iface_name, iface_value, d = "diffusivity_m2_s", diffusivity_m2_s, None
    else:
        if diffusivity_m2_s is not None:
            raise SpatialLayerError("diffusion_mode='tietjen_donor_depth' refuses diffusivity_m2_s")
        if interface_conductivity_m_s is None or diffusion_coefficient is None:
            raise SpatialLayerError("diffusion_mode='tietjen_donor_depth' requires BOTH interface_conductivity_m_s and diffusion_coefficient")
        iface_name, iface_value = "interface_conductivity_m_s", interface_conductivity_m_s
        d = _real(diffusion_coefficient, "diffusion_coefficient")
        if d < 0.0:
            raise SpatialLayerError(f"diffusion_coefficient must be >= 0, got {d!r}")
    if L > 1:
        inputs[iface_name] = _array_kind(iface_name, iface_value, (L - 1,), (L - 1, N), scalar_ok=True, ns_type=ns_type)
    else:  # one layer: a scalar or an EMPTY array (there is no interface)
        if _is_real_scalar(iface_value):  # no interface exists, but the accepted rule (finite, >= 0, even though no array results) still applies
            v = _real(iface_value, iface_name)
            if v < 0.0:
                raise SpatialLayerError(f"{iface_name} must be >= 0, got {v!r}")
            inputs[iface_name] = _Input(iface_name, LAYOUT_SCALAR, None, v, 0, 0)
        else:
            inputs[iface_name] = _array_kind(iface_name, iface_value, (0,), (0, N), scalar_ok=False, ns_type=ns_type)
    for key, v in inputs.items():
        if v.array is not None and type(v.array) is not type(depth_edges_m):
            raise SpatialLayerError(f"{key} is of a different array namespace than depth_edges_m; all array inputs must share one namespace (no implicit transfer)")
    stress = _real(theta_evaporation_stress, "theta_evaporation_stress")
    factor = _real(evaporation_factor, "evaporation_factor")
    if factor < 0.0:
        raise SpatialLayerError(f"evaporation_factor must be >= 0, got {factor!r}")
    return SpatialLayerSpec(L, N, namespace, MappingProxyType(inputs), mode, bottom, gravity_drainage, d, stress, factor)


def _revalidate_spec(spec: Any) -> SpatialLayerSpec:
    """Rebuild the spec from the LIVE arrays/scalars it references through the same factory (full canonical metadata check) and require that every cached field agrees. A caller may change an
    array's shape/dtype after the spec was made, or construct/replace a spec by hand: none of the cached metadata is trusted; ordinary metadata changes are typed refusals BEFORE any admission or allocation."""
    if type(spec) is not SpatialLayerSpec:
        raise SpatialLayerError(f"spec must be exactly a SpatialLayerSpec (no subclass), got {type(spec).__name__}")
    for name, kind in (("n_layers", int), ("n_cells", int), ("namespace", str), ("diffusion_mode", str), ("bottom_drainage", str)):
        if type(getattr(spec, name)) is not kind:
            raise SpatialLayerError(f"stale or edited spec: {name} must be a {kind.__name__}, got {type(getattr(spec, name)).__name__}")
    inputs = spec.inputs
    if not hasattr(inputs, "keys") or not hasattr(inputs, "__getitem__"):
        raise SpatialLayerError("stale or edited spec: inputs must be a mapping")
    mode = spec.diffusion_mode
    iface = "diffusivity_m2_s" if mode == "constant" else "interface_conductivity_m_s"
    expected = {"depth_edges_m", "theta_residual", "theta_field_capacity", "theta_saturation", iface}
    if mode not in ss.DIFFUSION_MODES or set(inputs.keys()) != expected:
        raise SpatialLayerError(f"stale or edited spec: inputs must be exactly {sorted(expected)} for mode {mode!r}, got {sorted(map(str, inputs.keys()))}")
    live = {}
    for key in expected:
        item = inputs[key]
        if type(item) is not _Input or item.name != key:
            raise SpatialLayerError(f"stale or edited spec: inputs[{key!r}] is not the canonical input record")
        live[key] = item.scalar if item.array is None else item.array
    kw = dict(theta_residual=live["theta_residual"], theta_field_capacity=live["theta_field_capacity"], theta_saturation=live["theta_saturation"],
              theta_evaporation_stress=spec.theta_evaporation_stress, evaporation_factor=spec.evaporation_factor, gravity_drainage=spec.gravity_drainage,
              diffusion_mode=mode, bottom_drainage=spec.bottom_drainage, n_cells=spec.n_cells)
    if mode == "constant":
        kw["diffusivity_m2_s"] = live[iface]
        if spec.diffusion_coefficient is not None:
            raise SpatialLayerError("stale or edited spec: diffusion_coefficient must be None in mode 'constant'")
    else:
        kw["interface_conductivity_m_s"], kw["diffusion_coefficient"] = live[iface], spec.diffusion_coefficient
    fresh = spatial_layer_spec(live["depth_edges_m"], **kw)
    stale = []
    if fresh.n_layers != spec.n_layers:
        stale.append(f"n_layers {spec.n_layers} -> {fresh.n_layers}")
    if fresh.namespace != spec.namespace:
        stale.append(f"namespace {spec.namespace} -> {fresh.namespace}")
    for key in expected:
        a, b = inputs[key], fresh.inputs[key]
        if (a.kind, a.rows, a.nbytes) != (b.kind, b.rows, b.nbytes):
            stale.append(f"{key}: recorded (kind {a.kind}, rows {a.rows}, {a.nbytes} B) but the live array is (kind {b.kind}, rows {b.rows}, {b.nbytes} B)")
    for name in ("gravity_drainage", "diffusion_coefficient", "theta_evaporation_stress", "evaporation_factor"):
        if type(getattr(spec, name)) is not type(getattr(fresh, name)) or getattr(spec, name) != getattr(fresh, name):
            stale.append(f"{name} {getattr(spec, name)!r} -> {getattr(fresh, name)!r}")
    if stale:
        raise SpatialLayerError("stale or edited spec (the caller changed an input's shape/dtype/layout or edited the spec); prepare again from a fresh spatial_layer_spec: " + "; ".join(stale))
    return fresh


# --- memory model --------------------------------------------------------------------------------------------------------------------
def _kind_bytes(kind: str, rows: int, n_cells: int) -> int:
    if rows <= 0:
        return 0
    return {LAYOUT_SCALAR: 8, LAYOUT_SHARED: 8 * rows, LAYOUT_CELL: 8 * rows * n_cells}[kind]


def _retained_layouts(L: int, layouts: dict, mode: str) -> dict[str, tuple[str, int]]:
    """name -> (kind, rows) of every retained map (absent maps for L = 1)."""
    iface = layouts["diffusivity_m2_s"] if mode == "constant" else layouts["interface_conductivity_m_s"]
    edges = layouts["depth_edges_m"]
    out = {"theta_residual": (layouts["theta_residual"], L), "theta_field_capacity": (layouts["theta_field_capacity"], L),
           "theta_saturation": (layouts["theta_saturation"], L), "dz": (edges, L)}
    if L > 1:
        join = LAYOUT_CELL if LAYOUT_CELL in (edges, iface) else LAYOUT_SHARED
        out["pair_inverse_thickness"] = (edges, L - 1)
        out["interface_conductance_down"] = (join, L - 1)
        if mode != "constant":
            out["interface_conductance_up"] = (join, L - 1)
    return out


@dataclass(frozen=True)
class SpatialMemoryEstimate:
    backend: str
    n_layers: int
    n_cells: int
    layouts: dict
    retained: dict  # name -> {kind, rows, bytes}
    retained_bytes: int
    mask_bytes: int
    input_bytes: dict  # the caller's arrays (actual bytes), EXCLUDED from the requirement
    chunk_columns: int
    hash_chunk_bytes: int
    main_required_bytes: int  # host for numpy/numba, device for cuda (includes the main reserve)
    host_required_bytes: int  # host side (== main for numpy/numba)
    device_required_bytes: int
    reserve_bytes: int
    host_reserve_bytes: int
    terms: dict
    scope: str = ("modelled bytes of this call's own arrays only; the caller's inputs, interpreter/JIT/NVRTC/driver state, the allocator pool policy and whole-process memory are excluded; "
                  "an availability reading is not a reservation")

    def __post_init__(self) -> None:
        for name in ("layouts", "retained", "input_bytes", "terms"):  # public metadata is read-only (deep): a caller cannot rewrite a recorded layout/plan
            object.__setattr__(self, name, _freeze(getattr(self, name)))


def estimate_spatial_memory(n_layers: int, n_cells: int, *, layouts: dict, diffusion_mode: str = "constant", backend: str = "numpy", chunk_columns: int = DEFAULT_CHUNK_COLUMNS,
                            hash_chunk_bytes: int = DEFAULT_HASH_CHUNK_BYTES, reserve_bytes: int = 0, host_reserve_bytes: int = 0,
                            input_bytes: dict | None = None) -> SpatialMemoryEstimate:
    """Metadata-only requirement of one `prepare_spatial_column` call. `layouts` maps depth_edges_m, theta_residual, theta_field_capacity, theta_saturation and the interface input
    (`diffusivity_m2_s` or `interface_conductivity_m_s`, by mode) to "scalar" | "shared" | "cell"."""
    L, N = _count(n_layers, "n_layers", minimum=1), _count(n_cells, "n_cells", minimum=1)
    if backend not in ss.BACKENDS:
        raise SpatialLayerError(f"backend must be one of {ss.BACKENDS}, got {backend!r}")
    mode = ss._check_diffusion_mode(diffusion_mode)
    chunk, hc = _count(chunk_columns, "chunk_columns", minimum=1), _count(hash_chunk_bytes, "hash_chunk_bytes", minimum=8)
    reserve, host_reserve = _count(reserve_bytes, "reserve_bytes"), _count(host_reserve_bytes, "host_reserve_bytes")
    if host_reserve and backend != "cuda":
        raise SpatialLayerError("host_reserve_bytes applies only to the cuda backend (the CPU backends use reserve_bytes on the host)")
    if backend == "cuda" and N > MAX_CUDA_CELLS:
        raise SpatialLayerError(f"the cuda backend supports at most {MAX_CUDA_CELLS} cells per call (64-bit packed diagnostic and the launch grid limit), got {N}")
    need = ["depth_edges_m", "theta_residual", "theta_field_capacity", "theta_saturation", "diffusivity_m2_s" if mode == "constant" else "interface_conductivity_m_s"]
    for key in need:
        if layouts.get(key) not in (LAYOUT_SCALAR, LAYOUT_SHARED, LAYOUT_CELL):
            raise SpatialLayerError(f"layouts[{key!r}] must be 'scalar', 'shared' or 'cell', got {layouts.get(key)!r}")
    if layouts["depth_edges_m"] == LAYOUT_SCALAR:
        raise SpatialLayerError("depth_edges_m cannot be a scalar")
    retained = {k: {"kind": kind, "rows": rows, "bytes": _kind_bytes(kind, rows, N)} for k, (kind, rows) in _retained_layouts(L, layouts, mode).items()}
    retained_bytes = sum(v["bytes"] for v in retained.values())
    mask_bytes = N
    chunk_eff = min(chunk, N)
    ref_work = REF_VEC_F64 * 8 * chunk_eff + REF_VEC_BOOL * chunk_eff + REF_VEC_U8 * chunk_eff if backend != "cuda" else 0
    hash_buf = hc if backend == "cuda" else 0
    packet = PACKET_BYTES if backend == "cuda" else 0
    inputs = dict(input_bytes or {})
    terms = {"retained_maps": retained_bytes, "mask_copy": mask_bytes, "numpy_chunk_work_vectors": ref_work, "cuda_host_identity_buffer": hash_buf, "cuda_packet": packet, "cuda_aux_array": AUX_BYTES if backend == "cuda" else 0,
             "python_allowance": PY_ALLOWANCE_BYTES, "cuda_kernel_temporaries": 0, "no_full_host_mirror": True, "no_LNK": True,
             "excluded": "NVRTC/kernel load, CUDA context, allocator pool policy, interpreter/JIT state; the device preparation uses no library reduction (no workspace exists)"}
    if backend == "cuda":
        device = retained_bytes + mask_bytes + PACKET_BYTES + AUX_BYTES + reserve  # the diagnostic words (read back to the host) and the 4-element auxiliary array live on the device
        host = hash_buf + packet + PY_ALLOWANCE_BYTES + host_reserve
        main = device
    else:
        host = retained_bytes + mask_bytes + ref_work + PY_ALLOWANCE_BYTES + reserve
        device, main = 0, host
    return SpatialMemoryEstimate(backend, L, N, dict(layouts), retained, retained_bytes, mask_bytes, inputs, chunk_eff, hc, main, host, device, reserve, host_reserve, terms)


def _admit(required: int, budget: int | None, reader: Callable[[], int] | None, side: str, est: SpatialMemoryEstimate, quota_name: str, cp: Any = None) -> None:
    if budget is not None and required > budget:
        raise SpatialLayerMemoryError(f"spatial layer preparation of {est.n_layers} layers x {est.n_cells} cells ({est.backend}) needs an estimated {required} B on the {side} side but the "
                                      f"caller's declared {quota_name} is {budget}; nothing was allocated. Scope: {est.scope}")
    if reader is None:
        reader = _cuda_module().default_device_available_bytes_fn(cp) if side == "device" else default_host_available_bytes
    available = reader()
    if isinstance(available, (bool, np.bool_)) or not isinstance(available, (int, np.integer)) or int(available) < 0:
        raise SpatialLayerError(f"the {side} available-bytes reader must return a non-Boolean integer >= 0, got {available!r}")
    if required > int(available):
        raise SpatialLayerMemoryError(f"spatial layer preparation of {est.n_layers} layers x {est.n_cells} cells ({est.backend}) needs an estimated {required} B on the {side} side but only "
                                      f"{int(available)} B are reported available; nothing was allocated. Scope: {est.scope}")


# --- the NumPy reference derivation (the definition of the arithmetic) ------------------------------------------------------------------
def _scalar_or_row(inp: _Input, k: int, s: int, e: int) -> Any:
    if inp.kind == LAYOUT_SCALAR:
        return np.float64(inp.scalar)
    if inp.kind == LAYOUT_SHARED:
        return inp.array[k]
    return inp.array[k, s:e]


def _bad(flags: np.ndarray, cond_bad: Any, bit: int, ok: np.ndarray | None = None) -> None:
    arr = np.broadcast_to(np.asarray(cond_bad, dtype=bool), flags.shape)
    if ok is not None:
        arr = arr & ok
    flags |= arr.astype(np.uint8) * np.uint8(bit)


def _derive_numpy(spec: SpatialLayerSpec, out: dict, coef: float, chunk: int) -> tuple[int, int, int | None, int]:
    """Validate and derive in chunks of `chunk` cells with O(chunk) work vectors, writing the retained maps in `out`. Returns `(flag OR, n_flagged, first flagged cell, its flags)`."""
    L, N = spec.n_layers, spec.n_cells
    mode_const = spec.diffusion_mode == "constant"
    dcoef = spec.diffusion_coefficient
    stress = np.float64(spec.theta_evaporation_stress)
    inp = spec.inputs
    edges, tr, tf, ts = inp["depth_edges_m"], inp["theta_residual"], inp["theta_field_capacity"], inp["theta_saturation"]
    ifc = inp["diffusivity_m2_s" if mode_const else "interface_conductivity_m_s"]
    or_all, n_flagged, first, first_flags = 0, 0, None, 0
    with np.errstate(all="ignore"):
        for s in range(0, N, chunk):
            e = min(N, s + chunk)
            flags = np.zeros(e - s, dtype=np.uint8)
            # ---- pass 1: the explicit inputs ----
            for k in range(L):
                e0, e1 = _scalar_or_row(edges, k, s, e), _scalar_or_row(edges, k + 1, s, e)
                dz = e1 - e0
                bad = ~(np.isfinite(e0) & np.isfinite(e1) & np.isfinite(dz) & (dz > 0.0))
                if k == 0:
                    bad = bad | ~(e0 == 0.0)
                _bad(flags, bad, FLAG_EDGES)
                r, f, sa = _scalar_or_row(tr, k, s, e), _scalar_or_row(tf, k, s, e), _scalar_or_row(ts, k, s, e)
                fin = np.isfinite(r) & np.isfinite(f) & np.isfinite(sa)
                _bad(flags, ~fin, FLAG_THETA)
                rel_bad = ~((r >= 0.0) & (f >= r) & (sa >= f) & (sa <= 1.0) & (sa > r))
                _bad(flags, rel_bad & fin, FLAG_REL)
                if k == 0:
                    _bad(flags, ~(stress > r) & np.isfinite(r), FLAG_STRESS)
                if k < L - 1:
                    v = _scalar_or_row(ifc, k, s, e)
                    _bad(flags, ~(np.isfinite(v) & (v >= 0.0)), FLAG_IFACE)
            ok = flags == 0
            # ---- pass 2: derived quantities (only for cells whose inputs passed) ----
            if ok.any():
                prev_inv = prev_centre = prev_dz = None
                for k in range(L):
                    e0, e1 = _scalar_or_row(edges, k, s, e), _scalar_or_row(edges, k + 1, s, e)
                    dz = e1 - e0
                    inv = 1.0 / dz
                    centre = e0 + dz / 2.0
                    bad = ~(np.isfinite(inv) & np.isfinite(centre))
                    r, f, sa = _scalar_or_row(tr, k, s, e), _scalar_or_row(tf, k, s, e), _scalar_or_row(ts, k, s, e)
                    res_m, fc_m, sat_m = r * dz, f * dz, sa * dz
                    bad = bad | ~(np.isfinite(res_m) & np.isfinite(fc_m) & np.isfinite(sat_m) & (sat_m > 0.0))
                    tol = coef * sat_m
                    bad = bad | ~(np.isfinite(tol) & ((sat_m + tol) > sat_m))
                    _write(out, "dz", edges.kind, k, s, e, dz)
                    if k > 0:
                        sep = centre - prev_centre
                        bad = bad | ~(np.isfinite(sep) & (sep > 0.0))
                        pair = prev_inv + inv
                        bad = bad | ~np.isfinite(pair)
                        v = _scalar_or_row(ifc, k - 1, s, e)
                        if mode_const:
                            g = v / sep
                            bad = bad | ~(np.isfinite(g) & np.isfinite(g * pair))
                            gu = None
                        else:
                            dd, du = (dcoef * v) * prev_dz, (dcoef * v) * dz
                            g, gu = dd / sep, du / sep
                            bad = bad | ~(np.isfinite(dd) & np.isfinite(du) & np.isfinite(g) & np.isfinite(gu) & np.isfinite(g * pair) & np.isfinite(gu * pair))
                        _write(out, "pair_inverse_thickness", edges.kind, k - 1, s, e, pair)
                        join = LAYOUT_CELL if LAYOUT_CELL in (edges.kind, ifc.kind) else LAYOUT_SHARED
                        _write(out, "interface_conductance_down", join, k - 1, s, e, g)
                        if gu is not None:
                            _write(out, "interface_conductance_up", join, k - 1, s, e, gu)
                    _bad(flags, bad, FLAG_DERIVED, ok)
                    prev_inv, prev_centre, prev_dz = inv, centre, dz
            nz = np.count_nonzero(flags)
            if nz:
                or_all |= int(np.bitwise_or.reduce(flags))
                n_flagged += int(nz)
                if first is None:
                    first = s + int(np.flatnonzero(flags)[0])
                    first_flags = int(flags[first - s])
    return or_all, n_flagged, first, first_flags


def _write(out: dict, name: str, kind: str, k: int, s: int, e: int, value: Any) -> None:
    arr = out[name]
    if kind == LAYOUT_CELL:
        arr[k, s:e] = value
    elif s == 0:
        arr[k] = value  # shared values are identical for every chunk: written once


# --- the CUDA kernel (one thread per cell; mirrors _derive_numpy statement by statement) ---------------------------------------------------
_SOURCE = r"""
#define A(p, sl, sc, l, c) (p)[(long long)(l) * (sl) + (long long)(c) * (sc)]
#define W(p, sl, sc, l, c, v) do { if ((sc) != 0 || (c) == 0) { (p)[(long long)(l) * (sl) + (long long)(c) * (sc)] = (v); } } while (0)

extern "C" __global__ void maple_syrup_spatial_prepare(
    const double* __restrict__ eo, const long long eL, const long long eC,
    const double* __restrict__ tr, const long long rL, const long long rC,
    const double* __restrict__ tf, const long long fL, const long long fC,
    const double* __restrict__ ts, const long long sL, const long long sC,
    const double* __restrict__ ic, const long long iL, const long long iC,
    double* __restrict__ dz_o, const long long dL, const long long dC,
    double* __restrict__ pv_o, const long long pL, const long long pC,
    double* __restrict__ gd_o, double* __restrict__ gu_o, const long long gL, const long long gC,
    const unsigned char* __restrict__ mask, unsigned long long* diag,
    const double coef, const double dcoef, const double stress, const long long L, const long long C, const int donor)
{
    const long long c = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;
    if (c >= C) return;
    if (mask[c]) atomicAdd(&diag[3], 1ULL);
    int flags = 0;
    for (long long k = 0; k < L; ++k) {
        double e0 = A(eo, eL, eC, k, c), e1 = A(eo, eL, eC, k + 1, c);
        double dz = e1 - e0;
        bool bad = !(isfinite(e0) && isfinite(e1) && isfinite(dz) && dz > 0.0);
        if (k == 0) bad = bad || !(e0 == 0.0);
        if (bad) flags |= FLAG_EDGES;
        double r = A(tr, rL, rC, k, c), f = A(tf, fL, fC, k, c), sa = A(ts, sL, sC, k, c);
        bool fin = isfinite(r) && isfinite(f) && isfinite(sa);
        if (!fin) flags |= FLAG_THETA;
        bool rel_bad = !((r >= 0.0) && (f >= r) && (sa >= f) && (sa <= 1.0) && (sa > r));
        if (rel_bad && fin) flags |= FLAG_REL;
        if (k == 0 && !(stress > r) && isfinite(r)) flags |= FLAG_STRESS;
        if (k < L - 1) {
            double v = A(ic, iL, iC, k, c);
            if (!(isfinite(v) && v >= 0.0)) flags |= FLAG_IFACE;
        }
    }
    if (flags == 0) {
        double prev_inv = 0.0, prev_centre = 0.0, prev_dz = 0.0;
        bool bad_d = false;
        for (long long k = 0; k < L; ++k) {
            double e0 = A(eo, eL, eC, k, c), e1 = A(eo, eL, eC, k + 1, c);
            double dz = e1 - e0;
            double inv = 1.0 / dz;
            double centre = e0 + dz / 2.0;
            bool bad = !(isfinite(inv) && isfinite(centre));
            double r = A(tr, rL, rC, k, c), f = A(tf, fL, fC, k, c), sa = A(ts, sL, sC, k, c);
            double res_m = r * dz, fc_m = f * dz, sat_m = sa * dz;
            bad = bad || !(isfinite(res_m) && isfinite(fc_m) && isfinite(sat_m) && sat_m > 0.0);
            double tol = coef * sat_m;
            bad = bad || !(isfinite(tol) && ((sat_m + tol) > sat_m));
            W(dz_o, dL, dC, k, c, dz);
            if (k > 0) {
                double sep = centre - prev_centre;
                bad = bad || !(isfinite(sep) && sep > 0.0);
                double pair = prev_inv + inv;
                bad = bad || !isfinite(pair);
                double v = A(ic, iL, iC, k - 1, c);
                double g, gu = 0.0;
                if (!donor) {
                    g = v / sep;
                    bad = bad || !(isfinite(g) && isfinite(g * pair));
                } else {
                    double dd = (dcoef * v) * prev_dz, du = (dcoef * v) * dz;
                    g = dd / sep; gu = du / sep;
                    bad = bad || !(isfinite(dd) && isfinite(du) && isfinite(g) && isfinite(gu) && isfinite(g * pair) && isfinite(gu * pair));
                }
                W(pv_o, pL, pC, k - 1, c, pair);
                W(gd_o, gL, gC, k - 1, c, g);
                if (donor) { W(gu_o, gL, gC, k - 1, c, gu); }
            }
            if (bad) bad_d = true;
            prev_inv = inv; prev_centre = centre; prev_dz = dz;
        }
        if (bad_d) flags |= FLAG_DERIVED;
    }
    if (flags != 0) {
        atomicOr(&diag[0], (unsigned long long)flags);
        atomicAdd(&diag[1], 1ULL);
        atomicMin(&diag[2], (((unsigned long long)c) << 8) | (unsigned long long)flags);
    }
}
"""
_LOCK = threading.Lock()
_KERNELS: dict[int, Any] = {}


def _cuda_source() -> str:
    defs = {"FLAG_EDGES": FLAG_EDGES, "FLAG_THETA": FLAG_THETA, "FLAG_REL": FLAG_REL, "FLAG_IFACE": FLAG_IFACE, "FLAG_DERIVED": FLAG_DERIVED, "FLAG_STRESS": FLAG_STRESS}
    return "".join(f"#define {k} {v}\n" for k, v in defs.items()) + _SOURCE


def _load_kernel(cp, device_id: int):
    with _LOCK:
        fn = _KERNELS.get(device_id)
        if fn is None:
            try:
                module = cp.RawModule(code=_cuda_source(), options=COMPILE_OPTIONS, backend="nvrtc")
                fn = module.get_function("maple_syrup_spatial_prepare")
            except Exception as exc:  # noqa: BLE001 - a setup failure, never a fallback
                raise SpatialLayerBackendUnavailableError(f"the CUDA preparation kernel could not be compiled/loaded on device {device_id} ({type(exc).__name__}: {exc})") from exc
            _KERNELS[device_id] = fn
        return fn


def _strides(kind: str, n_cells: int) -> tuple[int, int]:
    return {LAYOUT_SCALAR: (0, 0), LAYOUT_SHARED: (1, 0), LAYOUT_CELL: (int(n_cells), 1)}[kind]


def _derive_cuda(cp, spec: SpatialLayerSpec, mask: Any, out: dict, coef: float) -> tuple[tuple, int]:
    """One kernel launch (module docstring). Returns `((flag OR, n_flagged, first cell, its flags), active-cell count)`. Allocates only the 4-word diagnostic and ONE 4-element auxiliary
    array (a scalar interface input / the never-read stand-in pointer for absent interface maps); every other array is a retained map or a caller input."""
    L, N = spec.n_layers, spec.n_cells
    donor = spec.diffusion_mode != "constant"
    inp = spec.inputs
    ifc = inp["interface_conductivity_m_s" if donor else "diffusivity_m2_s"]
    rc = _cuda_module()
    aux = cp.zeros(4, dtype=cp.float64)  # aux[0] = the scalar interface value when the interface input is a scalar; otherwise a never-read stand-in
    if ifc.kind == LAYOUT_SCALAR:
        aux[0] = ifc.scalar
    args = []
    for key in ("depth_edges_m", "theta_residual", "theta_field_capacity", "theta_saturation"):
        i = inp[key]
        arr = i.array if key == "depth_edges_m" else out[key]  # the thetas are read from the retained detached copies (same layout)
        sl, sc = _strides(i.kind, N)
        args += [arr, np.int64(sl), np.int64(sc)]
    if L > 1 and ifc.kind != LAYOUT_SCALAR:
        isl, isc = _strides(ifc.kind, N)
        args += [ifc.array, np.int64(isl), np.int64(isc)]
    else:
        args += [aux, np.int64(0), np.int64(0)]
    edges_kind = inp["depth_edges_m"].kind
    join = LAYOUT_CELL if LAYOUT_CELL in (edges_kind, ifc.kind) else LAYOUT_SHARED
    dsl, dsc = _strides(edges_kind, N)
    gsl, gsc = _strides(join, N)
    pv_o = out.get("pair_inverse_thickness", aux)
    gd_o = out.get("interface_conductance_down", aux)
    gu_o = out.get("interface_conductance_up", aux)
    diag = cp.zeros(DIAG_WORDS, dtype=cp.uint64)
    diag[2] = np.uint64(0xFFFFFFFFFFFFFFFF)
    fn = _load_kernel(cp, rc.current_device_id(cp))
    grid = (N + CUDA_BLOCK - 1) // CUDA_BLOCK
    fn((grid,), (CUDA_BLOCK,), tuple(args) + (out["dz"], np.int64(dsl), np.int64(dsc), pv_o, np.int64(dsl), np.int64(dsc), gd_o, gu_o, np.int64(gsl), np.int64(gsc),
                                               mask, diag, np.float64(coef), np.float64(spec.diffusion_coefficient or 0.0), np.float64(spec.theta_evaporation_stress),
                                               np.int64(L), np.int64(N), np.int32(1 if donor else 0)))
    words = diag.get()  # the one device-to-host read: DIAG_WORDS raw uint64 words
    n_flagged = int(words[1])
    info = (0, 0, None, 0)
    if n_flagged:
        packed = int(words[2])
        info = (int(words[0]), n_flagged, packed >> 8, packed & 0xFF)
    return info, int(words[3])


# --- identity ------------------------------------------------------------------------------------------------------------------------
def _hash_array(h: Any, name: str, kind: str, array: Any, host_buf: np.ndarray | None) -> None:
    shape = tuple(int(n) for n in array.shape)
    h.update(f"{name}|{kind}|{shape}|float64\0".encode())
    if int(np.prod(shape)) == 0:
        return  # an empty array (no interface for one layer) carries only its header
    rows = shape[0] if len(shape) > 1 else 1
    flat = array.reshape(rows, -1) if len(shape) > 1 else array.reshape(1, -1)
    width = flat.shape[1]
    step = max(1, width if host_buf is None else host_buf.size)
    for row in range(rows):
        for a in range(0, width, step):
            piece = flat[row, a:a + step]
            if host_buf is None:
                h.update(memoryview(piece))
            else:
                n = int(piece.shape[0])
                piece.get(out=host_buf[:n])
                h.update(memoryview(host_buf[:n]))


def _identity(spec: SpatialLayerSpec, mask: Any, host_buf: np.ndarray | None) -> tuple[str, str]:
    h = hashlib.sha256()
    h.update(f"{SCHEMA}\0mode={spec.diffusion_mode};bottom={spec.bottom_drainage};gravity={spec.gravity_drainage};stress={spec.theta_evaporation_stress!r};"
             f"factor={spec.evaporation_factor!r};d={spec.diffusion_coefficient!r};L={spec.n_layers};N={spec.n_cells}\0".encode())
    for key in sorted(spec.inputs):
        i = spec.inputs[key]
        if i.kind == LAYOUT_SCALAR:
            h.update(f"{key}|scalar|{i.scalar!r}\0".encode())
        else:
            _hash_array(h, key, i.kind, i.array, host_buf)
    m = hashlib.sha256()
    m.update(f"{SCHEMA}.mask\0N={spec.n_cells}\0".encode())
    if host_buf is None:
        m.update(memoryview(np.ascontiguousarray(mask)))
    else:
        step = host_buf.size * 8  # reuse the float64 buffer's bytes for the bool mask pieces
        raw = host_buf.view(np.uint8)
        for a in range(0, spec.n_cells, step):
            piece = mask[a:a + step]
            n = int(piece.shape[0])
            piece.view(np.uint8).get(out=raw[:n])
            m.update(memoryview(raw[:n]))
    return h.hexdigest(), m.hexdigest()


# --- the prepared owner ----------------------------------------------------------------------------------------------------------------
# What `hydrology_step_supported = False` means, in one place. The OWNER is parameter-component-only / native-binding metadata: it has no step method, it is NOT a `subsurface.SoilColumn` and is refused by
# every legacy/production entry point (operators, lifecycle, checkpoints). Water evolution exists ONLY through the explicit module-level NumPy TRIAL functions below on a "numpy"-prepared owner.
_WATER_EVOLUTION = MappingProxyType({
    "owner_is_parameter_component_only": True, "legacy_soil_column_compatible": False, "production_or_lifecycle_accepted": False,
    "explicit_trial_functions": ("spatial_column_state", "spatial_uniform_state", "spatial_moisture", "spatial_storage_m3", "spatial_step", "spatial_step_with_recharge"),
    "evolution_backends": ("numpy", "numba", "cuda"), "unsupported_backends": (), "state_component_tags": ("spatial_explicit_cpu_state", "spatial_explicit_cuda_state"),
    "cuda_trial_status": "CUDA trial component (S2-A.4.2): this source asserts no qualification; any qualification of a specific source is recorded only in Root's source-specific receipts; native, lifecycle, restart and storm use is unsupported"})
_CONSTRUCT = threading.local()


@contextlib.contextmanager
def _constructing():
    """Only `prepare_spatial_column` may construct an owner: a direct constructor call, `dataclasses.replace` or a copy outside this context is refused."""
    _CONSTRUCT.allowed = True
    try:
        yield
    finally:
        _CONSTRUCT.allowed = False


_MODULE_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


@dataclass(frozen=True, eq=False)
class SpatialSoilColumn:
    """Sealed owner of the DETACHED retained maps of ONE explicit spatial parameter set. `component_tag = 'parameter_component_only'`: no hydrology step, state or kernel exists on it, and it is NOT a
    `SoilColumn` (every existing entry point refuses it). `hydrology_step_supported = False` describes THIS OWNER (parameter-component-only, native-binding metadata: no step method, not a legacy/production
    column); it does not contradict the explicit trial functions `spatial_step`/`spatial_step_with_recharge` of this module (see `summary()["water_evolution"]`), which evolve a `SpatialSoilState` on a
    "numpy"-, "numba"- or "cuda"-prepared owner (the CUDA trial of S2-A.4.2: qualification of a specific source is recorded only in Root's receipts, never asserted here) and imply no lifecycle, checkpoint or production acceptance. Arrays are private (`_maps`): read-only NumPy arrays, or CuPy arrays private to the column; use `summary()` (bounded metadata) or `inspect()` (an
    ADMITTED detached host copy) to look at values."""

    backend: str
    n_layers: int
    n_cells: int
    n_active: int
    cell_area_m2: float
    diffusion_mode: str
    bottom_drainage: str
    gravity_drainage: bool
    diffusion_coefficient: float | None
    theta_evaporation_stress: float
    evaporation_factor: float
    layouts: Any  # read-only mapping name -> layout kind
    parameters_sha256: str
    mask_sha256: str
    estimate: SpatialMemoryEstimate
    device_id: int | None
    _maps: dict
    _mask: Any
    component_tag: str = COMPONENT_TAG
    hydrology_step_supported: bool = False

    def __post_init__(self) -> None:
        if not getattr(_CONSTRUCT, "allowed", False):
            raise SpatialLayerError("a SpatialSoilColumn is created only by prepare_spatial_column (no direct construction, replace() or copy)")
        object.__setattr__(self, "layouts", _freeze(dict(self.layouts)))  # canonical public layout metadata is read-only
        object.__setattr__(self, "_maps", MappingProxyType(dict(self._maps)))

    def __copy__(self):
        raise SpatialLayerError("a SpatialSoilColumn owns private maps and is not copyable; prepare another from its spec")

    def __deepcopy__(self, memo):
        raise SpatialLayerError("a SpatialSoilColumn owns private maps and is not copyable; prepare another from its spec")

    def __reduce_ex__(self, protocol):
        raise SpatialLayerError("a SpatialSoilColumn is not serialisable; prepare another from its spec")

    @property
    def sha256(self) -> str:
        return hashlib.sha256(f"{self.parameters_sha256}|{self.mask_sha256}|{self.cell_area_m2!r}".encode()).hexdigest()

    @property
    def retained_bytes(self) -> int:
        return self.estimate.retained_bytes

    def retained_names(self) -> tuple[str, ...]:
        return tuple(self._maps)

    def summary(self) -> dict:
        """Bounded metadata only (no array value, no reduction, no host copy)."""
        return {"component_tag": self.component_tag, "schema": SCHEMA, "module_sha256": _MODULE_SHA, "hydrology_step_supported": False, "water_evolution": {k: (list(v) if isinstance(v, tuple) else v) for k, v in _WATER_EVOLUTION.items()}, "backend": self.backend, "n_layers": self.n_layers,
                "n_cells": self.n_cells, "n_active": self.n_active, "cell_area_m2": self.cell_area_m2, "diffusion_mode": self.diffusion_mode, "bottom_drainage": self.bottom_drainage,
                "gravity_drainage": self.gravity_drainage, "diffusion_coefficient": self.diffusion_coefficient, "theta_evaporation_stress": self.theta_evaporation_stress,
                "evaporation_factor": self.evaporation_factor, "input_layouts": dict(self.layouts), "retained": {k: dict(v) for k, v in self.estimate.retained.items()},
                "retained_bytes": self.retained_bytes, "mask_bytes": self.estimate.mask_bytes, "parameters_sha256": self.parameters_sha256, "mask_sha256": self.mask_sha256,
                "sha256": self.sha256}

    def inspect(self, name: str, *, host_memory_budget_bytes: Any, host_reserve_bytes: Any = 0, host_available_bytes_fn: Callable[[], int] | None = None) -> np.ndarray:
        """A DETACHED host copy of one retained map, admitted against the host quota/reading BEFORE it is made (a cold diagnostic; never used by a physical step)."""
        if name not in self._maps:
            raise SpatialLayerError(f"unknown retained map {name!r}; available: {tuple(self._maps)}")
        budget, reserve = _count(host_memory_budget_bytes, "host_memory_budget_bytes"), _count(host_reserve_bytes, "host_reserve_bytes")
        arr = self._maps[name]
        need = int(arr.nbytes) + reserve
        if need > budget:
            raise SpatialLayerMemoryError(f"inspecting {name!r} needs {need} B on the host but the declared host_memory_budget_bytes is {budget}; nothing was copied")
        reader = default_host_available_bytes if host_available_bytes_fn is None else host_available_bytes_fn
        available = reader()
        if isinstance(available, (bool, np.bool_)) or not isinstance(available, (int, np.integer)) or int(available) < 0:
            raise SpatialLayerError(f"the host available-bytes reader must return a non-Boolean integer >= 0, got {available!r}")
        if need > int(available):
            raise SpatialLayerMemoryError(f"inspecting {name!r} needs {need} B on the host but only {int(available)} B are reported available; nothing was copied")
        out = arr.get() if hasattr(arr, "get") else np.array(arr, dtype=np.float64, copy=True)  # exactly one detached host copy
        out.flags.writeable = True
        return out


def prepare_spatial_column(spec: Any, active_mask: Any, cell_area_m2: Any, *, backend: str = "numpy", memory_budget_bytes: Any, reserve_bytes: Any = 0,
                           available_bytes_fn: Callable[[], int] | None = None, host_memory_budget_bytes: Any = None, host_reserve_bytes: Any = 0,
                           host_available_bytes_fn: Callable[[], int] | None = None, chunk_columns: Any = None, hash_chunk_bytes: Any = DEFAULT_HASH_CHUNK_BYTES) -> SpatialSoilColumn:
    """Validate the explicit inputs of `spec`, derive and OWN the detached retained maps (module docstring), and return a `SpatialSoilColumn`.

    `active_mask` `(N,)` bool in the backend's namespace; `cell_area_m2` finite positive host scalar (stored for later volume conversion). Quotas/readers follow the S1 convention: `memory_budget_bytes`
    (required), `reserve_bytes` and `available_bytes_fn` apply to the main side (host for numpy/numba, DEVICE for cuda); the host side of the CUDA call (identity buffer + packet) is admitted separately
    against `host_memory_budget_bytes` (None = no separate quota), `host_reserve_bytes` and `host_available_bytes_fn` (default the `sysconf` reading) and these three are refused for the CPU backends.
    `chunk_columns` (default 65536) bounds the NumPy work vectors; `hash_chunk_bytes` bounds the identity streaming. Raises `SpatialLayerError` (metadata/value), `SpatialLayerMemoryError`,
    `SpatialLayerBackendUnavailableError`; nothing is published on failure and the inputs are never modified or retained."""
    if isinstance(spec, ss.SoilLayers):
        raise SpatialLayerError("prepare_spatial_column takes a SpatialLayerSpec, not a SoilLayers (use subsurface.prepare_column for the common-layer column)")
    if not isinstance(spec, SpatialLayerSpec):
        raise SpatialLayerError(f"spec must be a SpatialLayerSpec, got {type(spec).__name__}")
    spec = _revalidate_spec(spec)  # the cached metadata is never trusted: the live arrays are re-checked BEFORE any admission, allocation or kernel
    if backend not in ss.BACKENDS:
        raise SpatialLayerError(f"backend must be one of {ss.BACKENDS}, got {backend!r}")
    area = _real(cell_area_m2, "cell_area_m2")
    if area <= 0.0:
        raise SpatialLayerError(f"cell_area_m2 must be > 0, got {area!r}")
    budget, reserve = _count(memory_budget_bytes, "memory_budget_bytes"), _count(reserve_bytes, "reserve_bytes")
    host_reserve = _count(host_reserve_bytes, "host_reserve_bytes")
    host_budget = None if host_memory_budget_bytes is None else _count(host_memory_budget_bytes, "host_memory_budget_bytes")
    for fn_name, fn in (("available_bytes_fn", available_bytes_fn), ("host_available_bytes_fn", host_available_bytes_fn)):
        if fn is not None and not callable(fn):
            raise SpatialLayerError(f"{fn_name} must be callable or None")
    if backend != "cuda" and (host_reserve or host_budget is not None or host_available_bytes_fn is not None):
        raise SpatialLayerError("host_memory_budget_bytes / host_reserve_bytes / host_available_bytes_fn apply only to the cuda backend")
    chunk = DEFAULT_CHUNK_COLUMNS if chunk_columns is None else _count(chunk_columns, "chunk_columns", minimum=1)
    hc = _count(hash_chunk_bytes, "hash_chunk_bytes", minimum=8)
    want_ns = "cupy" if backend == "cuda" else "numpy"
    cp = None
    if backend == "cuda":
        cp = _require_cuda()
    elif backend == "numba":
        try:
            import numba  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            raise SpatialLayerBackendUnavailableError(f"backend 'numba' requires Numba ({type(exc).__name__}: {exc}); there is no fallback") from exc
    if spec.namespace != want_ns:
        raise SpatialLayerError(f"the spec's arrays live in the {spec.namespace} namespace but backend {backend!r} requires {want_ns}; no implicit transfer or conversion")
    L, N = spec.n_layers, spec.n_cells
    # mask metadata
    expected = cp.ndarray if cp is not None else np.ndarray
    if type(active_mask) is not expected or active_mask.dtype != np.bool_ or tuple(active_mask.shape) != (N,) or not active_mask.flags.c_contiguous:
        raise SpatialLayerError(f"active_mask must be exactly a {want_ns} bool C-contiguous array of shape ({N},)")
    if cp is not None:
        rc = _cuda_module()
        for key, i in spec.inputs.items():
            if i.array is not None and int(i.array.device.id) != rc.current_device_id(cp):
                raise SpatialLayerError(f"{key} lives on device {int(i.array.device.id)} but the current device is {rc.current_device_id(cp)}; no implicit transfer")
        if int(active_mask.device.id) != rc.current_device_id(cp):
            raise SpatialLayerError("active_mask lives on another device than the current one; no implicit transfer")
    # admission BEFORE any allocation
    est = estimate_spatial_memory(L, N, layouts=spec.layouts(), diffusion_mode=spec.diffusion_mode, backend=backend, chunk_columns=chunk, hash_chunk_bytes=hc, reserve_bytes=reserve,
                                  host_reserve_bytes=host_reserve, input_bytes=spec.input_bytes())
    if backend == "cuda":
        _admit(est.device_required_bytes, budget, available_bytes_fn, "device", est, "memory_budget_bytes", cp)
        _admit(est.host_required_bytes, host_budget, host_available_bytes_fn, "host", est, "host_memory_budget_bytes")
    else:
        _admit(est.main_required_bytes, budget, available_bytes_fn, "host", est, "memory_budget_bytes")
    if backend == "numba":  # S2-A.3: startup cost of the spatial variant of the ONE shared compiled body sits here (outside every step); the JIT itself is excluded from the memory model
        from maple_syrup import subsurface_numba

        try:
            subsurface_numba.compiled_spatial_kernel()
        except ss.SubsurfaceBackendUnavailableError as exc:
            raise SpatialLayerBackendUnavailableError(str(exc)) from exc
    from maple_syrup.conservation import volume_roundoff_bound_m3

    coef = volume_roundoff_bound_m3(ss.LAYER_OPERATION_TERMS, 1.0)
    xp = cp if cp is not None else np
    # retained maps: detached copies of the thetas, empty derived maps
    maps: dict[str, Any] = {}
    for key in ("theta_residual", "theta_field_capacity", "theta_saturation"):
        i = spec.inputs[key]
        maps[key] = xp.full(1, i.scalar, dtype=xp.float64) if i.kind == LAYOUT_SCALAR else xp.array(i.array, dtype=xp.float64, copy=True)
    for name, (kind, rows) in _retained_layouts(L, spec.layouts(), spec.diffusion_mode).items():
        if name not in maps:
            maps[name] = xp.zeros((rows, N) if kind == LAYOUT_CELL else (rows,), dtype=xp.float64)
    mask_copy = xp.array(active_mask, dtype=xp.bool_, copy=True)
    if cp is None:
        info = _derive_numpy(spec, maps, coef, min(chunk, N))
        n_active = int(np.count_nonzero(mask_copy))
    else:
        info, n_active = _derive_cuda(cp, spec, mask_copy, maps, coef)
    _raise(info)
    host_buf = np.empty(max(1, hc // 8), dtype=np.float64) if cp is not None else None
    parameters_sha, mask_sha = _identity(spec, mask_copy, host_buf)
    if cp is None:
        for arr in maps.values():
            arr.flags.writeable = False
        mask_copy.flags.writeable = False
    device_id = None if cp is None else _cuda_module().current_device_id(cp)
    with _constructing():
        return SpatialSoilColumn(backend, L, N, n_active, area, spec.diffusion_mode, spec.bottom_drainage, spec.gravity_drainage, spec.diffusion_coefficient,
                                 spec.theta_evaporation_stress, spec.evaporation_factor, spec.layouts(), parameters_sha, mask_sha, est, device_id, maps, mask_copy)


def _raise(info: tuple) -> None:
    bits, n_flagged, first, first_flags = info
    if not bits:
        return
    low = first_flags & -first_flags
    raise SpatialLayerError(f"{_FLAG_TEXT[low]} ({n_flagged} cell(s) flagged, first cell {first}, its flags {first_flags}; OR of all flags {bits}); nothing was published",
                            bit=low, n_flagged=n_flagged, first_cell=first)


# =====================================================================================================================================
# S2-A.2/A.3: CPU water evolution of a prepared SpatialSoilColumn: backend "numpy" on the ONE existing reference body (`subsurface._reference_step_body`), backend "numba" on the ONE compiled body
# (`subsurface_numba` spatial variant of the ONE body shared with the common-layer Numba column)
# =====================================================================================================================================
# Scope: backends "numpy" and "numba" (a cuda-prepared owner is refused before any copy or kernel -- S2-A.4 is later work; there is never a fallback between backends), fixed EXPLICIT geometry/properties, no interpolation or remapping, no
# LEAF/transpiration, no restart/lifecycle claim, no frame/water-fate/PTF/averaging/wetting-history rule. The physics is NOT copied here: `_SpatialStepParams` only adapts the retained maps to the
# indexable per-layer reads of `subsurface._reference_step_body` (the same body `subsurface.reference_step` runs for a common column); the "numba" path hands the SAME retained RAW maps, as flat
# read-only views with their layout codes, to the compiled body, which derives the metre bounds inline (no (L, N) depth/tolerance tensor, no copy of a map or of the mask). Per-layer values are derived LAZILY, row by row, from the
# retained RAW maps with the same IEEE operations as `soil_layers`: `res_m[k] = theta_res[k]*dz[k]`, `fc_m[k]`, `sat_m[k]` likewise and `cap_tol[k] = coef*sat_m[k]` (coef =
# `volume_roundoff_bound_m3(LAYER_OPERATION_TERMS, 1.0)`); no permanent (L, N) depth/tolerance map exists. A row is a scalar for a SCALAR/SHARED layout and an `(N,)` array for a CELL layout.
STATE_COMPONENT_TAG = "spatial_explicit_cpu_state"  # host state: the same state object is stepped by the NumPy reference body or the compiled Numba body
STATE_COMPONENT_TAG_CUDA = "spatial_explicit_cuda_state"  # device state (CuPy arrays of the owner's device); a CPU state and a CUDA owner are never converted into each other
STATE_COMPONENT_TAGS = (STATE_COMPONENT_TAG, STATE_COMPONENT_TAG_CUDA)
SPATIAL_ROW_TEMP_DOUBLES = 8  # declared extra live (N,) float64 rows of the lazily derived per-layer values on top of the common NumPy scratch (checked by tracemalloc in the tests)


class _Rows:
    """Indexable lazy per-layer reader: `rows[k]` evaluates one row (no cache, no (L, N) array)."""

    __slots__ = ("_fn",)

    def __init__(self, fn: Callable[[int], Any]) -> None:
        self._fn = fn

    def __getitem__(self, k: int) -> Any:
        return self._fn(k)


def _retained_row(arr: np.ndarray, kind: str, k: int) -> Any:
    if kind == LAYOUT_SCALAR:
        return arr[0]
    return arr[k]  # SHARED -> a float64 scalar, CELL -> an (N,) view of the owner's read-only map


class _SpatialStepParams:
    """PRIVATE typed parameter reader over a SpatialSoilColumn's retained RAW maps for `subsurface._reference_step_body`. It is NOT a legacy SoilLayers/SoilColumn and claims no legacy ownership."""

    def __init__(self, column: "SpatialSoilColumn") -> None:
        from maple_syrup.conservation import volume_roundoff_bound_m3

        maps, kinds = column._maps, {n: v["kind"] for n, v in column.estimate.retained.items()}
        coef = volume_roundoff_bound_m3(ss.LAYER_OPERATION_TERMS, 1.0)
        tr, tf, ts, dz = maps["theta_residual"], maps["theta_field_capacity"], maps["theta_saturation"], maps["dz"]
        ktr, ktf, kts, kdz = kinds["theta_residual"], kinds["theta_field_capacity"], kinds["theta_saturation"], kinds["dz"]
        self.n_layers, self.n_cells = column.n_layers, column.n_cells
        self.dz = _Rows(lambda k: _retained_row(dz, kdz, k))
        self.res_m = _Rows(lambda k: _retained_row(tr, ktr, k) * _retained_row(dz, kdz, k))
        self.fc_m = _Rows(lambda k: _retained_row(tf, ktf, k) * _retained_row(dz, kdz, k))
        self.sat_m = _Rows(lambda k: _retained_row(ts, kts, k) * _retained_row(dz, kdz, k))
        self.cap_tol = _Rows(lambda k: coef * (_retained_row(ts, kts, k) * _retained_row(dz, kdz, k)))
        if column.n_layers > 1:
            pv, gd = maps["pair_inverse_thickness"], maps["interface_conductance_down"]
            kpv, kgd = kinds["pair_inverse_thickness"], kinds["interface_conductance_down"]
            self.pair_inv = _Rows(lambda k: _retained_row(pv, kpv, k))
            self.cond_down = _Rows(lambda k: _retained_row(gd, kgd, k))
            if column.diffusion_mode != "constant":
                gu, kgu = maps["interface_conductance_up"], kinds["interface_conductance_up"]
                self.cond_up = _Rows(lambda k: _retained_row(gu, kgu, k))
            else:
                self.cond_up = self.cond_down  # constant mode: the "up" conductance IS the down one (unused: not directional)
        else:  # one layer: no interface maps exist (the body never reads them)
            self.pair_inv = self.cond_down = self.cond_up = _Rows(lambda k: (_ for _ in ()).throw(SpatialLayerError("a one-layer column has no interface maps")))
        self.directional = column.diffusion_mode != "constant"
        self.closed_bottom = column.bottom_drainage == "closed"
        self.gravity_drainage = column.gravity_drainage
        self.res_theta0 = _retained_row(tr, ktr, 0)  # layer-0 residual theta: a scalar or an (N,) row, never a new map
        self.stress, self.factor = column.theta_evaporation_stress, column.evaporation_factor


@dataclass(frozen=True, eq=False)
class SpatialSoilState:
    """Water state of ONE SpatialSoilColumn: `water_m` `(L, N)` float64 C-contiguous (m of water per layer; read-only), the physical time `t_s` (s), and the identity (`column_sha256`) of the
    parameter set and mask it belongs to, so it cannot silently be used with other geometry/properties. Created only by this module's functions or by a step; steps never write it. A CPU state's NumPy array is flagged read-only; a CUDA state's CuPy array has no such flag, so its immutability is LOGICAL only (module-created, `const` kernel arguments, fresh arrays per step, owner-bound identity): nothing detects an external write."""

    water_m: Any
    t_s: float
    column_sha256: str
    n_layers: int
    n_cells: int
    component_tag: str = STATE_COMPONENT_TAG
    namespace: str = "numpy"  # "numpy" | "cupy"
    device_id: int | None = None  # the owner's device for a CUDA state

    def summary(self) -> dict:
        return {"component_tag": self.component_tag, "t_s": self.t_s, "column_sha256": self.column_sha256, "shape": [self.n_layers, self.n_cells], "bytes": int(self.water_m.nbytes),
                "namespace": self.namespace, "device_id": self.device_id}


@dataclass(frozen=True, eq=False)
class SpatialStep:
    """Result of one spatial step (same fields as `subsurface.SubsurfaceStep`, with a `SpatialSoilState`)."""

    state: SpatialSoilState
    dt_s: float
    surface_return_m: Any  # (N,)
    drainage_m: Any  # (N,)
    evaporation_m: Any  # (N,) actual withdrawal from layer 0
    evaporation_unmet_m: Any  # (N,)
    percolation_m: Any  # (L - 1, N)
    diffusion_m: Any  # (L - 1, N), positive downward
    budget: ss.StepBudget


@dataclass(frozen=True)
class SpatialOperationMemory:
    """Modelled bytes of ONE allocating spatial operation (host/NumPy). Caller-owned arrays are EXCLUDED (`input_bytes`); the prepared owner's retained maps are already resident and are not re-counted."""

    operation: str
    n_layers: int
    n_cells: int
    output_bytes: int  # arrays that survive the call (state / result arrays)
    scratch_bytes: int  # transient work (NumPy scratch, derived rows, flags/masks, packet, Python allowance)
    reserve_bytes: int
    required_bytes: int  # output + scratch + reserve
    input_bytes: dict
    terms: dict
    scope: str = ("modelled bytes of this call's own arrays only (host NumPy); the caller's state/forcing arrays and the owner's retained maps are excluded; interpreter/allocator state and "
                  "whole-process memory are excluded; an availability reading is not a reservation")
    device_required_bytes: int = 0  # CUDA: the device requirement (== required_bytes); CPU backends: 0
    host_required_bytes: int = -1  # CUDA: the separate HOST requirement (incl. the Python allowance and the host reserve); CPU backends: == required_bytes
    host_reserve_bytes: int = 0

    def __post_init__(self) -> None:
        if self.host_required_bytes < 0:
            object.__setattr__(self, "host_required_bytes", self.required_bytes)
        object.__setattr__(self, "input_bytes", _freeze(dict(self.input_bytes)))
        object.__setattr__(self, "terms", _freeze(dict(self.terms)))


_OPERATIONS = ("state", "uniform_state", "moisture", "storage", "step", "step_with_recharge")
EVOLUTION_BACKENDS = ("numpy", "numba", "cuda")
NUMBA_MASK_BYTES_PER_CELL = 1  # the one `rows[3] < NO_FLAG` bool row of the packet read; the compiled body has no other per-cell scratch


def estimate_spatial_operation_memory(operation: str, n_layers: int, n_cells: int, *, with_evaporation_share: bool = False, reserve_bytes: int = 0,
                                      backend: str = "numpy", host_reserve_bytes: int = 0, hoisted_factor_arrays: int = 0) -> SpatialOperationMemory:
    """Pure arithmetic requirement of one spatial operation (see the module notes): `state`/`uniform_state`/`moisture` allocate one `(L, N)` array (+ a few `(N,)` rows for the uniform
    check), `storage` one `(N,)` accumulator and one mask, `step`/`step_with_recharge` the outputs of the common step `(LN + 2(L-1)N + 4N + ROW_COUNT*N)*8`, the NumPy scratch of the common
    estimate (`NUMPY_SCRATCH_DOUBLES_PER_CELL` doubles + `SCRATCH_MASK_BYTES_PER_CELL` bytes per cell + the packet), `SPATIAL_ROW_TEMP_DOUBLES` extra `(N,)` rows for the lazily derived values
    and the 64 KiB Python allowance. A step with the forcing tensor has caller inputs `(2LN + N)*8` (state, added water, potential rate), the pulse step `(LN + N)*8`; a share adds `N*8`.
    `backend="numba"` changes ONLY the two step operations: the compiled body keeps no `(N,)` float64 work row, so the scratch is the one `N`-byte packet bool row, the kernel's per-call LOCAL
    arrays (`subsurface_numba.local_scratch_doubles(L)` doubles, Numba NRT heap, independent of `N`: five per layer, three per interface) and the packet and Python allowance; the outputs, inputs and
    every other operation are those of the NumPy path (a host NumPy operation on a numba-prepared owner). `backend="cuda"` (S2-A.4.2) returns the DEVICE requirement in `required_bytes`/
    `device_required_bytes` (outputs + scratch + the device reserve) and the separate HOST requirement in `host_required_bytes` (the call's own host allocations + the Python allowance + `host_reserve_bytes`);
    the exact terms are `subsurface_cuda_spatial.operation_terms` (`hoisted_factor_arrays` = 0 when the interface factors are evaluated per cell, else 1 or 2). Another backend is refused."""
    if backend not in EVOLUTION_BACKENDS:
        raise SpatialLayerError(f"backend must be one of {EVOLUTION_BACKENDS} for a spatial water operation, got {backend!r}")
    if operation not in _OPERATIONS:
        raise SpatialLayerError(f"operation must be one of {_OPERATIONS}, got {operation!r}")
    L, N = _count(n_layers, "n_layers", minimum=1), _count(n_cells, "n_cells", minimum=1)
    reserve = _count(reserve_bytes, "reserve_bytes")
    if type(with_evaporation_share) is not bool:
        raise SpatialLayerError(f"with_evaporation_share must be a bool, got {with_evaporation_share!r}")
    host_reserve = _count(host_reserve_bytes, "host_reserve_bytes")
    factors = _count(hoisted_factor_arrays, "hoisted_factor_arrays")
    if backend != "cuda" and (host_reserve or factors):
        raise SpatialLayerError("host_reserve_bytes / hoisted_factor_arrays apply only to the cuda backend")
    if backend == "cuda":
        return _estimate_cuda(operation, L, N, with_evaporation_share, reserve, host_reserve, factors)
    f8, n_if = 8, L - 1
    inputs: dict = {}
    if operation == "state":
        out, scratch, inputs = f8 * L * N, 0, {"water_m": f8 * L * N}
    elif operation == "uniform_state":
        # (N,) comparison rows of the per-cell saturation check + the L-sized conversion lifetimes of `theta_by_layer` (scalar -> `np.full(L)`; sequence/array -> `np.asarray` (L doubles) +
        # the owned float64 copy (L doubles) + the L-byte finite mask), declared as THREE L-double arrays (24 L bytes) and counted BEFORE the conversion is made
        out, scratch = f8 * L * N, f8 * N + N + 3 * f8 * L
    elif operation == "moisture":
        out, scratch, inputs = f8 * L * N, 0, {"state.water_m": f8 * L * N}
    elif operation == "storage":
        out, scratch, inputs = 0, f8 * N + N, {"state.water_m": f8 * L * N}
    else:
        pulse = operation == "step_with_recharge"
        out = (L * N + 2 * n_if * N + 4 * N + ss.ROW_COUNT * N) * f8
        if backend == "numba":
            from maple_syrup import subsurface_numba

            scratch = NUMBA_MASK_BYTES_PER_CELL * N + subsurface_numba.local_scratch_doubles(L) * f8 + ss._PACKET_SCRATCH_BYTES
        else:
            scratch = (ss.NUMPY_SCRATCH_DOUBLES_PER_CELL + SPATIAL_ROW_TEMP_DOUBLES) * N * f8 + ss.SCRATCH_MASK_BYTES_PER_CELL * N + ss._PACKET_SCRATCH_BYTES
        inputs = {"state.water_m": f8 * L * N, "potential_evaporation_m_s": f8 * N}
        if not pulse:
            inputs["added_water_m"] = f8 * L * N
        if with_evaporation_share:
            inputs["evaporation_share"] = f8 * N
    scratch += PY_ALLOWANCE_BYTES
    terms = {"outputs": out, "scratch": scratch, "theta_conversion_bytes": 3 * f8 * L if operation == "uniform_state" else 0, "numpy_scratch_doubles_per_cell": ss.NUMPY_SCRATCH_DOUBLES_PER_CELL, "spatial_row_temp_doubles_per_cell": SPATIAL_ROW_TEMP_DOUBLES, "backend": backend,
             "no_permanent_LN_derived_maps": True, "no_internal_zero_forcing_tensor": True, "no_history": True}
    return SpatialOperationMemory(operation, L, N, out, scratch, reserve, out + scratch + reserve, inputs, terms)


def _estimate_cuda(operation: str, L: int, N: int, with_share: bool, reserve: int, host_reserve: int, factors: int) -> SpatialOperationMemory:
    from maple_syrup import subsurface_cuda_spatial as cs

    cs.check_launch_ranges(L, N)
    f8 = 8
    t = cs.operation_terms(operation, L, N, factors=factors)
    inputs: dict = {}
    if operation == "state":
        inputs = {"water_m": f8 * L * N}
    elif operation == "uniform_state":
        inputs = {"theta_by_layer (host)": f8 * L}
    elif operation in ("moisture", "storage"):
        inputs = {"state.water_m": f8 * L * N}
    else:
        inputs = {"state.water_m": f8 * L * N, "potential_evaporation_m_s": f8 * N}
        if operation == "step":
            inputs["added_water_m"] = f8 * L * N
        if with_share:
            inputs["evaporation_share"] = f8 * N
    device = t["outputs"] + t["scratch"] + reserve
    host = t["host"] + PY_ALLOWANCE_BYTES + host_reserve
    terms = {"outputs": t["outputs"], "scratch": t["scratch"], "host_payload": t["host"], "python_allowance": PY_ALLOWANCE_BYTES, "n_blocks": t["n_blocks"], "factor_arrays": t["factor_arrays"],
             "static_shared_bytes_per_block": t["shared"], "backend": "cuda", "no_rows_array": True, "no_cub_or_cupy_reduction_workspace": True, "no_permanent_LN_derived_maps": True, "no_history": True,
             "excluded": "NVRTC/module load, CUDA context, allocator pool policy and block rounding, interpreter; estimates are payload models, not reservations"}
    scope = ("modelled bytes of this call's own arrays: DEVICE main requirement in required_bytes (outputs + released scratch + reserve) and the separate HOST requirement in host_required_bytes; caller state/forcing "
             "and the owner's retained maps are excluded; allocator rounding, compiler/context and whole-process memory are excluded; an availability reading is not a reservation")
    return SpatialOperationMemory(operation, L, N, t["outputs"], t["scratch"], reserve, device, inputs, terms, scope, device, host, host_reserve)


def _admit_cuda_operation(cp, column: "SpatialSoilColumn", est: SpatialOperationMemory, budget: Any, reader: Callable[[], int] | None, host_budget: Any, host_reader: Callable[[], int] | None) -> None:
    """Device and host admission of ONE CUDA operation, arithmetic only, BEFORE any allocation, conversion, copy or launch. `host_budget` None = no separate host quota (the host reading still applies)."""
    b = _count(budget, "memory_budget_bytes")
    hb = None if host_budget is None else _count(host_budget, "host_memory_budget_bytes")
    for name, fn in (("available_bytes_fn", reader), ("host_available_bytes_fn", host_reader)):
        if fn is not None and not callable(fn):
            raise SpatialLayerError(f"{name} must be callable or None")
    if est.device_required_bytes > b:
        raise SpatialLayerMemoryError(f"spatial {est.operation} on {est.n_layers} layers x {est.n_cells} cells needs an estimated {est.device_required_bytes} B on the device (reserve {est.reserve_bytes} B included) but "
                                      f"the caller's declared memory_budget_bytes is {b}; nothing was allocated. Scope: {est.scope}")
    if hb is not None and est.host_required_bytes > hb:
        raise SpatialLayerMemoryError(f"spatial {est.operation} needs an estimated {est.host_required_bytes} B on the host (reserve {est.host_reserve_bytes} B included) but the caller's declared "
                                      f"host_memory_budget_bytes is {hb}; nothing was allocated. Scope: {est.scope}")
    rc = _cuda_module()
    for side, need, fn in (("device", est.device_required_bytes, reader if reader is not None else rc.default_device_available_bytes_fn(cp)),
                           ("host", est.host_required_bytes, host_reader if host_reader is not None else default_host_available_bytes)):
        available = fn()
        if isinstance(available, (bool, np.bool_)) or not isinstance(available, (int, np.integer)) or int(available) < 0:
            raise SpatialLayerError(f"the {side} available-bytes reader must return a non-Boolean integer >= 0, got {available!r}")
        if need > int(available):
            raise SpatialLayerMemoryError(f"spatial {est.operation} on {est.n_layers} layers x {est.n_cells} cells needs an estimated {need} B on the {side} but only {int(available)} B are reported "
                                          f"available; nothing was allocated. Scope: {est.scope}")


def _admit_operation(est: SpatialOperationMemory, budget: Any, reader: Callable[[], int] | None) -> None:
    b = _count(budget, "memory_budget_bytes")
    if reader is not None and not callable(reader):
        raise SpatialLayerError("available_bytes_fn must be callable or None")
    if est.required_bytes > b:
        raise SpatialLayerMemoryError(f"spatial {est.operation} on {est.n_layers} layers x {est.n_cells} cells needs an estimated {est.required_bytes} B (reserve {est.reserve_bytes} B included) but the "
                                      f"caller's declared memory_budget_bytes is {b}; nothing was allocated. Scope: {est.scope}")
    available = (default_host_available_bytes if reader is None else reader)()
    if isinstance(available, (bool, np.bool_)) or not isinstance(available, (int, np.integer)) or int(available) < 0:
        raise SpatialLayerError(f"available_bytes_fn must return a non-Boolean integer >= 0, got {available!r}")
    if est.required_bytes > int(available):
        raise SpatialLayerMemoryError(f"spatial {est.operation} on {est.n_layers} layers x {est.n_cells} cells needs an estimated {est.required_bytes} B (reserve {est.reserve_bytes} B included) but "
                                      f"only {int(available)} B are reported available; nothing was allocated. Scope: {est.scope}")


def _check_spatial_column(column: Any) -> None:
    if type(column) is not SpatialSoilColumn:
        raise SpatialLayerError(f"column must be a SpatialSoilColumn, got {type(column).__name__} (a legacy SoilColumn is not accepted here)")
    if column.backend not in EVOLUTION_BACKENDS:
        raise SpatialLayerBackendUnavailableError(f"spatial water evolution is implemented for the backends {EVOLUTION_BACKENDS} only; this column was prepared for {column.backend!r}. "
                                                  "There is no fallback and no copy is made")


def _device_cp(column: SpatialSoilColumn):
    """CuPy for a CUDA owner after the CURRENT-device check (typed refusals; nothing is allocated, copied or launched here); `None` for a CPU owner."""
    if column.backend != "cuda":
        return None
    cp = _require_cuda()
    current = _cuda_module().current_device_id(cp)
    if current != column.device_id:
        raise SpatialLayerError(f"the CUDA owner is bound to device {column.device_id} but the current device is {current}; select the owner's device (no implicit transfer)")
    return cp


def _check_array(name: str, array: Any, shape: tuple, column: Any = None) -> None:
    if column is not None and column.backend == "cuda":
        cp = _device_cp(column)
        if type(array) is not cp.ndarray:
            raise SpatialLayerError(f"{name} must be exactly a cupy.ndarray on the owner's device (no NumPy array, subclass or implicit transfer), got {type(array).__module__}.{type(array).__name__}")
        if array.dtype != np.float64:
            raise SpatialLayerError(f"{name} must be float64, got {array.dtype}")
        if tuple(array.shape) != tuple(shape):
            raise SpatialLayerError(f"{name} must have shape {tuple(shape)}, got {tuple(array.shape)}")
        if not array.flags.c_contiguous:
            raise SpatialLayerError(f"{name} must be C-contiguous")
        if int(array.device.id) != column.device_id:
            raise SpatialLayerError(f"{name} lives on device {int(array.device.id)} but the owner is bound to device {column.device_id}; no implicit transfer")
        return
    if type(array) is not np.ndarray:
        raise SpatialLayerError(f"{name} must be exactly a numpy.ndarray (no subclass, no masked array, no implicit transfer), got {type(array).__module__}.{type(array).__name__}")
    if array.dtype != np.float64:
        raise SpatialLayerError(f"{name} must be float64, got {array.dtype}")
    if tuple(array.shape) != tuple(shape):
        raise SpatialLayerError(f"{name} must have shape {tuple(shape)}, got {tuple(array.shape)}")
    if not array.flags.c_contiguous:
        raise SpatialLayerError(f"{name} must be C-contiguous")


def _check_spatial_state(column: SpatialSoilColumn, state: Any) -> float:
    if type(state) is not SpatialSoilState:
        raise SpatialLayerError(f"state must be a SpatialSoilState, got {type(state).__name__} (a legacy SoilColumnState is not accepted here)")
    if state.column_sha256 != column.sha256 or (state.n_layers, state.n_cells) != (column.n_layers, column.n_cells):
        raise SpatialLayerError("the state belongs to a different spatial parameter set/mask (identity mismatch); it cannot be used with this column")
    want_ns, want_dev = ("cupy", column.device_id) if column.backend == "cuda" else ("numpy", None)
    if state.namespace != want_ns or state.device_id != want_dev:
        raise SpatialLayerError(f"the state lives in namespace {state.namespace!r} (device {state.device_id}) but this {column.backend!r} owner needs {want_ns!r} (device {want_dev}); states are never converted between backends")
    _check_array("state.water_m", state.water_m, (column.n_layers, column.n_cells), column)
    return _real(state.t_s, "state.t_s")


def _new_state(column: SpatialSoilColumn, water: Any, t_s: float) -> SpatialSoilState:
    if column.backend == "cuda":  # CuPy has no read-only flag: the state is logically immutable only
        return SpatialSoilState(water, float(t_s), column.sha256, column.n_layers, column.n_cells, STATE_COMPONENT_TAG_CUDA, "cupy", column.device_id)
    water.flags.writeable = False
    return SpatialSoilState(water, float(t_s), column.sha256, column.n_layers, column.n_cells)


def _quotas(column: SpatialSoilColumn, host_memory_budget_bytes: Any, host_reserve_bytes: Any, host_available_bytes_fn: Any) -> tuple:
    """The separate HOST quota/reserve/reading exist only for a CUDA owner (as in `prepare_spatial_column`); a CPU owner refuses them."""
    if column.backend != "cuda":
        if host_memory_budget_bytes is not None or host_reserve_bytes or host_available_bytes_fn is not None:
            raise SpatialLayerError("host_memory_budget_bytes / host_reserve_bytes / host_available_bytes_fn apply only to a cuda owner")
        return None, 0, None
    return host_memory_budget_bytes, _count(host_reserve_bytes, "host_reserve_bytes"), host_available_bytes_fn


@contextlib.contextmanager
def _device_errors(what: str):
    """Typed device failures: an out-of-memory error during an operation becomes `SpatialLayerMemoryError`, a CUDA runtime/driver/launch error (surfacing at the synchronising packet read at the
    latest) `SpatialLayerBackendUnavailableError`; the freshly allocated outputs are dropped with the frame and no state is published. Recovery of a poisoned CUDA context is NOT claimed."""
    try:
        yield
    except (SpatialLayerError, ss.SubsurfaceError, SpatialLayerBackendUnavailableError, ss.SubsurfaceBackendUnavailableError, ss.SubsurfaceMemoryError):
        raise
    except MemoryError as exc:
        raise SpatialLayerMemoryError(f"the device allocation of the CUDA {what} failed ({type(exc).__name__}: {exc}); nothing was published and the accepted state is unchanged") from exc
    except Exception as exc:  # noqa: BLE001 - CuPy/CUDA runtime errors are not importable without CuPy
        if type(exc).__name__.startswith(("CUDA", "Cuda", "Runtime", "Driver")) or "cuda" in type(exc).__module__.lower():
            raise SpatialLayerBackendUnavailableError(f"the CUDA {what} failed ({type(exc).__name__}: {exc}); nothing was published and the accepted state is unchanged") from exc
        raise


def _factor_arrays(column: SpatialSoilColumn) -> int:
    from maple_syrup import subsurface_cuda_spatial as cs

    hoisted = column.n_layers > 1 and column.estimate.retained["interface_conductance_down"]["kind"] == LAYOUT_SHARED
    return cs.factor_arrays(column.n_layers, hoisted, column.diffusion_mode != "constant")


def _admit_for(column: SpatialSoilColumn, operation: str, *, share: bool = False, budget: Any, reserve: Any, reader: Any, host_budget: Any, host_reserve: Any, host_reader: Any) -> Any:
    """Estimate + admission of ONE operation for either backend; returns CuPy (CUDA owner) or None. All metadata checks of the caller precede this; nothing is allocated before it returns."""
    hb, hr, hrd = _quotas(column, host_budget, host_reserve, host_reader)
    if column.backend == "cuda":
        cp = _device_cp(column)
        est = estimate_spatial_operation_memory(operation, column.n_layers, column.n_cells, with_evaporation_share=share, reserve_bytes=reserve, backend="cuda", host_reserve_bytes=hr,
                                                hoisted_factor_arrays=_factor_arrays(column) if operation in ("step", "step_with_recharge") else 0)
        _admit_cuda_operation(cp, column, est, budget, reader, hb, hrd)
        return cp
    est = estimate_spatial_operation_memory(operation, column.n_layers, column.n_cells, with_evaporation_share=share, reserve_bytes=reserve, backend=column.backend)
    _admit_operation(est, budget, reader)
    return None


def spatial_column_state(column: SpatialSoilColumn, water_m: Any, t_s: float = 0.0, *, memory_budget_bytes: Any, reserve_bytes: Any = 0,
                         available_bytes_fn: Callable[[], int] | None = None, host_memory_budget_bytes: Any = None, host_reserve_bytes: Any = 0, host_available_bytes_fn: Callable[[], int] | None = None) -> SpatialSoilState:
    """A state owning a COPY of the caller's `(L, N)` float64 water (m): no interpolation, no remapping, no residual added. Metadata, time and the memory admission (one state copy) precede the
    allocation; the VALUES (finite, `>= 0`, `<= saturation`) are checked by the next step's staged validation, exactly as for the common column. CUDA owner: `water_m` must be a `cupy.ndarray` on the
    owner's device (current device checked first), the copy is device-to-device and may complete asynchronously on the current stream (the caller owns later cross-stream ordering); the three `host_*`
    arguments are the separate host quota/reserve/reading and are refused for a CPU owner; `memory_budget_bytes`/`reserve_bytes`/`available_bytes_fn` are then the DEVICE quota/reserve/reading."""
    _check_spatial_column(column)
    _check_array("water_m", water_m, (column.n_layers, column.n_cells), column)
    t = _real(t_s, "t_s")
    cp = _admit_for(column, "state", budget=memory_budget_bytes, reserve=reserve_bytes, reader=available_bytes_fn, host_budget=host_memory_budget_bytes, host_reserve=host_reserve_bytes, host_reader=host_available_bytes_fn)
    if cp is not None:
        with _device_errors("state copy"):
            copy = cp.array(water_m, dtype=np.float64, copy=True, order="C")
        return _new_state(column, copy, t)
    return _new_state(column, np.array(water_m, dtype=np.float64, copy=True, order="C"), t)


def _preflight_theta_by_layer(theta: Any, n_layers: int) -> None:
    """Bounded METADATA check of `theta_by_layer` BEFORE any conversion or admission: a real scalar, a 0-d array, a 1-D `numpy.ndarray` of exactly `L` real elements, or a list/tuple of exactly `L`
    real scalars. Anything else (objects with `__array__`, generators, nested or wrong-length sequences, Booleans, strings) is refused WITHOUT being converted, so no unbounded or
    caller-controlled allocation can precede the memory admission. The check reads no more than `L` elements."""
    if isinstance(theta, (bool, np.bool_)):
        raise ss.SubsurfaceError("theta_by_layer must be numeric, got a Boolean")
    if _is_real_scalar(theta):
        return
    if type(theta) is np.ndarray:
        if theta.dtype.kind not in "iuf":
            raise ss.SubsurfaceError(f"theta_by_layer must be real numbers, got dtype {theta.dtype}")
        if theta.ndim == 0 or (theta.ndim == 1 and theta.shape[0] == n_layers):
            return
        raise ss.SubsurfaceError(f"theta_by_layer must be a scalar or have shape ({n_layers},), got shape {tuple(theta.shape)}")
    if type(theta) in (list, tuple):
        if len(theta) != n_layers:
            raise ss.SubsurfaceError(f"theta_by_layer must be a scalar or have {n_layers} elements, got {len(theta)}")
        for i, v in enumerate(theta):
            if not _is_real_scalar(v):
                raise ss.SubsurfaceError(f"theta_by_layer[{i}] must be a real number, got {type(v).__name__}")
        return
    raise ss.SubsurfaceError(f"theta_by_layer must be a real scalar, a numpy.ndarray or a list/tuple of {n_layers} real numbers; {type(theta).__name__} is not converted")


def spatial_uniform_state(column: SpatialSoilColumn, theta_by_layer: Any, t_s: float = 0.0, *, memory_budget_bytes: Any, reserve_bytes: Any = 0,
                          available_bytes_fn: Callable[[], int] | None = None, host_memory_budget_bytes: Any = None, host_reserve_bytes: Any = 0, host_available_bytes_fn: Callable[[], int] | None = None) -> SpatialSoilState:
    """Antecedent moisture from explicit per-layer volumetric theta (length `L`, the same in every cell): `water[k, c] = theta[k] * dz[k, c]`, each theta required within `[0, saturation theta]` of
    EVERY cell (`0` allowed; nothing is added for the residual). `theta_by_layer` is a real scalar, a 0-d/1-D `numpy.ndarray` of `L` reals or a list/tuple of `L` reals: its METADATA is checked before
    anything is converted, the memory admission counts the `(L, N)` result, the `(N,)` comparison rows AND the L-sized conversion lifetimes (`3 * 8 * L` bytes), and only then is the accepted
    `subsurface._layer_array` conversion (finite values) made. CUDA owner: the theta stays a HOST input (admitted explicitly: `24 L` conversion lifetimes + the 32-byte diagnostic read on the host quota,
    the `8 L` device copy + result + diagnostic on the device quota); ONE counted host-to-device copy of `8 L` bytes and ONE counted 32-byte read; the threshold is checked in every cell on the device and a
    violation discards the fresh water and raises the CPU message."""
    _check_spatial_column(column)
    _preflight_theta_by_layer(theta_by_layer, column.n_layers)
    t = _real(t_s, "t_s")
    cp = _admit_for(column, "uniform_state", budget=memory_budget_bytes, reserve=reserve_bytes, reader=available_bytes_fn, host_budget=host_memory_budget_bytes, host_reserve=host_reserve_bytes, host_reader=host_available_bytes_fn)
    theta = ss._layer_array(theta_by_layer, column.n_layers, "theta_by_layer")  # only now: bounded metadata passed and the L-sized lifetimes are admitted
    if cp is not None:
        from maple_syrup import subsurface_cuda_spatial as cs

        with _device_errors("uniform initialisation"):
            water, bad = cs.uniform_water(cp, column, theta)
        if bad is not None:
            raise ss.SubsurfaceError(f"theta_by_layer[{bad}] = {theta[bad]!r} must lie within [0, theta_saturation] in every cell")
        return _new_state(column, water, t)
    params = _SpatialStepParams(column)
    water = np.empty((column.n_layers, column.n_cells), dtype=np.float64)
    ts, kts = column._maps["theta_saturation"], column.estimate.retained["theta_saturation"]["kind"]
    for k in range(column.n_layers):
        if np.any(theta[k] < 0.0) or np.any(theta[k] > _retained_row(ts, kts, k)):
            raise ss.SubsurfaceError(f"theta_by_layer[{k}] = {theta[k]!r} must lie within [0, theta_saturation] in every cell")
    for k in range(column.n_layers):
        np.multiply(theta[k], params.dz[k], out=water[k])  # the same IEEE product as the common `theta * thickness`
    return _new_state(column, water, t)


def spatial_moisture(column: SpatialSoilColumn, state: SpatialSoilState, *, memory_budget_bytes: Any, reserve_bytes: Any = 0,
                     available_bytes_fn: Callable[[], int] | None = None, host_memory_budget_bytes: Any = None, host_reserve_bytes: Any = 0, host_available_bytes_fn: Callable[[], int] | None = None) -> Any:
    """Volumetric moisture `theta = water / dz`: a fresh `(L, N)` float64 array (admitted before allocation); a `cupy.ndarray` for a CUDA owner (may complete asynchronously on the current stream)."""
    _check_spatial_column(column)
    _check_spatial_state(column, state)
    cp = _admit_for(column, "moisture", budget=memory_budget_bytes, reserve=reserve_bytes, reader=available_bytes_fn, host_budget=host_memory_budget_bytes, host_reserve=host_reserve_bytes, host_reader=host_available_bytes_fn)
    if cp is not None:
        from maple_syrup import subsurface_cuda_spatial as cs

        with _device_errors("moisture query"):
            return cs.moisture(cp, column, state.water_m)
    params = _SpatialStepParams(column)
    out = np.empty((column.n_layers, column.n_cells), dtype=np.float64)
    for k in range(column.n_layers):
        np.divide(state.water_m[k], params.dz[k], out=out[k])
    return out


def spatial_storage_m3(column: SpatialSoilColumn, state: SpatialSoilState, *, memory_budget_bytes: Any, reserve_bytes: Any = 0,
                       available_bytes_fn: Callable[[], int] | None = None, host_memory_budget_bytes: Any = None, host_reserve_bytes: Any = 0, host_available_bytes_fn: Callable[[], int] | None = None) -> float:
    """Total soil water over ACTIVE cells (m3): the layer-sequential per-cell sum, inactive cells zeroed (never clipped or counted), the cell reduction, times the area (CPU: one `(N,)` accumulator and one
    mask; CUDA: block partials + a one-word packet, summed in a fixed tree order -- roundoff-level different from NumPy's pairwise order; admitted first)."""
    _check_spatial_column(column)
    _check_spatial_state(column, state)
    cp = _admit_for(column, "storage", budget=memory_budget_bytes, reserve=reserve_bytes, reader=available_bytes_fn, host_budget=host_memory_budget_bytes, host_reserve=host_reserve_bytes, host_reader=host_available_bytes_fn)
    if cp is not None:
        from maple_syrup import subsurface_cuda_spatial as cs

        with _device_errors("storage query"), np.errstate(all="ignore"):
            depth = cs.storage_depth(cp, column, state.water_m)
            volume = depth * column.cell_area_m2
    else:
        total = np.zeros(column.n_cells, dtype=np.float64)
        for k in range(column.n_layers):
            np.add(total, state.water_m[k], out=total)
        np.copyto(total, 0.0, where=~column._mask)
        with np.errstate(all="ignore"):
            depth = float(np.sum(total))
            volume = depth * column.cell_area_m2
    if not (math.isfinite(depth) and math.isfinite(volume)):  # a finite state can still have an unrepresentable global volume (huge area); never publish a non-finite inventory
        raise SpatialLayerError(f"the global soil-water volume is not representable in FP64 (active-cell depth sum {depth!r} m x cell area {column.cell_area_m2!r} m2 = {volume!r}) "
                                "or the state holds non-finite water; the query is refused")
    return volume


_KIND_CODE = {LAYOUT_SCALAR: 0, LAYOUT_SHARED: 1, LAYOUT_CELL: 2}


def _compiled_arrays(column: SpatialSoilColumn, state: SpatialSoilState, added: Any, pet: np.ndarray, dt: float, share: Any, pulse: float | None) -> Any:
    """The "numba" body of one step: the owner's retained RAW maps as flat read-only VIEWS (`reshape(-1)` of a C-contiguous map copies nothing), their layout codes, and the owner's mask as a uint8 view
    (no copy): the compiled body derives the metre bounds and the exchange factors itself (`subsurface_numba.spatial_launch`)."""
    from maple_syrup import subsurface_numba as nb
    from maple_syrup.conservation import volume_roundoff_bound_m3

    maps = {name: arr.reshape(-1) for name, arr in column._maps.items()}
    kinds = {name: _KIND_CODE[v["kind"]] for name, v in column.estimate.retained.items()}
    return nb.spatial_launch(state.water_m, added, pet, column._mask.view(np.uint8), dt, maps=maps, kinds=kinds, tolerance_coefficient=volume_roundoff_bound_m3(ss.LAYER_OPERATION_TERMS, 1.0),
                             residual_theta_stress_factor=(column.theta_evaporation_stress, column.evaporation_factor), gravity_code=1 if column.gravity_drainage else 0,
                             directional=1 if column.diffusion_mode != "constant" else 0, bottom_code=0 if column.bottom_drainage == "closed" else 1, share=share, recharge=pulse)


def _evolve(column: SpatialSoilColumn, state: SpatialSoilState, added: Any, pet: Any, dt: float, share: Any, pulse: float | None) -> Any:
    if column.backend == "numba":
        return _compiled_arrays(column, state, added, pet, dt, share, pulse)
    return ss._reference_step_body(_SpatialStepParams(column), column._mask, state.water_m, added, pet, dt, share, pulse)


def _evolve_cuda(cp, column: SpatialSoilColumn, state: SpatialSoilState, added: Any, pet: Any, dt: float, share: Any, pulse: float | None) -> tuple:
    """The CUDA step body: `(StepArrays with rows=None, the reconstructed nine-value float packet)` (`subsurface_cuda_spatial.run_step`)."""
    from maple_syrup import subsurface_cuda_spatial as cs

    try:
        with _device_errors("step"):
            return cs.run_step(cp, column, state.water_m, added, pet, share, dt, pulse)
    except ss.SubsurfaceBackendUnavailableError as exc:
        raise SpatialLayerBackendUnavailableError(str(exc)) from exc


def _finish(column: SpatialSoilColumn, arrays: Any, dt: float, t1: float, recharge: bool, packet: Any = None) -> SpatialStep:
    if packet is None:
        packet = ss._read_packet(np, arrays)  # the existing ONE batched reduction packet (CPU owners; a CUDA owner passes its own explicit packet)
    budget = ss._resolve(column, packet, recharge=recharge)  # the existing conservation bound and typed errors (SubsurfaceStepError / SubsurfaceFitFailure)
    return SpatialStep(_new_state(column, arrays.water_m, t1), dt, arrays.surface_return_m, arrays.drainage_m, arrays.evaporation_m, arrays.evaporation_unmet_m, arrays.percolation_m,
                       arrays.diffusion_m, budget)


def spatial_step(column: SpatialSoilColumn, state: SpatialSoilState, dt_s: Any, *, added_water_m: Any, potential_evaporation_m_s: Any, evaporation_share: Any = None,
                 memory_budget_bytes: Any, reserve_bytes: Any = 0, available_bytes_fn: Callable[[], int] | None = None,
                 host_memory_budget_bytes: Any = None, host_reserve_bytes: Any = 0, host_available_bytes_fn: Callable[[], int] | None = None) -> SpatialStep:
    """One pure step of the prepared explicit spatial parameters on the ONE existing reference body: the same stage order, flags, bounds and error types as `subsurface.step` (addition with
    saturation return -> field-capacity gravity cascade with open/closed bottom -> layer-0 evaporation (optionally the post-cap share law) -> sequential pair diffusion in the column's mode -> output
    checks and the global closure). Units: `dt_s` s, `added_water_m` `(L, N)` m of water actually directed into each layer, `potential_evaporation_m_s` `(N,)` m/s (0 on inactive cells),
    `evaporation_share` optional `(N,)` in `[0, 1]` on every cell. Backends: "numpy" (the NumPy reference body) and "numba" (the compiled body; `expm1` may differ from NumPy by an ulp, declared bound rtol 2e-12 /
    atol 1e-14, bitwise where the diffusivity is 0) and "cuda" (S2-A.4.2 trial: the shared CUDA body; arrays are `cupy.ndarray` on the owner's device, no implicit transfer, no fallback between backends; the step
    keeps no `rows` array, one 72-byte packet is read; device quota = `memory_budget_bytes`, separate `host_*` quota). A refused step publishes nothing. Metadata, time and the memory admission precede any allocation; values are validated in the staged flags;
    a refused step publishes nothing and no input (state, forcing, retained maps) is written. The clock advances by exactly `dt_s` (`dt_s = 0` is the accepted zero-time case); there is no
    residual water, no remapping, no transpiration."""
    _check_spatial_column(column)
    t0 = _check_spatial_state(column, state)
    dt = ss._check_dt(dt_s)
    t1 = t0 + dt
    if not math.isfinite(t1):
        raise ss.SubsurfaceError(f"the candidate time t + dt = {t0!r} + {dt!r} is not finite; the step is refused before any numerical work (state untouched)")
    shape = (column.n_layers, column.n_cells)
    _check_array("added_water_m", added_water_m, shape, column)
    _check_array("potential_evaporation_m_s", potential_evaporation_m_s, (column.n_cells,), column)
    if evaporation_share is not None:
        _check_array("evaporation_share", evaporation_share, (column.n_cells,), column)
    cp = _admit_for(column, "step", share=evaporation_share is not None, budget=memory_budget_bytes, reserve=reserve_bytes, reader=available_bytes_fn, host_budget=host_memory_budget_bytes,
                host_reserve=host_reserve_bytes, host_reader=host_available_bytes_fn)
    if cp is not None:
        arrays, packet = _evolve_cuda(cp, column, state, added_water_m, potential_evaporation_m_s, dt, evaporation_share, None)
        return _finish(column, arrays, dt, t1, False, packet)
    arrays = _evolve(column, state, added_water_m, potential_evaporation_m_s, dt, evaporation_share, None)
    return _finish(column, arrays, dt, t1, False)


def spatial_step_with_recharge(column: SpatialSoilColumn, state: SpatialSoilState, dt_s: Any, *, recharge_depth_m: Any, potential_evaporation_m_s: Any, evaporation_share: Any = None,
                               memory_budget_bytes: Any, reserve_bytes: Any = 0, available_bytes_fn: Callable[[], int] | None = None,
                               host_memory_budget_bytes: Any = None, host_reserve_bytes: Any = 0, host_available_bytes_fn: Callable[[], int] | None = None) -> SpatialStep:
    """The Phase 1.5 pulse-aware daily step on the same body: ONE finite non-negative host depth `recharge_depth_m` [m] is added uniformly to the top layer of every ACTIVE cell with no saturation
    return, the existing gravity cascade runs once, then the fit test of every active cell (`SubsurfaceFitFailure`, flag 17, when any cell exceeds saturation + tolerance; nothing is clipped or
    partly evaporated), then the historical evaporation and diffusion. `dt_s > 0` is required and must advance the clock (`t + dt > t`); no (L, N) forcing tensor exists."""
    _check_spatial_column(column)
    t0 = _check_spatial_state(column, state)
    dt = ss._check_dt(dt_s)
    if not dt > 0.0:
        raise ss.SubsurfaceError("the pulse-aware daily step requires dt_s > 0 (no zero-time rain step)")
    pulse = ss._real(recharge_depth_m, "recharge_depth_m")
    if pulse < 0.0:
        raise ss.SubsurfaceError(f"recharge_depth_m must be >= 0, got {pulse!r}")
    t1 = t0 + dt
    if not math.isfinite(t1):
        raise ss.SubsurfaceError(f"the candidate time t + dt = {t0!r} + {dt!r} is not finite; the step is refused before any numerical work (state untouched)")
    if not t1 > t0:
        raise ss.SubsurfaceError(f"the positive interval dt = {dt!r} s does not advance the native clock at t = {t0!r} s (t + dt == t); the pulse-aware step is refused before any work")
    _check_array("potential_evaporation_m_s", potential_evaporation_m_s, (column.n_cells,), column)
    if evaporation_share is not None:
        _check_array("evaporation_share", evaporation_share, (column.n_cells,), column)
    cp = _admit_for(column, "step_with_recharge", share=evaporation_share is not None, budget=memory_budget_bytes, reserve=reserve_bytes, reader=available_bytes_fn, host_budget=host_memory_budget_bytes,
                host_reserve=host_reserve_bytes, host_reader=host_available_bytes_fn)
    if cp is not None:
        arrays, packet = _evolve_cuda(cp, column, state, None, potential_evaporation_m_s, dt, evaporation_share, pulse)
        return _finish(column, arrays, dt, t1, True, packet)
    arrays = _evolve(column, state, None, potential_evaporation_m_s, dt, evaporation_share, pulse)
    return _finish(column, arrays, dt, t1, True)
