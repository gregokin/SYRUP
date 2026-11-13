"""Phase 2 S1 component: conservative interval remapping of piecewise-uniform water along explicit vertical intervals.

COMPONENT / TRIAL ONLY. This is a pure mathematical utility. It does not choose an erosion/deposition order, a layer frame, groundwater, the wetting of
transported sediment, a capillary rule, a saturation redistribution or a residual-water rule; it adopts no accepted state, reads no clock or ledger, binds no
parent and carries no restart or qualification claim. A later caller gathers the active changed columns, supplies the geometry and the incoming water, and
composes ordered native exchanges itself.

Model (one definition for all backends)
---------------------------------------
Per selected column `c` (the same `L >= 1` intervals in every column, `C >= 0` columns). Depth is one coordinate per column, increasing downward, in the SAME
reference for the old and the new edges (negative values are allowed; the new top may differ from the old top, the new bottom from the old bottom, the two
extents may even be disjoint).

    old edges  eo[0..L]  strictly increasing, finite, every width eo[i+1]-eo[i] finite and > 0
    new edges  en[0..L]  the same
    water      W[i]      m of water in old interval i; the volumetric fraction is theta_i = W_i / w_i with w_i = eo[i+1]-eo[i], required 0 <= W_i <= w_i
                         (a volumetric fraction in [0,1]; NO hydraulic residual or any other floor is applied or implied)
    upper_theta, lower_theta   explicit, finite, in [0,1], (C,) arrays, no defaults: the volumetric fraction of the material that enters the profile
                         ABOVE the old top and BELOW the old bottom respectively

New interval j receives the length-weighted water of the pieces that cover it:

    N_j = sum_i theta_i * |old_i  ∩ new_j|  +  upper_theta * |new_j ∩ (-inf, eo[0])|  +  lower_theta * |new_j ∩ (eo[L], +inf)|,
    |old_i ∩ new_j| = max(0, min(eo[i+1], en[j+1]) - max(eo[i], en[j]))

Four separate per-column quantities are returned and are NEVER routed, removed or called loss/drainage by this module:

    removed_above  old water located above the NEW top   (z <  en[0]),
    removed_below  old water located below the NEW bottom (z >= en[L]),
    added_above    upper_theta * length of the new span above the OLD top,
    added_below    lower_theta * length of the new span below the OLD bottom.

Closure per column (identity, up to the FP64 bound):  sum_j N_j - sum_i W_i = added_above + added_below - removed_above - removed_below.
No saturation/property clipping, no redistribution, no dt, no clock and no ledger exist here. Zero geometry change (en == eo bitwise) returns every water value
bit for bit and every flux exactly 0: the contribution of old interval i to new interval j is evaluated as `W_i * (overlap / w_i)` (not `(W_i/w_i)*overlap`), and the
ratio is exactly 1 when the overlap equals the width. Equal removal and deposition (zero NET elevation change) are NOT a no-change case here: the caller must
state the geometry, and different material may enter; this module never classifies it.

Arithmetic order (the definition; the compiled kernels repeat it term for term, no fastmath, no FMA contraction, no reordering)
------------------------------------------------------------------------------------------------------------------------------
    N_j      = ((0.0 + sum_{i=0..L-1} W_i * f_ij) + above_j) + below_j,  f_ij = max(0, min(eo[i+1],en[j+1]) - max(eo[i],en[j])) / w_i,
    above_j  = upper_theta * max(0, min(en[j+1], eo[0]) - en[j]),   below_j = lower_theta * max(0, en[j+1] - max(en[j], eo[L])),
    removed_above = 0.0 + sum_i W_i * (max(0, min(eo[i+1], en[0]) - eo[i]) / w_i),   removed_below = 0.0 + sum_i W_i * (max(0, eo[i+1] - max(eo[i], en[L])) / w_i),
    added_above   = upper_theta * max(0, min(eo[0], en[L]) - en[0]),                  added_below   = lower_theta * max(0, en[L] - max(eo[L], en[0])).
Sums run over ascending index. The NumPy reference vectorises over the C columns with a short Python loop over layers (no `L x L x C` allocation, no per-column
Python object); the Numba and CUDA kernels run one column per iteration / thread.

Checks (metadata first, then values, nothing published on failure)
------------------------------------------------------------------
Before anything is allocated: backend name, exact array types and namespace (NumPy `ndarray` for "numpy"/"numba", CuPy `ndarray` on the CURRENT device for "cuda"; no
subclass, no masked array, no implicit transfer), dtype float64, C-contiguity, shapes `(L+1,C)`, `(L+1,C)`, `(L,C)`, `(C,)`, `(C,)` with `L >= 1`, `C >= 0`, the scalar
`cell_area_m2` (finite positive real, not a Boolean) and the integer budget/reserve. Then the memory admission (below). After the computation and BEFORE the result is built:
per-column flags (the error reports the lowest bit of the FIRST flagged column) 1 old edges not finite/increasing/representable, 2 new edges, 4 water not finite or outside [0, width], 8 theta not finite or outside
[0,1], 16 an output (new water or a flux) not finite or a new interval holding more than its width plus its derived bound, 32 per-column closure above its derived bound; and
the global closure. Any failure raises a typed error; the fresh outputs are discarded and the inputs were never written.

Roundoff bounds (MAPLE's existing coefficient, no new epsilon)
---------------------------------------------------------------
`coef(n) = maple_syrup.conservation.volume_roundoff_bound_m3(n, 1.0)` (MAPLE's `summation_error_bound_kg(n, 1.0)`).
 * Column closure: `n_col = 2 L^2 + 9 L + 8` counts every product and sum that can enter the identity (per interval `L` products + `L+2` adds + 2 products = `2L+4`, times
   `L` intervals; `L` terms of the old total; `2 L` products-and-adds in each of the two removed sums; `4` added/removed terms; `4` final combinations), scale
   `S_c = sum_i w_i + sum_j wn_j` (every operand, being a product of a fraction in [0,1] and a length or a water amount at most a width, is at most `S_c`), bound `coef(n_col) * S_c`.
 * Output sanity: a new interval may exceed its width by at most `coef(2 L + 4) * wn_j` (the terms of its own sum); it is NEVER clipped and the returned value is the computed one.
 * Global closure (m3, area `A`): `n_glob = C n_col + 8`, scale `A * sum_c S_c`, applied to the residual of the totals formed from the per-column totals.

Memory (modelled, not measured resident memory; see `estimate_remap_memory`)
----------------------------------------------------------------------------
The caller owns and keeps its five input arrays (reported as `input_bytes_excluded`, never copied). This call allocates: the retained outputs `(L+4)*8*C` bytes, per-column
scratch (a `(5, C)` float64 block of residual / bound / old total / new total / scale and, on the CPU backends, a `C` byte flag array), plus for "numpy" a declared number of live `(C,)` work arrays
(`REFERENCE_TEMP_F64`, `REFERENCE_TEMP_BOOL`, checked against `tracemalloc` by the tests) and for "cuda" a 5-word device diagnostic (no flag/index/mask array) and a small host packet; CPU backends add a declared small allowance for Python objects and chunk-sized host temporaries. No
`L x L x C` array, no `(L, C, K)` array, no per-column Python object and no whole-bed array exists. The estimate plus the caller's reserve must fit BOTH the caller's declared
`memory_budget_bytes` and the available-bytes reading (`available_bytes_fn`, default the host `sysconf` reading, or the device free memory plus pool for "cuda"); otherwise
`SoilRemapMemoryError` is raised before any allocation. A reading is not a reservation, and this is not a whole-process guarantee: interpreter, JIT/NVRTC and driver costs
are excluded.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

__all__ = [
    "BACKENDS",
    "CLOSURE_TERMS_FORMULA",
    "FLAG_CLOSURE",
    "FLAG_NEW_EDGES",
    "FLAG_OLD_EDGES",
    "FLAG_OUTPUT",
    "FLAG_THETA",
    "FLAG_WATER",
    "LABEL",
    "RemapMemoryEstimate",
    "RemapResult",
    "SoilRemapBackendUnavailableError",
    "SoilRemapClosureError",
    "SoilRemapError",
    "SoilRemapMemoryError",
    "SoilRemapValidationError",
    "closure_terms",
    "cuda_block_count",
    "default_host_available_bytes",
    "estimate_remap_memory",
    "reference_remap",
    "remap_intervals",
]

BACKENDS = ("numpy", "numba", "cuda")
LABEL = "component_trial_only"
FLAG_OLD_EDGES, FLAG_NEW_EDGES, FLAG_WATER, FLAG_THETA, FLAG_OUTPUT, FLAG_CLOSURE = 1, 2, 4, 8, 16, 32
_FLAG_NAMES = {FLAG_OLD_EDGES: "old edges are not finite, strictly increasing and representable", FLAG_NEW_EDGES: "new edges are not finite, strictly increasing and representable",
               FLAG_WATER: "old water is not finite or lies outside [0, interval width]", FLAG_THETA: "upper/lower theta is not finite or lies outside [0, 1]",
               FLAG_OUTPUT: "an output is not finite or a new interval holds more than its width plus its derived bound",
               FLAG_CLOSURE: "the per-column closure exceeds its derived bound"}
CLOSURE_TERMS_FORMULA = "n_col = 2 L^2 + 9 L + 8"

# Live (C,) work arrays of `reference_remap` at its peak (counted from the code below and checked by tracemalloc in the tests).
REFERENCE_TEMP_F64 = 12
REFERENCE_TEMP_BOOL = 6
CUDA_BLOCK = 128  # threads per block of the CUDA column kernel (a power of two; the block reduction tree depends on it)
CUDA_FINAL_BLOCK = 256  # threads of the single finalize block that reduces the per-block partials (a power of two)
CUDA_REDUCED_ROWS = 5  # per-column contributions reduced on the device: old total, new total, added, removed, scale
DIAG_WORDS = 10  # the device diagnostic/packet: 10 uint64 words = flag OR, flagged count, first flagged column<<8|flags, bit patterns of max|res| and max res/bound, then the bit patterns of the 5 sums
PACKET_BYTES = DIAG_WORDS * 8  # the ONE host read of the CUDA path: exactly the diagnostic words (raw integers, no float conversion)
DEVICE_DIAG_BYTES = PACKET_BYTES
MAX_CUDA_COLUMNS = min(2**56 - 1, (2**31 - 1) * CUDA_BLOCK)  # explicit range gate: (column << 8 | flags) must fit 64 bits and the grid must fit the launch limit
FLAG_SEARCH_CHUNK = 4096  # the host search for the first flagged column walks the flag array in chunks of this many bytes (no domain-sized index array)
HOST_OBJECT_ALLOWANCE_BYTES = 16384 + 3 * 8 * FLAG_SEARCH_CHUNK  # declared allowance: small Python objects/exception text (16 KiB) + three chunk-sized host temporaries (flag-search indices, |residual|, residual/bound)


class SoilRemapError(ValueError):
    """Invalid input (metadata or value), an unrepresentable result or a failed closure. Nothing was published."""


class SoilRemapValidationError(SoilRemapError):
    """An input check failed. `bit` is the lowest failing flag, `n_flagged` the number of flagged columns, `first_column` the lowest flagged column index."""

    def __init__(self, message: str, *, bit: int | None = None, n_flagged: int = 0, first_column: int | None = None):
        super().__init__(message)
        self.bit, self.n_flagged, self.first_column = bit, int(n_flagged), first_column


class SoilRemapClosureError(SoilRemapError):
    """An output was not representable or the per-column/global closure exceeded its derived bound."""

    def __init__(self, message: str, *, bit: int | None = None, n_flagged: int = 0, first_column: int | None = None):
        super().__init__(message)
        self.bit, self.n_flagged, self.first_column = bit, int(n_flagged), first_column


class SoilRemapBackendUnavailableError(RuntimeError):
    """The requested backend (Numba, CuPy/CUDA, or the host memory query) is unavailable. There is no fallback to another backend."""


class SoilRemapMemoryError(MemoryError):
    """The estimate plus the caller's reserve exceeds the declared budget or the reported available bytes. Nothing was allocated."""


# --- memory ------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class RemapMemoryEstimate:
    backend: str
    n_layers: int
    n_columns: int
    input_bytes_excluded: int  # the caller's five input arrays (kept by the caller, never copied)
    output_bytes: int  # retained outputs: new water (L, C) and four flux (C,) arrays
    scratch_bytes: int  # (5, C) stats + C flag bytes + backend work arrays (+ the CUDA host packet)
    reserve_bytes: int
    required_bytes: int  # output + scratch + reserve on the MAIN admitted side (host for numpy/numba, device for cuda)
    terms: dict
    host_required_bytes: int = 0  # host side: everything for numpy/numba (== required_bytes); for cuda the small packet + object allowance + host reserve
    device_required_bytes: int = 0  # device side (cuda only; 0 on the CPU backends)
    host_reserve_bytes: int = 0
    scope: str = ("modelled bytes of this call's own arrays only; the caller's inputs, interpreter/JIT/NVRTC/driver state, the allocator pool policy and whole-process "
                  "memory are excluded; an availability reading is not a reservation")


def cuda_block_count(n_columns: int) -> int:
    """Blocks of the column kernel (= columns of the partial-sum array): `ceil(C / CUDA_BLOCK)`."""
    return (int(n_columns) + CUDA_BLOCK - 1) // CUDA_BLOCK


def _count(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or int(value) < minimum:
        raise SoilRemapError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return int(value)


def estimate_remap_memory(n_layers: int, n_columns: int, *, backend: str = "numpy", reserve_bytes: int = 0, host_reserve_bytes: int = 0) -> RemapMemoryEstimate:
    """Modelled bytes of one `remap_intervals` call from the shape alone (metadata only; nothing is allocated). `reserve_bytes` is counted once on the main admitted side
    (host for numpy/numba, device for cuda); `host_reserve_bytes` (cuda only, else must be 0) is the separate host reserve of the small host packet."""
    L, C = _count(n_layers, "n_layers", minimum=1), _count(n_columns, "n_columns")
    reserve = _count(reserve_bytes, "reserve_bytes")
    host_reserve = _count(host_reserve_bytes, "host_reserve_bytes")
    if host_reserve and backend != "cuda":
        raise SoilRemapError("host_reserve_bytes applies only to the cuda backend (the CPU backends use reserve_bytes on the host)")
    if backend not in BACKENDS:
        raise SoilRemapError(f"backend must be one of {BACKENDS}, got {backend!r}")
    inputs = 8 * ((L + 1) * C * 2 + L * C + 2 * C)
    output = 8 * L * C + 4 * 8 * C
    if backend == "cuda" and C > MAX_CUDA_COLUMNS:
        raise SoilRemapError(f"the cuda backend supports at most {MAX_CUDA_COLUMNS} columns per call (64-bit packed first-column diagnostic and the launch grid limit), got {C}")
    n_blocks = cuda_block_count(C)
    # CUDA keeps no per-column stats/flag array: contributions are reduced in registers/shared memory into `n_blocks` partials; atomics fill the diagnostic words
    stats, flags = (0, 0) if backend == "cuda" else (5 * 8 * C, C)
    work = REFERENCE_TEMP_F64 * 8 * C + REFERENCE_TEMP_BOOL * C if backend == "numpy" else 0
    diag = DEVICE_DIAG_BYTES if backend == "cuda" else 0
    partials = CUDA_REDUCED_ROWS * 8 * n_blocks if backend == "cuda" else 0
    packet = PACKET_BYTES if backend == "cuda" else 0
    scratch = stats + flags + work + diag + partials + packet
    host_fixed = HOST_OBJECT_ALLOWANCE_BYTES
    if backend == "cuda":
        device_required = output + diag + partials + reserve
        host_required = packet + host_fixed + host_reserve
        required = device_required
    else:
        device_required = 0
        required = output + scratch + host_fixed + reserve
        host_required = required
    terms = {"outputs": output, "stats_block_5xC": stats, "flags": flags, "reference_work_arrays": work, "cuda_device_diagnostics_and_packet_words": diag, "cuda_block_partials_5xBlocks": partials, "cuda_blocks": n_blocks if backend == "cuda" else 0,
             "cuda_static_shared_memory_bytes_per_block": {"column_kernel": CUDA_REDUCED_ROWS * 8 * CUDA_BLOCK, "finalize_kernel": CUDA_REDUCED_ROWS * 8 * CUDA_FINAL_BLOCK} if backend == "cuda" else None,
             "cuda_host_packet": packet,
             "host_object_and_flag_search_allowance": host_fixed, "side": "device+host" if backend == "cuda" else "host", "no_LxLxC_no_LCK_no_whole_bed": True,
             "excluded": "NVRTC/kernel module load, CUDA context, allocator pool policy, interpreter and JIT state (the device reductions are explicit RawKernels: no library reduction workspace exists)"}
    return RemapMemoryEstimate(backend, L, C, inputs, output, scratch, reserve, required, terms, host_required, device_required, host_reserve)


def default_host_available_bytes() -> int:
    """Available physical host memory (`sysconf`); a platform without it is a setup error (inject `available_bytes_fn`). A reading, not a reservation."""
    try:
        pages, page = os.sysconf("SC_AVPHYS_PAGES"), os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, ValueError, OSError) as exc:
        raise SoilRemapBackendUnavailableError(f"the host cannot report available memory ({type(exc).__name__}: {exc}); pass available_bytes_fn") from exc
    if pages < 0 or page <= 0:
        raise SoilRemapBackendUnavailableError("the host reported an invalid available-memory figure; pass available_bytes_fn")
    return int(pages) * int(page)


def closure_terms(n_layers: int) -> int:
    """`n_col = 2 L^2 + 9 L + 8`, the term count of the per-column closure (derivation in the module docstring)."""
    L = _count(n_layers, "n_layers", minimum=1)
    return 2 * L * L + 9 * L + 8


def _bound(n_terms: int, scale: float, what: str) -> float:
    """MAPLE's coefficient times a derived scale, with the scale/bound checked finite HERE so an unrepresentable derived quantity is a typed `SoilRemapClosureError`
    (never a raw `ValueError` from the helper and never a vacuous `<= inf` comparison)."""
    from maple_syrup.conservation import volume_roundoff_bound_m3

    if not (isinstance(scale, float) and np.isfinite(scale) and scale >= 0.0):
        raise SoilRemapClosureError(f"the derived {what} scale {scale!r} is not finite and nonnegative; nothing was published", bit=FLAG_CLOSURE)
    try:
        bound = volume_roundoff_bound_m3(int(n_terms), float(scale))
    except (ValueError, OverflowError) as exc:
        raise SoilRemapClosureError(f"the derived {what} bound is not representable ({exc}); nothing was published", bit=FLAG_CLOSURE) from None
    if not np.isfinite(bound):
        raise SoilRemapClosureError(f"the derived {what} bound {bound!r} is not finite; nothing was published", bit=FLAG_CLOSURE)
    return bound


def _coefficients(L: int) -> tuple[float, float]:
    return _bound(closure_terms(L), 1.0, "column closure coefficient"), _bound(2 * L + 4, 1.0, "output coefficient")


def _global_bound(n_columns: int, L: int, scale_sum: float, area: float) -> float:
    with np.errstate(all="ignore"):
        scale = float(area * scale_sum)
    return _bound(n_columns * closure_terms(L) + 8, scale, "global closure")


# --- result ------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class RemapResult:
    """Fresh arrays of the input backend. `water_m` is `(L, C)`; the four fluxes are `(C,)`. Component/trial only: not an accepted physical state."""

    water_m: Any
    removed_above_m: Any
    removed_below_m: Any
    added_above_m: Any
    added_below_m: Any
    n_layers: int
    n_columns: int
    backend: str
    closure: dict
    memory: RemapMemoryEstimate
    label: str = LABEL

    def summary(self) -> dict:
        return {"label": self.label, "backend": self.backend, "n_layers": self.n_layers, "n_columns": self.n_columns, "closure": dict(self.closure),
                "memory": {"required_bytes": self.memory.required_bytes, "output_bytes": self.memory.output_bytes, "scratch_bytes": self.memory.scratch_bytes,
                           "input_bytes_excluded": self.memory.input_bytes_excluded}}


# --- metadata checks -----------------------------------------------------------------------------------------------------------
def _cupy_for_cuda():
    from maple_syrup import soil_water_remap_cuda as rc

    return rc.require_cuda()


def _check_array(name: str, array: Any, shape: tuple[int, ...], backend: str, cp: Any) -> None:
    expected = cp.ndarray if backend == "cuda" else np.ndarray
    if type(array) is not expected:
        raise SoilRemapError(f"{name} must be exactly a {'cupy' if backend == 'cuda' else 'numpy'}.ndarray for backend {backend!r} (no subclass, no masked array, no implicit "
                             f"transfer), got {type(array).__module__}.{type(array).__name__}")
    if array.dtype != np.float64:
        raise SoilRemapError(f"{name} must be float64, got {array.dtype}")
    if tuple(array.shape) != tuple(shape):
        raise SoilRemapError(f"{name} must have shape {tuple(shape)}, got {tuple(array.shape)}")
    if not array.flags.c_contiguous:
        raise SoilRemapError(f"{name} must be C-contiguous")
    if backend == "cuda":
        from maple_syrup import soil_water_remap_cuda as rc

        if int(array.device.id) != rc.current_device_id(cp):
            raise SoilRemapError(f"{name} lives on device {int(array.device.id)} but the current device is {rc.current_device_id(cp)}; no implicit device transfer")


def _area(value: Any) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise SoilRemapError(f"cell_area_m2 must be a finite positive real scalar, got {value!r}")
    out = float(value)
    if not np.isfinite(out) or out <= 0.0:
        raise SoilRemapError(f"cell_area_m2 must be finite and > 0, got {value!r}")
    return out


def _raise_flags(flags_info: tuple[int, int, int | None, int]) -> None:
    """`flags_info = (OR of all flags, n_flagged, first flagged column, the flags of that first column)`; the reported bit is the lowest flag OF THE FIRST FLAGGED COLUMN."""
    bits, n_flagged, first, first_flags = flags_info
    if bits == 0:
        return
    low = first_flags & -first_flags
    text = f"{_FLAG_NAMES[low]} ({n_flagged} column(s) flagged, first column {first}; OR of all flags {bits}); nothing was published"
    if low >= FLAG_OUTPUT:
        raise SoilRemapClosureError(text, bit=low, n_flagged=n_flagged, first_column=first)
    raise SoilRemapValidationError(text, bit=low, n_flagged=n_flagged, first_column=first)


# --- the NumPy reference (the definition of the arithmetic) ----------------------------------------------------------------------
def reference_remap(old_edges_m: np.ndarray, new_edges_m: np.ndarray, water_m: np.ndarray, upper_theta: np.ndarray, lower_theta: np.ndarray,
                    coef_col: float, coef_out: float) -> tuple:
    """Vectorised over the C columns with a Python loop over layers; O(C) live work arrays (no L x L x C). Returns `(new_water, rem_a, rem_b, add_a, add_b, flags, stats)` with
    `stats` `(5, C)` = residual, bound, old total, new total, scale. Inputs are only read. Host NumPy only."""
    eo, en, W = old_edges_m, new_edges_m, water_m
    L, C = int(W.shape[0]), int(W.shape[1])
    out = np.zeros((L, C))
    rem_a, rem_b, add_a, add_b = np.zeros(C), np.zeros(C), np.zeros(C), np.zeros(C)
    flags = np.zeros(C, dtype=np.uint8)
    stats = np.zeros((5, C))
    if C == 0:
        return out, rem_a, rem_b, add_a, add_b, flags, stats
    up, lo = upper_theta, lower_theta
    old_total, new_total, scale = stats[2], stats[3], stats[4]
    with np.errstate(all="ignore"):
        wi, wn = np.zeros(C), np.zeros(C)
        b1, b2, b3, acc = np.zeros(C), np.zeros(C), np.zeros(C), np.zeros(C)
        bad = np.zeros(C, dtype=bool)
        # -- value checks of the inputs (per layer, O(C) temporaries) --
        for i in range(L):
            np.subtract(eo[i + 1], eo[i], out=wi)
            bad = ~(np.isfinite(wi) & (wi > 0.0))
            flags |= bad.astype(np.uint8) * np.uint8(FLAG_OLD_EDGES)
            np.subtract(en[i + 1], en[i], out=wn)
            bad = ~(np.isfinite(wn) & (wn > 0.0))
            flags |= bad.astype(np.uint8) * np.uint8(FLAG_NEW_EDGES)
            np.subtract(eo[i + 1], eo[i], out=wi)
            bad = ~(np.isfinite(W[i]) & (W[i] >= 0.0) & (W[i] <= wi))
            flags |= bad.astype(np.uint8) * np.uint8(FLAG_WATER)
        bad = ~(np.isfinite(up) & (up >= 0.0) & (up <= 1.0) & np.isfinite(lo) & (lo >= 0.0) & (lo <= 1.0))
        flags |= bad.astype(np.uint8) * np.uint8(FLAG_THETA)
        ok = flags == 0
        if not ok.any():
            return out, rem_a, rem_b, add_a, add_b, flags, stats
        # -- totals and scale (ascending order) --
        for i in range(L):
            np.subtract(eo[i + 1], eo[i], out=wi)
            np.add(scale, wi, out=scale)
            np.add(old_total, W[i], out=old_total)
        for j in range(L):
            np.subtract(en[j + 1], en[j], out=wn)
            np.add(scale, wn, out=scale)
        # -- removed_above / removed_below --
        for i in range(L):
            np.subtract(eo[i + 1], eo[i], out=wi)
            np.minimum(eo[i + 1], en[0], out=b1)
            np.subtract(b1, eo[i], out=b1)
            np.maximum(b1, 0.0, out=b1)
            np.divide(b1, wi, out=b1)
            np.multiply(W[i], b1, out=b1)
            np.add(rem_a, b1, out=rem_a)
            np.maximum(eo[i], en[L], out=b2)
            np.subtract(eo[i + 1], b2, out=b2)
            np.maximum(b2, 0.0, out=b2)
            np.divide(b2, wi, out=b2)
            np.multiply(W[i], b2, out=b2)
            np.add(rem_b, b2, out=rem_b)
        # -- added_above / added_below --
        np.minimum(eo[0], en[L], out=b1)
        np.subtract(b1, en[0], out=b1)
        np.maximum(b1, 0.0, out=b1)
        np.multiply(up, b1, out=add_a)
        np.maximum(eo[L], en[0], out=b2)
        np.subtract(en[L], b2, out=b2)
        np.maximum(b2, 0.0, out=b2)
        np.multiply(lo, b2, out=add_b)
        # -- the new water, interval by interval --
        bad_out = np.zeros(C, dtype=bool)
        for j in range(L):
            acc[:] = 0.0
            for i in range(L):
                np.subtract(eo[i + 1], eo[i], out=wi)
                np.minimum(eo[i + 1], en[j + 1], out=b1)
                np.maximum(eo[i], en[j], out=b2)
                np.subtract(b1, b2, out=b1)
                np.maximum(b1, 0.0, out=b1)
                np.divide(b1, wi, out=b1)
                np.multiply(W[i], b1, out=b1)
                np.add(acc, b1, out=acc)
            np.minimum(en[j + 1], eo[0], out=b1)
            np.subtract(b1, en[j], out=b1)
            np.maximum(b1, 0.0, out=b1)
            np.multiply(up, b1, out=b1)
            np.add(acc, b1, out=acc)
            np.maximum(en[j], eo[L], out=b2)
            np.subtract(en[j + 1], b2, out=b2)
            np.maximum(b2, 0.0, out=b2)
            np.multiply(lo, b2, out=b2)
            np.add(acc, b2, out=acc)
            out[j] = acc
            np.subtract(en[j + 1], en[j], out=wn)
            np.multiply(wn, coef_out, out=b3)
            np.add(wn, b3, out=b3)
            bad_out |= ~(np.isfinite(acc) & (acc <= b3))
        # -- unchanged geometry (numerically equal edges) is an IDENTITY: every water bit is copied (signed zeros, subnormals) and every flux is exactly +0 --
        same = bad  # reuse the boolean work array
        same[:] = True
        for i in range(L + 1):
            same &= eo[i] == en[i]
        for j in range(L):
            np.copyto(out[j], W[j], where=same)
        for flux in (rem_a, rem_b, add_a, add_b):
            np.copyto(flux, 0.0, where=same)
        for j in range(L):
            np.add(new_total, out[j], out=new_total)
        bad_out |= ~(np.isfinite(rem_a) & np.isfinite(rem_b) & np.isfinite(add_a) & np.isfinite(add_b))
        flags |= (bad_out & ok).astype(np.uint8) * np.uint8(FLAG_OUTPUT)
        # -- per-column closure --
        np.subtract(new_total, old_total, out=b1)
        np.add(add_a, add_b, out=b2)
        np.add(rem_a, rem_b, out=b3)
        np.subtract(b2, b3, out=b2)
        np.subtract(b1, b2, out=stats[0])
        np.multiply(scale, coef_col, out=stats[1])
        bad_cl = ~(np.isfinite(stats[0]) & np.isfinite(stats[1]) & (np.abs(stats[0]) <= stats[1]))
        flags |= (bad_cl & ok & ~bad_out).astype(np.uint8) * np.uint8(FLAG_CLOSURE)
    return out, rem_a, rem_b, add_a, add_b, flags, stats


# --- dispatch ----------------------------------------------------------------------------------------------------------------
def _flags_info_host(flags: np.ndarray) -> tuple[int, int, int | None, int]:
    if flags.size == 0:
        return 0, 0, None, 0
    n = int(np.count_nonzero(flags))  # no allocation
    if n == 0:
        return 0, 0, None, 0
    first = None
    for start in range(0, flags.size, FLAG_SEARCH_CHUNK):  # chunked: no domain-sized index array
        hit = np.flatnonzero(flags[start:start + FLAG_SEARCH_CHUNK])
        if hit.size:
            first = start + int(hit[0])
            break
    return int(np.bitwise_or.reduce(flags)), n, first, int(flags[first])


def remap_intervals(old_edges_m: Any, new_edges_m: Any, water_m: Any, *, upper_theta: Any, lower_theta: Any, cell_area_m2: Any, backend: str = "numpy",
                    memory_budget_bytes: Any, reserve_bytes: Any = 0, available_bytes_fn: Callable[[], int] | None = None,
                    host_memory_budget_bytes: Any = None, host_reserve_bytes: Any = 0, host_available_bytes_fn: Callable[[], int] | None = None) -> RemapResult:
    """Remap piecewise-uniform water from the old to the new explicit intervals of `C` selected columns (module docstring: model, order, checks, bounds).

    Arguments (all arrays float64, C-contiguous, exactly of the backend's namespace; `L >= 1`, `C >= 0`):
      old_edges_m `(L+1, C)`, new_edges_m `(L+1, C)` depth coordinates (m, increasing downward, one reference per column); water_m `(L, C)` m of water per old interval
      (0 <= W <= width); upper_theta, lower_theta `(C,)` volumetric fractions in [0, 1] of the material entering above the old top / below the old bottom (no defaults);
      cell_area_m2 finite positive host scalar (volumes of the global closure); backend "numpy" (the reference), "numba" (compiled CPU) or "cuda" (CuPy RawKernel); there is no
      fallback between them; memory_budget_bytes (required int >= 0) the caller's declared quota for this call's own allocation (inputs excluded); reserve_bytes the caller's
      reserve counted once inside the admitted side; available_bytes_fn an injectable reading of the admitted side's available bytes (host for "numpy"/"numba", device for "cuda").
      CUDA host admission (new, narrow; "cuda" only, refused for the CPU backends): the call also keeps a small host packet (`estimate.host_required_bytes`: the 128 B scalar packet + a declared
      object allowance + `host_reserve_bytes`), admitted SEPARATELY before any allocation against `host_memory_budget_bytes` (None = no separate host quota) and a truthful host reading
      `host_available_bytes_fn` (default the host `sysconf` reading). The device side uses `memory_budget_bytes`, `reserve_bytes` and `available_bytes_fn` as above.
    Returns a fresh `RemapResult` (never aliases an input). Raises `SoilRemapError` (metadata/value), `SoilRemapValidationError`, `SoilRemapClosureError`,
    `SoilRemapMemoryError` or `SoilRemapBackendUnavailableError`; the inputs are never modified and nothing is published on failure.
    """
    if backend not in BACKENDS:
        raise SoilRemapError(f"backend must be one of {BACKENDS}, got {backend!r}")
    budget = _count(memory_budget_bytes, "memory_budget_bytes")
    reserve = _count(reserve_bytes, "reserve_bytes")
    area = _area(cell_area_m2)
    if available_bytes_fn is not None and not callable(available_bytes_fn):
        raise SoilRemapError("available_bytes_fn must be callable or None")
    host_reserve = _count(host_reserve_bytes, "host_reserve_bytes")
    host_budget = None if host_memory_budget_bytes is None else _count(host_memory_budget_bytes, "host_memory_budget_bytes")
    if host_available_bytes_fn is not None and not callable(host_available_bytes_fn):
        raise SoilRemapError("host_available_bytes_fn must be callable or None")
    if backend != "cuda" and (host_reserve or host_budget is not None or host_available_bytes_fn is not None):
        raise SoilRemapError("host_memory_budget_bytes / host_reserve_bytes / host_available_bytes_fn apply only to the cuda backend (the CPU backends use the main budget, reserve and reading on the host)")
    cp = _cupy_for_cuda() if backend == "cuda" else None
    if backend == "numba":
        from maple_syrup import soil_water_remap_numba as rn

        rn.require_numba()
    # -- shape metadata (before any array of this call exists) --
    if type(water_m) is not (cp.ndarray if backend == "cuda" else np.ndarray):
        raise SoilRemapError(f"water_m must be exactly a {'cupy' if backend == 'cuda' else 'numpy'}.ndarray for backend {backend!r}, got {type(water_m).__module__}.{type(water_m).__name__}")
    if water_m.ndim != 2:
        raise SoilRemapError(f"water_m must be 2-D (L, C), got shape {tuple(water_m.shape)}")
    L, C = int(water_m.shape[0]), int(water_m.shape[1])
    if L < 1:
        raise SoilRemapError("at least one layer is required (L >= 1)")
    _check_array("water_m", water_m, (L, C), backend, cp)
    _check_array("old_edges_m", old_edges_m, (L + 1, C), backend, cp)
    _check_array("new_edges_m", new_edges_m, (L + 1, C), backend, cp)
    _check_array("upper_theta", upper_theta, (C,), backend, cp)
    _check_array("lower_theta", lower_theta, (C,), backend, cp)
    # -- admission (before allocation); the two sides are checked separately --
    estimate = estimate_remap_memory(L, C, backend=backend, reserve_bytes=reserve, host_reserve_bytes=host_reserve)
    side = "device" if backend == "cuda" else "host"
    _admit(estimate.required_bytes, budget, available_bytes_fn, side, backend, L, C, reserve, estimate, cp, quota_name="memory_budget_bytes")
    if backend == "cuda":
        _admit(estimate.host_required_bytes, host_budget, host_available_bytes_fn, "host", backend, L, C, host_reserve, estimate, None, quota_name="host_memory_budget_bytes")
    coef_col, coef_out = _coefficients(L)
    # -- compute --
    if backend == "numpy":
        out, rem_a, rem_b, add_a, add_b, flags, stats = reference_remap(old_edges_m, new_edges_m, water_m, upper_theta, lower_theta, coef_col, coef_out)
        info = _flags_info_host(flags)
        _raise_flags(info)
        sums = _host_sums(out, rem_a, rem_b, add_a, add_b, stats)
    elif backend == "numba":
        out, rem_a, rem_b, add_a, add_b, flags, stats = rn.run(old_edges_m, new_edges_m, water_m, upper_theta, lower_theta, coef_col, coef_out)
        info = _flags_info_host(flags)
        _raise_flags(info)
        sums = _host_sums(out, rem_a, rem_b, add_a, add_b, stats)
    else:
        from maple_syrup import soil_water_remap_cuda as rc

        out, rem_a, rem_b, add_a, add_b, info, sums = rc.run(cp, old_edges_m, new_edges_m, water_m, upper_theta, lower_theta, coef_col, coef_out)
        _raise_flags(info)
    closure = _global_closure(sums, L, C, area)
    residual, bound_g = closure["global_residual_m3"], closure["global_bound_m3"]
    if not (np.isfinite(residual) and np.isfinite(bound_g) and abs(residual) <= bound_g):  # NaN/inf never pass a comparison
        raise SoilRemapClosureError(f"the global closure {closure['global_residual_m3']!r} m3 exceeds its derived bound {closure['global_bound_m3']!r} m3; nothing was published",
                                    bit=FLAG_CLOSURE)
    return RemapResult(out, rem_a, rem_b, add_a, add_b, L, C, backend, closure, estimate)


def _admit(required: int, budget: int | None, reading_fn: Callable[[], int] | None, side: str, backend: str, L: int, C: int, reserve: int, estimate: RemapMemoryEstimate,
           cp: Any, *, quota_name: str) -> None:
    """One side's admission BEFORE any allocation: the requirement against the caller's declared quota (None = no quota) and against a truthful available-bytes reading."""
    if budget is not None and required > budget:
        raise SoilRemapMemoryError(f"remap of {L} layers x {C} columns ({backend}) needs an estimated {required} B on the {side} side (reserve {reserve} B included) but the caller's declared "
                                   f"{quota_name} is {budget}; nothing was allocated. Scope: {estimate.scope}")
    if reading_fn is None:
        if side == "device":
            from maple_syrup import soil_water_remap_cuda as rc

            reading_fn = rc.default_device_available_bytes_fn(cp)
        else:
            reading_fn = default_host_available_bytes
    available = reading_fn()
    if isinstance(available, (bool, np.bool_)) or not isinstance(available, (int, np.integer)) or int(available) < 0:
        raise SoilRemapError(f"the {side} available-bytes reader must return a non-Boolean integer >= 0, got {available!r}")
    if required > int(available):
        raise SoilRemapMemoryError(f"remap of {L} layers x {C} columns ({backend}) needs an estimated {required} B on the {side} side (reserve {reserve} B included) but only "
                                   f"{int(available)} B are reported available; nothing was allocated. Scope: {estimate.scope}")


def _host_sums(out, rem_a, rem_b, add_a, add_b, stats) -> dict:
    """Totals of the per-column arrays (host arrays, `np.sum`): old, new, added, removed, scale, max |residual|, max residual/bound. The two maxima are taken in chunks of
    `FLAG_SEARCH_CHUNK` columns (bounded temporaries, no domain-sized |residual| or ratio array). Every total must be finite (typed error otherwise)."""
    if stats.shape[1] == 0:
        return {"old": 0.0, "new": 0.0, "added": 0.0, "removed": 0.0, "scale": 0.0, "max_abs_residual": 0.0, "max_ratio": 0.0}
    max_abs, max_ratio = 0.0, 0.0
    with np.errstate(all="ignore"):
        for start in range(0, stats.shape[1], FLAG_SEARCH_CHUNK):
            r = np.abs(stats[0, start:start + FLAG_SEARCH_CHUNK])
            max_abs = max(max_abs, float(r.max()))
            np.divide(r, stats[1, start:start + FLAG_SEARCH_CHUNK], out=r)
            max_ratio = max(max_ratio, float(r.max()))
        sums = {"old": float(np.sum(stats[2])), "new": float(np.sum(stats[3])), "added": float(np.sum(add_a) + np.sum(add_b)),
                "removed": float(np.sum(rem_a) + np.sum(rem_b)), "scale": float(np.sum(stats[4])), "max_abs_residual": max_abs, "max_ratio": max_ratio}
    _require_finite(sums)
    return sums


def _require_finite(sums: dict) -> None:
    bad = [k for k, v in sums.items() if not np.isfinite(v)]
    if bad:
        raise SoilRemapClosureError(f"the derived totals {bad} are not finite (overflow or NaN); nothing was published", bit=FLAG_CLOSURE)


def _global_closure(sums: dict, L: int, C: int, area: float) -> dict:
    coef_col, _ = _coefficients(L)
    if C == 0:
        return {"column_bound_terms": closure_terms(L), "global_bound_terms": 8, "max_abs_column_residual_m": 0.0, "max_column_residual_over_bound": 0.0,
                "total_old_m3": 0.0, "total_new_m3": 0.0, "added_m3": 0.0, "removed_m3": 0.0, "global_residual_m3": 0.0, "global_bound_m3": 0.0}
    with np.errstate(all="ignore"):
        old, new, added, removed = (float(area * sums[k]) for k in ("old", "new", "added", "removed"))
        residual = float((new - old) - (added - removed))
    if not all(np.isfinite(v) for v in (old, new, added, removed, residual)):
        raise SoilRemapClosureError("a derived global total or the global residual is not finite (overflow or NaN) for this cell area; nothing was published", bit=FLAG_CLOSURE)
    return {"column_bound_terms": closure_terms(L), "global_bound_terms": C * closure_terms(L) + 8, "column_bound_coefficient": coef_col,
            "max_abs_column_residual_m": sums["max_abs_residual"], "max_column_residual_over_bound": sums["max_ratio"], "total_old_m3": old, "total_new_m3": new,
            "added_m3": added, "removed_m3": removed, "global_residual_m3": residual, "global_bound_m3": _global_bound(C, L, sums["scale"], area)}
