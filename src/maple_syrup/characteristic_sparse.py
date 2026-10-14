"""Sparse nonempty-bin (CSR) characteristic transport -- CPU/Numba reference (CW-B2 candidate).

Isolated candidate for the corrected CW-B0 design
(agent_handoffs/tasks/conservative_water/cw_b_design_corrected_report.md
section 2): the SAME physics, quantization (`B` bins of width `dx / B`,
1..128, default 32), logarithmic-mean merge, containment, request semantics
and `TransportStep` contract as `characteristic_transport.characteristic_step`,
with the phase state stored as nonempty bins only. Nothing in the production
modules is modified; the dense kernels remain the comparator.

Backends (CW-B4). `implementation="numba"` (DEFAULT, unchanged CPU arithmetic)
runs on a host NumPy network with host arrays and a NumPy `SparsePhaseState`;
`implementation="cuda"` (`characteristic_cuda_csr`) runs on a CuPy network with
CuPy arrays and a CuPy `SparsePhaseState` (same dataclass, `xp = cupy`, device
int64 row pointers and slot maps, uint8 bins, FP64 fractions/positions). The
two never mix: a CuPy network or array on the Numba path, a NumPy network or
host array on the CUDA path, or a phase in the other namespace is refused
explicitly; there is no transfer, conversion or fallback in either direction.
`empty_sparse_phase_state`, `validate_sparse_phase_state`, `sparse_to_dense`,
`sparse_from_dense` and `rebind_sparse_phase_state` dispatch on the namespace
of the network/phase they receive; the CPU functions below are unchanged.

State (`SparsePhaseState`, frozen, host NumPy)
---------------------------------------------
    row_ptr     (n_active nc + 1,) int64   row = slot nc + class; slot indexes the
                                           ascending flat active cells (`cell_of_slot`)
    bin         (nnz,) uint8               bin 0..B-1, STRICTLY increasing within a row
    fraction    (nnz,) FP64                share of the cell/class MAPLE mobile total
                                           (> 0; sums to 1 per nonempty row)
    position_m  (nnz,) FP64                representative distance from the upstream
                                           face, in [bin dx/B, (bin+1) dx/B) and < dx
    cell_of_slot (n_active,) int64, slot_of_cell (n,) int64 (-1 inactive),
    network_sha256 (binds the routing graph), dx_m, n_bins, n_classes, shape

A row with zero mobile mass is EMPTY (its canonical dense value, fraction
`e_0` and position 0, is implicit and never stored); a nonempty row stores
exactly its occupied bins; every class is dynamic (rows appear when a
cell/class gains mobile mass and disappear when it settles or exports). There
is no frozen composition, lagged pickup, fixed capacity truncation or reduced
bin count: `B` is unchanged. The dense `(ny, nx, nc, B)` arrays are produced
ONLY by the explicit small-domain helper `sparse_to_dense(max_bytes)` (and
read by `sparse_from_dense`), both of which refuse above their byte budget
BEFORE allocating; validation, initialisation, transport and normalisation
never allocate a `(..., B)` array.

Step (`sparse_characteristic_step`)
-----------------------------------
Same positional contract as the dense step: `(network, mobile_before_kg
(pre-pickup, (ny, nx, nc)), phase, pickup_kg (ACTUAL pickup, injected once at
x = 0 in the first substep), v, r = 1/L, settle_mask, dt_s)`, keywords
`n_substeps`, `courant_max` (<= 0.5: one face crossing per packet per
substep; a violation raises the recoverable `TransportStepRejected`),
`implementation` (`"numba"` on a NumPy network, or `"cuda"`, which delegates
to `characteristic_cuda_csr.cuda_sparse_step` on a CuPy network; there is no
array path, no implicit fallback and no device transfer; any other value or
mismatched namespace is refused explicitly) and
`max_nnz` (predeclared capacity: a substep whose scanned `nnz` exceeds it
raises `SparseCapacityError` naming the required `nnz`; mass is never
discarded). Returns `SparseCharacteristicStep` with the unchanged
`TransportStep` (`T = M_remaining + Dep + E`, requests, MAPLE face layout,
per-class budgets at the actual MAPLE `summation_error_bound_kg` policy with
the dense `operation_count`) and the `SparsePhaseState` of `M_remaining`.

Internally the substeps carry the UNNORMALISED packet-mass CSR (masses, not
fractions; `characteristic_numba_csr` count -> host scan -> write), as the
dense Numba kernel carries `W`, and normalise once at the end. The kernel
module documents the accumulation order that makes bitwise agreement with the
dense Numba kernel an EXPECTATION tested by `tests/conservative_csr`; it is
not assumed here.

Row sums (final normalisation). The per-row remaining mass that normalises
the fractions is `W.sum(axis=-1)` on the dense path. Two implementations exist
here, both without an `(n, nc, B)` array:

* `row_sums_numpy` -- NumPy's own `sum(axis=-1)` over a BOUNDED
  `(ROW_SUM_CHUNK_ROWS, B)` scratch (rows staged in chunks with exact zeros in
  the empty bins). It is the REFERENCE: whatever the running NumPy's reduction
  arithmetic is, this reproduces it. Used by validation/tests and by the
  runtime self-check below.
* `characteristic_numba_csr.run_row_sums` -- the compiled EMULATION of NumPy's
  pairwise reduction (8 accumulators, identity start). It is the PRODUCTION
  path of `sparse_characteristic_step` (selected after the root's independent
  byte-exact probes on every supported bin count and a ~6x local
  microbenchmark; see the CW-B2 correction report). Bitwise equality of an
  emulation with NumPy is NOT assumed across NumPy versions: before the first
  production use in a process, `row_sum_emulation_self_check` compares the
  compiled sums with `row_sums_numpy` and with NumPy's dense 3-D reduction,
  raw bytes, on a deterministic fixture covering `ROW_SUM_SELF_CHECK_BINS`
  (every power-of-two edge from 1 to 128, their neighbours, 100), signed and
  33-decade magnitudes, empty, single-entry and full rows. A mismatch raises
  `RowSumEmulationMismatch` (a `TransportError`) naming the NumPy version;
  there is NO fallback to the NumPy path. The NumPy versions on which the
  root recorded byte-exact probes are listed in
  `ROW_SUM_EMULATION_VERIFIED_NUMPY_VERSIONS`; `row_sum_policy()` reports the
  running versions and the self-check outcome for provenance.

Validation (`validate_sparse_phase_state`): namespace, dtypes, shapes,
`row_ptr` monotone with the right endpoints, strictly increasing bins below
`B`, finiteness, non-negativity, positions inside their bin and below `dx`,
nonempty row <=> positive mobile, fraction sums within
`fraction_sum_tolerance(B)`, mobile zero on inactive cells, slot map equal to
the network's active cells, graph digest. Every refusal is a `TransportError`
raised before any result exists; no caller-owned array is modified; returned
arrays are frozen.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from types import ModuleType
from typing import Any

import numpy as np
from maple.surface.voxels._numerics import summation_error_bound_kg

from maple_syrup.characteristic_transport import (
    DEFAULT_COURANT_MAX,
    DEFAULT_N_BINS,
    MAX_N_BINS,
    MAX_SUBSTEPS,
    NARROW_SPREAD,
    PhaseState,
    bin_index,
    fraction_sum_tolerance,
    operation_count,
    validate_phase_state,
)
from maple_syrup.routing import EXPORT, INACTIVE, PIT_STORAGE
from maple_syrup.sediment_transport import (
    TransportError,
    TransportNetwork,
    TransportStep,
    TransportStepRejected,
)

__all__ = [
    "DEFAULT_DENSE_MAX_BYTES",
    "DEFAULT_IMPLEMENTATION",
    "ENTRY_BYTES",
    "IMPLEMENTATIONS",
    "ROW_PTR_ENTRY_BYTES",
    "ROW_SUM_CHUNK_ROWS",
    "ROW_SUM_EMULATION_VERIFIED_NUMPY_VERSIONS",
    "ROW_SUM_SELF_CHECK_BINS",
    "DenseConversionRefused",
    "HostNetworkEvidence",
    "RowSumEmulationMismatch",
    "SparseCapacityError",
    "SparseCharacteristicStep",
    "SparsePhaseState",
    "canonical_sparse_phase_state",
    "dense_phase_bytes",
    "empty_sparse_phase_state",
    "host_network_evidence",
    "rebind_sparse_phase_state",
    "reconcile_sparse_rows",
    "row_sum_emulation_self_check",
    "row_sum_policy",
    "row_sums_numpy",
    "slot_maps",
    "slot_maps_from_host_evidence",
    "sparse_characteristic_step",
    "sparse_from_dense",
    "sparse_state_bytes",
    "sparse_to_dense",
    "validate_sparse_phase_state",
]

_EPS = float(np.finfo(np.float64).eps)
# "numba": host NumPy network/arrays/state (default; CPU arithmetic unchanged).
# "cuda": CuPy network/arrays/state through characteristic_cuda_csr (CW-B4).
IMPLEMENTATIONS = ("numba", "cuda")
DEFAULT_IMPLEMENTATION = "numba"
# Per stored entry: fraction FP64 + position FP64 + bin uint8.
ENTRY_BYTES = 8 + 8 + 1
ROW_PTR_ENTRY_BYTES = 8
# `sparse_to_dense` / `sparse_from_dense` are SMALL-domain compatibility
# helpers; above this many bytes of dense phase arrays they refuse unless the
# caller raises `max_bytes` explicitly.
DEFAULT_DENSE_MAX_BYTES = 1 << 26
# Rows staged per NumPy row-sum reduction: a bounded `(512, B)` FP64 scratch
# (0.5 MB at B = 128), never an `(n, nc, B)` array.
ROW_SUM_CHUNK_ROWS = 512
# NumPy versions on which the ROOT independently recorded raw-byte equality of
# the compiled row-sum emulation with NumPy's reduction (30 probes, every
# supported bin edge, signed/positive wide magnitudes, empty rows; see
# agent_handoffs/tasks/conservative_water/cw_b2_root_extra_probes.json and
# csr_row_sum_profile.json). Provenance only: the running NumPy is ALWAYS
# re-checked by `row_sum_emulation_self_check` before production use.
ROW_SUM_EMULATION_VERIFIED_NUMPY_VERSIONS = ("2.5.3",)
# Bin counts probed by the runtime self-check: every power-of-two edge of the
# supported range 1..128, both neighbours of each edge, and 100.
ROW_SUM_SELF_CHECK_BINS = (1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 127, 128)
_ROW_SUM_SELF_CHECK_ROWS = 96  # divisible by 1, 2, 3 and 4 (dense shapes with 1..4 classes)
_ROW_SUM_SELF_CHECK: dict[str, Any] | None = None
_COURANT_REJECTION = "characteristic sediment Courant number v dt / (n_substeps dx) exceeds"


class SparseCapacityError(TransportError):
    """A substep needs more nonempty bins than the predeclared `max_nnz`.
    Raised before the write pass allocates anything; nothing was mutated and
    no mass was discarded. Retry with `max_nnz >= required_nnz`."""

    def __init__(self, required_nnz: int, max_nnz: int, substep: int, n_substeps: int):
        self.required_nnz = int(required_nnz)
        self.max_nnz = int(max_nnz)
        self.substep = int(substep)
        self.n_substeps = int(n_substeps)
        super().__init__(
            f"sparse characteristic transport needs nnz = {self.required_nnz} nonempty bins in substep "
            f"{self.substep + 1} of {self.n_substeps} but max_nnz = {self.max_nnz}; step refused before "
            "allocation (raise max_nnz; no mass was discarded)"
        )


class DenseConversionRefused(TransportError):
    """The dense `(ny, nx, nc, B)` phase arrays would exceed `max_bytes`;
    refused before any allocation."""

    def __init__(self, required_bytes: int, max_bytes: int):
        self.required_bytes = int(required_bytes)
        self.max_bytes = int(max_bytes)
        super().__init__(
            f"dense phase arrays need {self.required_bytes} bytes > max_bytes = {self.max_bytes}; the dense "
            "conversion is a small-domain compatibility helper and was refused before allocating"
        )


class RowSumEmulationMismatch(TransportError):
    """The compiled row-sum emulation does not reproduce the running NumPy's
    `sum(axis=-1)` raw bytes on the self-check fixture. Production refuses;
    there is no fallback to the NumPy path."""

    def __init__(self, numpy_version: str, n_bins: int, detail: str):
        self.numpy_version = str(numpy_version)
        self.n_bins = int(n_bins)
        super().__init__(
            f"compiled CSR row-sum emulation differs from NumPy {self.numpy_version} sum(axis=-1) at n_bins = "
            f"{self.n_bins} ({detail}); the sparse step refuses rather than falling back (bitwise correspondence "
            "with the dense path is not assumed across NumPy versions; re-verify the emulation for this NumPy)"
        )


def _real(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TransportError(f"{name} must be a real number, got {type(value).__name__}")
    return float(value)


def _strict_int(value: Any, name: str, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not (low <= int(value) <= high):
        raise TransportError(f"{name} must be an int in [{low}, {high}], got {value!r}")
    return int(value)


def sparse_state_bytes(n_rows: int, nnz: int) -> int:
    """Bytes of the CSR arrays (`row_ptr`, `bin`, `fraction`, `position_m`)
    for a preflight estimate; slot maps are reported separately."""
    return int(nnz) * ENTRY_BYTES + (int(n_rows) + 1) * ROW_PTR_ENTRY_BYTES


def dense_phase_bytes(shape: tuple[int, int], n_classes: int, n_bins: int) -> int:
    """Bytes of the dense `fraction` + `position_m` `(ny, nx, nc, B)` FP64 pair."""
    ny, nx = int(shape[0]), int(shape[1])
    return 2 * ny * nx * int(n_classes) * int(n_bins) * 8


# --- network slot maps ---------------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class _SlotMaps:
    cell_of_slot: np.ndarray  # (n_active,) ascending flat active cells
    slot_of_cell: np.ndarray  # (n,) slot or -1
    receiver_slot: np.ndarray  # (n_active,) slot of the receiver (the slot itself at outlets)
    outlet: np.ndarray  # (n_active,) bool


def _require_host_network(network: Any) -> None:
    if not isinstance(network, TransportNetwork):
        raise TransportError("network must be a TransportNetwork (use transport_network)")
    if network.xp is not np:
        raise TransportError(
            "implementation 'numba' of the sparse CSR characteristic reference runs on host NumPy networks only; "
            f"the network lives in {network.xp.__name__!r} and no host/device transfer or array fallback is "
            "performed (use implementation='cuda' with CuPy arrays and a CuPy phase state)"
        )


def _is_host_network(network: Any) -> bool:
    if not isinstance(network, TransportNetwork):
        raise TransportError("network must be a TransportNetwork (use transport_network)")
    return network.xp is np


def slot_maps_from_host_evidence(active_flat: np.ndarray, receiver_index_flat: np.ndarray,
                                 outlet_flat: np.ndarray) -> _SlotMaps:
    """Compact active-cell maps from HOST evidence (ascending flat order): `active_flat` `(n,)` bool,
    `receiver_index_flat` `(n,)` int64 (the receiver's flat index, the cell itself at outlets/inactive cells),
    `outlet_flat` `(n,)` bool. Shared by the host path and the CUDA binding (built once, uploaded once)."""
    active = np.asarray(active_flat, dtype=np.bool_).reshape(-1)
    n = int(active.size)
    cell_of_slot = np.flatnonzero(active).astype(np.int64)
    slot_of_cell = np.full(n, -1, dtype=np.int64)
    slot_of_cell[cell_of_slot] = np.arange(cell_of_slot.size, dtype=np.int64)
    receiver = np.asarray(receiver_index_flat, dtype=np.int64).reshape(-1)
    if receiver.size != n or np.any(receiver[cell_of_slot] < 0) or np.any(receiver[cell_of_slot] >= n):
        raise TransportError("internal error: receiver index evidence is inconsistent with the active cells")
    receiver_slot = slot_of_cell[receiver[cell_of_slot]]
    if np.any(receiver_slot < 0):
        raise TransportError("internal error: an active cell's receiver is inactive")
    outlet = np.asarray(outlet_flat, dtype=np.bool_).reshape(-1)[cell_of_slot]
    return _SlotMaps(cell_of_slot=cell_of_slot, slot_of_cell=slot_of_cell, receiver_slot=receiver_slot,
                     outlet=outlet)


@dataclass(frozen=True, eq=False)
class HostNetworkEvidence:
    """Validated HOST view of a network's connectivity: flat `active`, `outlet`, the normalised `receiver_index`
    (receiver flat index for internal cells; the cell ITSELF at outlets, inactive cells and declared terminals)
    and `terminal` (flat bool, or None when the network declares no terminal storage)."""

    active: np.ndarray
    outlet: np.ndarray
    receiver_index: np.ndarray
    terminal: np.ndarray | None


def host_network_evidence(network: TransportNetwork) -> HostNetworkEvidence:
    """Derive and validate the connectivity evidence from the network's HOST arrays (`active_host`,
    `outlet_host`, `receiver_host` and, when the network provides it, `terminal_host`).

    Receiver codes are accepted ONLY in their declared places: `EXPORT` at outlets, `INACTIVE` on inactive cells,
    `PIT_STORAGE` on cells flagged in `terminal_host` (an optional field of newer networks: a host bool
    `(ny, nx)` mask of terminal-storage cells, which must be active and not outlets; its receiver is normalised to
    the cell itself, so a terminal never sends). Without a `terminal_host` field the strict rule applies
    unchanged: every active non-outlet cell must have a non-negative active receiver. Any other negative encoding
    is refused explicitly."""
    if not isinstance(network, TransportNetwork):
        raise TransportError("network must be a TransportNetwork (use transport_network)")
    ny, nx = int(network.shape[0]), int(network.shape[1])
    n = ny * nx
    for name, dtype in (("active_host", np.bool_), ("outlet_host", np.bool_), ("receiver_host", np.int64)):
        array = getattr(network, name)
        if type(array) is not np.ndarray or array.dtype != dtype or tuple(array.shape) != (ny, nx):
            raise TransportError(f"network.{name} must be a host numpy.ndarray {np.dtype(dtype)}{(ny, nx)}")
    active = network.active_host.reshape(-1)
    outlet = network.outlet_host.reshape(-1)
    receiver = network.receiver_host.reshape(-1)
    terminal_host = getattr(network, "terminal_host", None)
    terminal = None
    if terminal_host is not None:
        if type(terminal_host) is not np.ndarray or terminal_host.dtype != np.bool_ or tuple(terminal_host.shape) != (ny, nx):
            raise TransportError(f"network.terminal_host must be a host numpy.ndarray bool{(ny, nx)}")
        terminal = terminal_host.reshape(-1)
        if np.any(terminal & ~active):
            raise TransportError("a terminal-storage cell must be active")
        if np.any(terminal & outlet):
            raise TransportError("a terminal-storage cell cannot be an outlet")
        if np.any(terminal & (receiver != PIT_STORAGE)):
            raise TransportError("a terminal-storage cell must carry the PIT_STORAGE receiver code")
        if np.any(active & ~terminal & (receiver == PIT_STORAGE)):
            raise TransportError("a PIT_STORAGE receiver on a cell not flagged in terminal_host is refused")
    if np.any(outlet & (receiver != EXPORT)):
        raise TransportError("an outlet cell must carry the EXPORT receiver code")
    if np.any(~active & (receiver != INACTIVE)):
        raise TransportError("an inactive cell must carry the INACTIVE receiver code")
    sending = active & ~outlet if terminal is None else active & ~outlet & ~terminal
    if np.any(sending & (receiver < 0)):
        raise TransportError("an active non-outlet cell has a negative receiver code that is not a declared terminal "
                             "(no implicit storage, filling or export)")
    if np.any(sending & (receiver >= n)):
        raise TransportError("a receiver index lies outside the grid")
    receiver_index = np.where(sending, receiver, np.arange(n, dtype=np.int64)).astype(np.int64)
    if np.any(sending & ~active[receiver_index]):
        raise TransportError("an internal cell's receiver is inactive")
    return HostNetworkEvidence(active=active, outlet=outlet, receiver_index=receiver_index, terminal=terminal)


def _host_terminal_compact(network: TransportNetwork, cell_of_slot: np.ndarray) -> np.ndarray | None:
    """Compact `(n_active,)` host terminal mask, or None for a strict network."""
    evidence = host_network_evidence(network)
    return None if evidence.terminal is None else np.ascontiguousarray(evidence.terminal[cell_of_slot])


def slot_maps(network: TransportNetwork) -> _SlotMaps:
    """Compact active-cell maps of a host network (ascending flat order), from the validated host evidence; the
    network's runtime `receiver_index` must agree with it (same values for strict networks; terminals are
    normalised to themselves by both)."""
    _require_host_network(network)
    evidence = host_network_evidence(network)
    runtime = np.asarray(network.receiver_index, dtype=np.int64).reshape(-1)
    if not np.array_equal(runtime, evidence.receiver_index):
        raise TransportError("network.receiver_index disagrees with the host receiver evidence (receiver_host / "
                             "outlet_host / active_host / terminal_host)")
    if not np.array_equal(np.asarray(network.active_flat, dtype=np.bool_).reshape(-1), evidence.active) or \
            not np.array_equal(np.asarray(network.outlet_flat, dtype=np.bool_).reshape(-1), evidence.outlet):
        raise TransportError("network.active_flat / outlet_flat disagree with the host masks")
    return slot_maps_from_host_evidence(evidence.active, evidence.receiver_index, evidence.outlet)


# --- state ---------------------------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class SparsePhaseState:
    """Nonempty-bin CSR phase state (module docstring). Arrays are frozen host NumPy."""

    row_ptr: np.ndarray
    bin: np.ndarray
    fraction: np.ndarray
    position_m: np.ndarray
    dx_m: float
    n_bins: int
    n_classes: int
    shape: tuple[int, int]
    cell_of_slot: np.ndarray
    slot_of_cell: np.ndarray
    network_sha256: str
    xp: ModuleType

    @property
    def n_active(self) -> int:
        return int(self.cell_of_slot.shape[0])

    @property
    def n_rows(self) -> int:
        return self.n_active * int(self.n_classes)

    @property
    def nnz(self) -> int:
        return int(self.fraction.shape[0])

    @property
    def n_nonempty_rows(self) -> int:
        """Host: free. CuPy: ONE counted scalar read (`maple.core.backend.to_int`)."""
        if self.xp is np:
            return int(np.count_nonzero(np.diff(self.row_ptr)))
        from maple.core.backend import to_int

        return to_int(self.xp.count_nonzero(self.xp.diff(self.row_ptr)))

    @property
    def nbytes(self) -> int:
        """CSR arrays plus the two slot maps (actual bytes in the state's namespace)."""
        return int(self.row_ptr.nbytes + self.bin.nbytes + self.fraction.nbytes + self.position_m.nbytes
                   + self.cell_of_slot.nbytes + self.slot_of_cell.nbytes)

    def occupancy(self) -> dict[str, Any]:
        """Cheap preflight report: counts and bytes, no dense arrays (two counted scalar reads on CuPy)."""
        counts = self.xp.diff(self.row_ptr)
        if self.xp is np:
            nonempty = int(np.count_nonzero(counts))
            max_bins = int(counts.max()) if counts.size else 0
        else:
            from maple.core.backend import to_int

            nonempty = to_int(self.xp.count_nonzero(counts))
            max_bins = to_int(counts.max()) if counts.size else 0
        return {
            "n_rows": self.n_rows,
            "n_nonempty_rows": nonempty,
            "nnz": self.nnz,
            "mean_bins_per_nonempty_row": (self.nnz / nonempty) if nonempty else 0.0,
            "max_bins_in_a_row": max_bins,
            "n_bins": int(self.n_bins),
            "csr_bytes": sparse_state_bytes(self.n_rows, self.nnz),
            "slot_map_bytes": int(self.cell_of_slot.nbytes + self.slot_of_cell.nbytes),
            "dense_equivalent_bytes": dense_phase_bytes(self.shape, self.n_classes, self.n_bins),
        }


def _row_of_entry(row_ptr: np.ndarray) -> np.ndarray:
    return np.repeat(np.arange(row_ptr.shape[0] - 1, dtype=np.int64), np.diff(row_ptr))


def row_sums_numpy(row_ptr: np.ndarray, bins: np.ndarray, masses: np.ndarray, n_bins: int) -> np.ndarray:
    """`(n_rows,)` per-row sum of `masses` as NumPy's `sum(axis=-1)` computes it
    on the dense `(..., B)` layout: rows are staged `ROW_SUM_CHUNK_ROWS` at a
    time into a bounded `(ROW_SUM_CHUNK_ROWS, B)` scratch holding exact zeros
    in the empty bins and reduced by NumPy itself (always the full scratch, so
    the reduced array has one fixed non-degenerate shape). Empty rows are
    exactly 0.0 (`0.0 + sum(zeros)`)."""
    nb = int(n_bins)
    n_rows = int(row_ptr.shape[0]) - 1
    out = np.zeros(n_rows, dtype=np.float64)
    scratch = np.zeros((ROW_SUM_CHUNK_ROWS, nb), dtype=np.float64)
    for start in range(0, n_rows, ROW_SUM_CHUNK_ROWS):
        stop = min(start + ROW_SUM_CHUNK_ROWS, n_rows)
        e0, e1 = int(row_ptr[start]), int(row_ptr[stop])
        if e1 == e0:
            continue
        local_rows = np.repeat(np.arange(stop - start, dtype=np.int64), np.diff(row_ptr[start:stop + 1]))
        local_bins = bins[e0:e1]
        scratch[local_rows, local_bins] = masses[e0:e1]
        out[start:stop] = scratch.sum(axis=-1)[:stop - start]
        scratch[local_rows, local_bins] = 0.0
    return out


def _self_check_rows(n_bins: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic CSR fixture for one bin count: `_ROW_SUM_SELF_CHECK_ROWS`
    rows at ~40 % occupancy with signed 33-decade magnitudes, plus a full row,
    an empty row, a single-entry row at the last bin and (B > 1) a two-entry
    row at the first and last bins. Returns `(dense, row_ptr, bins, values)`."""
    n_rows = _ROW_SUM_SELF_CHECK_ROWS
    magnitude = 10.0 ** rng.uniform(-30.0, 3.0, (n_rows, n_bins))
    sign = np.where(rng.uniform(size=(n_rows, n_bins)) < 0.3, -1.0, 1.0)
    dense = np.where(rng.uniform(size=(n_rows, n_bins)) < 0.4, magnitude * sign, 0.0)
    dense[0] = magnitude[0]
    dense[1] = 0.0
    dense[2] = 0.0
    dense[2, n_bins - 1] = 1.0 + 1.0e-15
    if n_bins > 1:
        dense[3] = 0.0
        dense[3, 0] = 1.0e3
        dense[3, n_bins - 1] = -1.0e-12
    rows, bins = np.nonzero(dense != 0.0)
    counts = np.bincount(rows, minlength=n_rows)
    row_ptr = np.zeros(n_rows + 1, dtype=np.int64)
    np.cumsum(counts, out=row_ptr[1:])
    return dense, row_ptr, bins.astype(np.uint8), np.ascontiguousarray(dense[rows, bins])


def row_sum_emulation_self_check(*, force: bool = False) -> dict[str, Any]:
    """Verify, on the running NumPy, that the compiled row-sum emulation
    (`characteristic_numba_csr.run_row_sums`) reproduces RAW BYTES of both
    `row_sums_numpy` and NumPy's dense `(rows/k, k, B).sum(axis=-1)` for
    `k = 1..4` on the fixture of `_self_check_rows` for every bin count in
    `ROW_SUM_SELF_CHECK_BINS`. Cached per process (`force=True` re-runs).
    Returns the provenance record; raises `RowSumEmulationMismatch` (no
    fallback) or `NumbaUnavailableError`."""
    global _ROW_SUM_SELF_CHECK
    if _ROW_SUM_SELF_CHECK is not None and not force:
        return dict(_ROW_SUM_SELF_CHECK)
    from maple_syrup import characteristic_numba_csr as kernels

    rng = np.random.default_rng(20261008)
    n_rows = _ROW_SUM_SELF_CHECK_ROWS
    for nb in ROW_SUM_SELF_CHECK_BINS:
        dense, row_ptr, bins, values = _self_check_rows(nb, rng)
        compiled = np.asarray(kernels.run_row_sums(row_ptr, bins, values, nb))
        if compiled.shape != (n_rows,) or compiled.dtype != np.float64:
            raise RowSumEmulationMismatch(np.__version__, nb, f"shape/dtype {compiled.shape} {compiled.dtype}")
        reference = row_sums_numpy(row_ptr, bins, values, nb)
        if compiled.tobytes() != reference.tobytes():
            raise RowSumEmulationMismatch(np.__version__, nb, "differs from the bounded NumPy reference")
        for k in (1, 2, 3, 4):
            dense_ref = dense.reshape(n_rows // k, k, nb).sum(axis=-1).reshape(-1)
            if compiled.tobytes() != dense_ref.tobytes():
                raise RowSumEmulationMismatch(np.__version__, nb, f"differs from the dense reduction with {k} classes")
    try:
        import numba

        numba_version = str(numba.__version__)
    except ImportError:  # unreachable after run_row_sums succeeded; kept explicit
        numba_version = None
    _ROW_SUM_SELF_CHECK = {
        "numpy": str(np.__version__),
        "numba": numba_version,
        "bins": tuple(ROW_SUM_SELF_CHECK_BINS),
        "rows_per_bin_count": n_rows,
        "byte_exact": True,
        "root_verified_numpy_versions": tuple(ROW_SUM_EMULATION_VERIFIED_NUMPY_VERSIONS),
        "running_numpy_root_verified": str(np.__version__) in ROW_SUM_EMULATION_VERIFIED_NUMPY_VERSIONS,
    }
    return dict(_ROW_SUM_SELF_CHECK)


def row_sum_policy() -> dict[str, Any]:
    """Provenance of the production row-sum path: `"compiled"` (Numba emulation,
    self-checked against the running NumPy), the reference name, and the
    cached self-check record (`None` until the first production use)."""
    return {
        "production": "compiled",
        "reference": "row_sums_numpy",
        "self_check": None if _ROW_SUM_SELF_CHECK is None else dict(_ROW_SUM_SELF_CHECK),
        "root_verified_numpy_versions": tuple(ROW_SUM_EMULATION_VERIFIED_NUMPY_VERSIONS),
        "running_numpy": str(np.__version__),
    }


def empty_sparse_phase_state(network: TransportNetwork, n_classes: int,
                             n_bins: int = DEFAULT_N_BINS) -> SparsePhaseState:
    """Canonical empty state (no mobile mass anywhere): `nnz = 0`, in the network's namespace (host arrays for a
    NumPy network; device arrays bound to the network's owned slot maps for a CuPy network)."""
    from maple.core.backend import freeze

    if not _is_host_network(network):
        from maple_syrup import characteristic_cuda_csr

        return characteristic_cuda_csr.empty_cuda_sparse_phase_state(network, n_classes, n_bins)
    maps = slot_maps(network)
    nc = _strict_int(n_classes, "n_classes", 1, 1 << 30)
    nb = _strict_int(n_bins, "n_bins", 1, MAX_N_BINS)
    n_rows = maps.cell_of_slot.size * nc
    return SparsePhaseState(
        row_ptr=freeze(np.zeros(n_rows + 1, dtype=np.int64)),
        bin=freeze(np.zeros(0, dtype=np.uint8)),
        fraction=freeze(np.zeros(0, dtype=np.float64)),
        position_m=freeze(np.zeros(0, dtype=np.float64)),
        dx_m=network.dx_m, n_bins=nb, n_classes=nc, shape=(int(network.shape[0]), int(network.shape[1])),
        cell_of_slot=freeze(maps.cell_of_slot), slot_of_cell=freeze(maps.slot_of_cell),
        network_sha256=network.graph_input_sha256, xp=np,
    )


def canonical_sparse_phase_state(network: TransportNetwork, mobile_kg: Any, n_bins: int = DEFAULT_N_BINS) -> SparsePhaseState:
    """The canonical CSR partition of an existing mobile field: one stored bin (bin 0, fraction 1, position 0,
    the upstream face) for every cell/class whose mobile mass is positive, nothing for the others. Built
    DIRECTLY from the mobile mask in the network's namespace -- no `(ny, nx, nc, B)` array, no dense detour. The
    result is validated against `mobile_kg` (finite, >= 0, zero on inactive cells) before it is returned. ONE
    counted scalar read (`nnz`) on CuPy."""
    from maple.core.backend import freeze, to_int

    host = _is_host_network(network)
    nb = _strict_int(n_bins, "n_bins", 1, MAX_N_BINS)
    if host:
        nc = _check_mobile(mobile_kg, network)  # type, shape, dtype, finiteness, sign, inactive cells
        template = empty_sparse_phase_state(network, nc, nb)
        xp = np
    else:
        from maple_syrup import characteristic_cuda_csr

        cp = characteristic_cuda_csr._cupy()
        binding = characteristic_cuda_csr.bind_network(network)
        nc = characteristic_cuda_csr._check_mobile_structure(cp, mobile_kg, binding)
        template = characteristic_cuda_csr.empty_cuda_sparse_phase_state(network, nc, nb)
        xp = cp
    n = int(network.shape[0]) * int(network.shape[1])
    positive = (mobile_kg.reshape(n, nc)[template.cell_of_slot] > 0.0).reshape(-1)
    counts = positive.astype(np.int64)
    row_ptr = xp.zeros(template.n_rows + 1, dtype=np.int64)
    xp.cumsum(counts, out=row_ptr[1:])
    nnz = int(row_ptr[-1]) if host else to_int(row_ptr[-1])
    state = dataclasses.replace(
        template,
        row_ptr=freeze(row_ptr),
        bin=freeze(xp.zeros(nnz, dtype=np.uint8)),
        fraction=freeze(xp.ones(nnz, dtype=np.float64)),
        position_m=freeze(xp.zeros(nnz, dtype=np.float64)),
    )
    validate_sparse_phase_state(state, mobile_kg, network)
    return state


def reconcile_sparse_rows(phase: SparsePhaseState, drop_rows: Any, add_canonical_rows: Any) -> SparsePhaseState:
    """Row-level reconciliation of a CSR state in its own namespace: every stored bin of a row flagged in
    `drop_rows` `(n_rows,)` bool is removed (the row becomes empty), and every row flagged in
    `add_canonical_rows` `(n_rows,)` bool receives the single canonical bin (bin 0, fraction 1, position 0).
    A row may not be both, and a canonical row may not be added to a row that still stores bins (refused;
    nothing is densified). Kept entries keep their relative order, so bins stay strictly increasing per row.
    Namespace-generic; the compaction uses boolean indexing / `flatnonzero`, which on CuPy synchronise to size
    the new arrays (two implicit size reads, not counted by MAPLE's transfer counters; the caller invokes this
    only when a row actually changes). The metadata (shape, dx, B, classes, slot maps, digest) is unchanged."""
    from maple.core.backend import freeze, read_flags, true_flag

    if not isinstance(phase, SparsePhaseState):
        raise TransportError("phase must be a SparsePhaseState")
    xp = phase.xp
    n_rows = phase.n_rows
    for name, mask in (("drop_rows", drop_rows), ("add_canonical_rows", add_canonical_rows)):
        if not hasattr(mask, "shape") or tuple(mask.shape) != (n_rows,) or mask.dtype != np.bool_:
            raise TransportError(f"{name} must be a bool array of shape {(n_rows,)}")
    row_ptr = phase.row_ptr
    counts = xp.diff(row_ptr)
    counts_kept = xp.where(drop_rows, 0, counts)
    bad_overlap, bad_add = read_flags([true_flag(drop_rows & add_canonical_rows),
                                       true_flag(add_canonical_rows & (counts_kept > 0))])
    if bad_overlap:
        raise TransportError("a row cannot be both dropped and given a canonical bin")
    if bad_add:
        raise TransportError("a canonical bin cannot be added to a row that still stores bins")
    nnz = phase.nnz
    rows = xp.searchsorted(row_ptr, xp.arange(nnz, dtype=np.int64), side="right") - 1 if nnz else xp.zeros(0, dtype=np.int64)
    keep_entry = ~drop_rows[rows] if nnz else xp.zeros(0, dtype=np.bool_)
    kept_idx = xp.flatnonzero(keep_entry)
    add_idx = xp.flatnonzero(add_canonical_rows)
    new_counts = counts_kept + add_canonical_rows.astype(np.int64)
    row_ptr_new = xp.zeros(n_rows + 1, dtype=np.int64)
    xp.cumsum(new_counts, out=row_ptr_new[1:])
    n_kept, n_add = int(kept_idx.shape[0]), int(add_idx.shape[0])
    nnz_new = n_kept + n_add
    bins_new = xp.zeros(nnz_new, dtype=np.uint8)
    fraction_new = xp.ones(nnz_new, dtype=np.float64)
    position_new = xp.zeros(nnz_new, dtype=np.float64)
    if n_kept:
        rows_k = rows[kept_idx]
        dest = row_ptr_new[rows_k] + (kept_idx - row_ptr[rows_k])  # the rank within the row is unchanged
        bins_new[dest] = phase.bin[kept_idx]
        fraction_new[dest] = phase.fraction[kept_idx]
        position_new[dest] = phase.position_m[kept_idx]
    # inserted canonical bins: bin 0, fraction 1, position 0 are the array defaults at row_ptr_new[add_idx]
    return dataclasses.replace(phase, row_ptr=freeze(row_ptr_new), bin=freeze(bins_new), fraction=freeze(fraction_new),
                               position_m=freeze(position_new))


def rebind_sparse_phase_state(phase: SparsePhaseState, network: TransportNetwork) -> SparsePhaseState:
    """EXPLICIT rebind of a phase state to a refreshed network (e.g. routing rebuilt after a bed change) in the
    SAME namespace, with the same shape, `dx` and ACTIVE cell set: returns the same bins/fractions/positions bound
    to the new graph digest and the network's slot maps. Anything else (namespace, shape, dx, active set) is
    refused; nothing is transferred. Positions keep their meaning as distances from the cell's upstream face
    along its (possibly new) outflow direction -- the receiver of a packet is looked up on the new network at the
    next step, which is the documented approximation of a rebind, not a re-projection."""
    if not isinstance(phase, SparsePhaseState):
        raise TransportError("phase must be a SparsePhaseState")
    host = _is_host_network(network)
    if host != (phase.xp is np):
        raise TransportError("phase and network namespaces differ; rebind performs no transfer")
    if tuple(phase.shape) != tuple(network.shape):
        raise TransportError(f"phase shape {tuple(phase.shape)} differs from the network shape {network.shape}")
    if float(phase.dx_m) != network.dx_m:
        raise TransportError(f"phase dx_m {phase.dx_m} differs from the network dx_m {network.dx_m}")
    if host:
        from maple.core.backend import freeze

        maps = slot_maps(network)
        if maps.cell_of_slot.size != phase.n_active or not np.array_equal(maps.cell_of_slot, phase.cell_of_slot):
            raise TransportError("the network's active cells differ from the phase state's slots; a rebind needs "
                                 "the same active set (rows are indexed by active slot)")
        return dataclasses.replace(phase, cell_of_slot=freeze(maps.cell_of_slot), slot_of_cell=freeze(maps.slot_of_cell),
                                   network_sha256=network.graph_input_sha256)
    from maple.core.backend import read_flags, true_flag

    from maple_syrup import characteristic_cuda_csr

    binding = characteristic_cuda_csr.bind_network(network)
    if binding.n_active != phase.n_active:
        raise TransportError("the network's active cells differ from the phase state's slots; a rebind needs the "
                             "same active set (rows are indexed by active slot)")
    if phase.cell_of_slot is not binding.cell_of_slot and read_flags([true_flag(phase.cell_of_slot != binding.cell_of_slot)])[0]:
        raise TransportError("the network's active cells differ from the phase state's slots; a rebind needs the "
                             "same active set (rows are indexed by active slot)")
    return dataclasses.replace(phase, cell_of_slot=binding.cell_of_slot, slot_of_cell=binding.slot_of_cell,
                               network_sha256=binding.graph_input_sha256)


def _require_host_arrays(named: dict[str, Any], shape: tuple[int, ...], dtype) -> None:
    for name, array in named.items():
        if not isinstance(array, np.ndarray):
            raise TransportError(f"{name} must be a host NumPy array (CSR reference namespace 'numpy'), "
                                 f"got {type(array).__name__}")
        if tuple(array.shape) != tuple(shape):
            raise TransportError(f"{name} shape {tuple(array.shape)} != {tuple(shape)}")
        if array.dtype != dtype:
            raise TransportError(f"{name} must be {np.dtype(dtype)}, got {array.dtype}")


def _check_structure(phase: Any, network: TransportNetwork, n_classes: int, maps: _SlotMaps) -> None:
    """Immediate structural refusals (types, dtypes, shapes, bindings)."""
    if not isinstance(phase, SparsePhaseState):
        raise TransportError("phase must be a SparsePhaseState (use empty_sparse_phase_state)")
    if phase.xp is not np:
        raise TransportError(f"sparse phase namespace {getattr(phase.xp, '__name__', phase.xp)!r} is not 'numpy'")
    _strict_int(phase.n_bins, "phase.n_bins", 1, MAX_N_BINS)
    if _real(phase.dx_m, "phase.dx_m") != network.dx_m:
        raise TransportError(f"phase dx_m {phase.dx_m} differs from the network dx_m {network.dx_m}")
    if tuple(phase.shape) != tuple(network.shape):
        raise TransportError(f"phase shape {tuple(phase.shape)} differs from the network shape {network.shape}")
    if _strict_int(phase.n_classes, "phase.n_classes", 1, 1 << 30) != n_classes:
        raise TransportError(f"phase n_classes {phase.n_classes} differs from the mobile field's {n_classes}")
    if phase.network_sha256 != network.graph_input_sha256:
        raise TransportError("phase state was built for a different routing graph (graph digest mismatch)")
    _require_host_arrays({"phase.cell_of_slot": phase.cell_of_slot}, (maps.cell_of_slot.size,), np.int64)
    _require_host_arrays({"phase.slot_of_cell": phase.slot_of_cell}, (maps.slot_of_cell.size,), np.int64)
    if not np.array_equal(phase.cell_of_slot, maps.cell_of_slot) or \
            not np.array_equal(phase.slot_of_cell, maps.slot_of_cell):
        raise TransportError("phase slot maps differ from the network's active cells")
    n_rows = maps.cell_of_slot.size * n_classes
    _require_host_arrays({"phase.row_ptr": phase.row_ptr}, (n_rows + 1,), np.int64)
    if phase.row_ptr[0] != 0:
        raise TransportError("phase.row_ptr[0] must be 0")
    if np.any(np.diff(phase.row_ptr) < 0):
        raise TransportError("phase.row_ptr must be non-decreasing")
    nnz = int(phase.row_ptr[-1])
    _require_host_arrays({"phase.bin": phase.bin}, (nnz,), np.uint8)
    _require_host_arrays({"phase.fraction": phase.fraction, "phase.position_m": phase.position_m}, (nnz,),
                         np.float64)


def _check_values(phase: SparsePhaseState, mobile_compact: np.ndarray) -> None:
    """Value refusals on a structurally valid state; `mobile_compact` is the
    `(n_active, nc)` host mobile field."""
    nb = int(phase.n_bins)
    dx = float(phase.dx_m)
    counts = np.diff(phase.row_ptr)
    occupied_rows = counts > 0
    positive = mobile_compact.reshape(-1) > 0.0
    if np.any(occupied_rows & ~positive):
        raise TransportError("phase state stores bins on a row whose mobile total is 0 (canonical empty rows "
                             "store nothing)")
    if np.any(positive & ~occupied_rows):
        raise TransportError("phase state has no bins on a row whose mobile total is positive")
    nnz = int(phase.row_ptr[-1])
    if nnz == 0:
        return
    bins = phase.bin
    f = phase.fraction
    x = phase.position_m
    rows = _row_of_entry(phase.row_ptr)
    if np.any(bins >= nb):
        raise TransportError(f"phase.bin must be < n_bins = {nb}")
    same_row = rows[1:] == rows[:-1]
    if np.any(same_row & ~(bins[1:] > bins[:-1])):
        raise TransportError("phase.bin must be strictly increasing within each row")
    if not np.all(np.isfinite(f)):
        raise TransportError("phase.fraction must be finite everywhere")
    if not np.all(np.isfinite(x)):
        raise TransportError("phase.position_m must be finite everywhere")
    if np.any(f <= 0.0):
        raise TransportError("phase.fraction must be > 0 on every stored bin (empty bins are not stored)")
    if np.any(x < 0.0):
        raise TransportError("phase.position_m must be >= 0 everywhere")
    if np.any(x >= dx):
        raise TransportError("phase.position_m must be < dx everywhere")
    if np.any(bin_index(x, dx, nb, np) != bins.astype(np.int64)):
        raise TransportError("phase.position_m must lie inside its own bin")
    sums = np.bincount(rows, weights=f, minlength=counts.size)
    if np.any(occupied_rows & (np.abs(sums - 1.0) > fraction_sum_tolerance(nb))):
        raise TransportError("phase.fraction must sum to 1 within the declared bound on every nonempty row")


def _check_mobile(mobile_kg: Any, network: TransportNetwork) -> int:
    if not isinstance(mobile_kg, np.ndarray) or mobile_kg.ndim != 3 or mobile_kg.shape[-1] < 1:
        raise TransportError("mobile_kg must be a host (ny, nx, nc) NumPy array")
    ny, nx = network.shape
    nc = int(mobile_kg.shape[-1])
    _require_host_arrays({"mobile_kg": mobile_kg}, (ny, nx, nc), np.float64)
    if not np.all(np.isfinite(mobile_kg)):
        raise TransportError("mobile_kg must be finite everywhere")
    if np.any(mobile_kg < 0.0):
        raise TransportError("mobile_kg must be >= 0 everywhere")
    inactive = ~np.asarray(network.active_host, dtype=np.bool_)
    if np.any(mobile_kg[inactive] != 0.0):
        raise TransportError("mobile_kg must be 0 on inactive cells")
    return nc


def validate_sparse_phase_state(phase: Any, mobile_kg: Any, network: Any) -> None:
    """Raise `TransportError` unless `mobile_kg` is a valid `(ny, nx, nc)`
    FP64 mobile field (finite, >= 0, 0 on inactive cells) and `phase` a
    valid CSR partition of it on `network` (module docstring). Host network:
    host checks; CuPy network: device flags resolved in one read
    (`characteristic_cuda_csr.validate_cuda_sparse_phase_state`)."""
    if not _is_host_network(network):
        from maple_syrup import characteristic_cuda_csr

        characteristic_cuda_csr.validate_cuda_sparse_phase_state(phase, mobile_kg, network)
        return
    nc = _check_mobile(mobile_kg, network)
    maps = slot_maps(network)
    _check_structure(phase, network, nc, maps)
    compact = mobile_kg.reshape(-1, nc)[maps.cell_of_slot]
    _check_values(phase, compact)


# --- dense compatibility helpers (SMALL domains only) ---------------------------------------------------
def sparse_to_dense(phase: SparsePhaseState, *, max_bytes: int = DEFAULT_DENSE_MAX_BYTES) -> PhaseState:
    """Dense `PhaseState` view of a (structurally valid) sparse state, for small
    domains and tests. Refuses with `DenseConversionRefused` BEFORE allocating
    when the two `(ny, nx, nc, B)` arrays exceed `max_bytes`."""
    from maple.core.backend import freeze

    if not isinstance(phase, SparsePhaseState):
        raise TransportError("phase must be a SparsePhaseState")
    limit = _strict_int(max_bytes, "max_bytes", 0, 1 << 62)
    if phase.xp is not np:
        from maple_syrup import characteristic_cuda_csr

        return characteristic_cuda_csr.sparse_to_dense_cuda(phase, max_bytes=limit)
    ny, nx = phase.shape
    nc, nb = int(phase.n_classes), int(phase.n_bins)
    required = dense_phase_bytes(phase.shape, nc, nb)
    if required > limit:
        raise DenseConversionRefused(required, limit)
    n = ny * nx
    fraction = np.zeros((n, nc, nb), dtype=np.float64)
    fraction[:, :, 0] = 1.0
    position = np.zeros((n, nc, nb), dtype=np.float64)
    counts = np.diff(phase.row_ptr)
    nonempty = np.flatnonzero(counts > 0)
    if nonempty.size:
        fraction[phase.cell_of_slot[nonempty // nc], nonempty % nc, 0] = 0.0
        rows = _row_of_entry(phase.row_ptr)
        cells = phase.cell_of_slot[rows // nc]
        classes = rows % nc
        bins = phase.bin.astype(np.int64)
        fraction[cells, classes, bins] = phase.fraction
        position[cells, classes, bins] = phase.position_m
    return PhaseState(fraction=freeze(fraction.reshape(ny, nx, nc, nb)),
                      position_m=freeze(position.reshape(ny, nx, nc, nb)), dx_m=float(phase.dx_m), n_bins=nb, xp=np)


def sparse_from_dense(phase: PhaseState, mobile_kg: Any, network: TransportNetwork, *,
                      max_bytes: int = DEFAULT_DENSE_MAX_BYTES) -> SparsePhaseState:
    """CSR state of a dense `PhaseState` (validated with the dense
    `validate_phase_state`) partitioning `mobile_kg`. Reads dense arrays of
    the caller's size and one dense boolean mask, hence the same small-domain
    budget as `sparse_to_dense`.

    Resource contract: the byte cap is evaluated from metadata only (network
    shape, class count from the mobile field's shape, `phase.n_bins`) and
    refused with `DenseConversionRefused` BEFORE the dense value validator
    runs -- that validator itself creates full-size masks and bin-index
    temporaries. Ordinary (in-budget) inputs are then fully validated before
    anything is converted."""
    from maple.core.backend import freeze

    if not _is_host_network(network):
        from maple_syrup import characteristic_cuda_csr

        return characteristic_cuda_csr.sparse_from_dense_cuda(
            phase, mobile_kg, network, max_bytes=_strict_int(max_bytes, "max_bytes", 0, 1 << 62))
    if not isinstance(phase, PhaseState):
        raise TransportError("phase must be a dense PhaseState")
    limit = _strict_int(max_bytes, "max_bytes", 0, 1 << 62)
    # Metadata/type checks only (no value scan, no dense temporaries) ...
    if not isinstance(mobile_kg, np.ndarray) or mobile_kg.ndim != 3 or mobile_kg.shape[-1] < 1:
        raise TransportError("mobile_kg must be a host (ny, nx, nc) NumPy array")
    nc = int(mobile_kg.shape[-1])
    nb = _strict_int(phase.n_bins, "phase.n_bins", 1, MAX_N_BINS)
    # ... then the cap, before the dense validator can allocate anything.
    required = dense_phase_bytes(network.shape, nc, nb)
    if required > limit:
        raise DenseConversionRefused(required, limit)
    validate_phase_state(phase, mobile_kg, network)
    maps = slot_maps(network)
    n = network.n_cells
    F = np.asarray(phase.fraction).reshape(n, nc, nb)[maps.cell_of_slot]
    X = np.asarray(phase.position_m).reshape(n, nc, nb)[maps.cell_of_slot]
    M = np.asarray(mobile_kg).reshape(n, nc)[maps.cell_of_slot]
    stored = (M > 0.0)[..., None] & (F > 0.0)
    slots, classes, bins = np.nonzero(stored)  # C order: rows ascending, bins ascending within a row
    rows = slots * nc + classes
    n_rows = maps.cell_of_slot.size * nc
    counts = np.bincount(rows, minlength=n_rows)
    row_ptr = np.zeros(n_rows + 1, dtype=np.int64)
    np.cumsum(counts, out=row_ptr[1:])
    state = SparsePhaseState(
        row_ptr=freeze(row_ptr), bin=freeze(bins.astype(np.uint8)),
        fraction=freeze(np.ascontiguousarray(F[slots, classes, bins])),
        position_m=freeze(np.ascontiguousarray(X[slots, classes, bins])),
        dx_m=network.dx_m, n_bins=nb, n_classes=nc, shape=(int(network.shape[0]), int(network.shape[1])),
        cell_of_slot=freeze(maps.cell_of_slot), slot_of_cell=freeze(maps.slot_of_cell),
        network_sha256=network.graph_input_sha256, xp=np,
    )
    _check_values(state, M)
    return state


# --- result ---------------------------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class SparseCharacteristicStep:
    """`transport` is the unchanged `TransportStep`; `phase_after` partitions
    `mobile_remaining_kg = T - Dep - E`. `mobile_before_kg = M + P`;
    `arrival_deposition_kg` the deposition booked in a cell by mass that
    entered it this step; `crossing_kg = Out + E` per source cell. `nnz_after`
    is the returned state's nonempty-bin count and `peak_nnz` the largest
    scanned `nnz` of any substep (the capacity actually needed)."""

    transport: TransportStep
    phase_after: SparsePhaseState
    mobile_before_kg: Any
    mobile_remaining_kg: Any
    arrival_deposition_kg: Any
    crossing_kg: Any
    n_bins: int
    implementation: str
    nnz_after: int
    peak_nnz: int
    # CUDA only: launch shape, compiled kernel attributes, local-scratch estimate vs compiled bytes, row-sum
    # self-check record and the step's counted transfers. None on the Numba path.
    device_report: dict[str, Any] | None = None


# --- CW-B93 PRIVATE accounting retention (CUDA CSR, paired production accounting path only) ----------------------
@dataclass(frozen=True, eq=False)
class _TransportAccountingStep:
    """CW-B93 PRIVATE: exactly the transport-step objects the paired production accounting path reads -- MAPLE's
    deposit demand (post-transfer pool, deposition / export requests, all four face fields), the loop's
    `TransportAccounting` record and phase reconciliation. The SAME arrays the complete `TransportStep` would carry,
    computed by the same kernels and expressions; the return-only diagnostics (decay / settled deposition, internal
    in / out, divergence, per-class before / after totals) are not part of it. Never a public result: every field is
    present (no None placeholders) and it is not a `TransportStep`, so a public consumer refuses it."""

    dt_s: float
    n_substeps: int
    mobile_after_transfer_kg: Any
    deposition_request_kg: Any
    export_request_kg: Any
    x_face_gross_kg: Any
    y_face_gross_kg: Any
    x_face_net_kg: Any
    y_face_net_kg: Any
    deposition_request_by_class_kg: Any
    export_request_by_class_kg: Any
    budget_residual_by_class_kg: Any
    budget_tolerance_by_class_kg: Any
    max_courant: Any
    max_decay_exponent: Any
    max_cell_balance_residual_kg: Any


@dataclass(frozen=True, eq=False)
class _SparseAccountingStep:
    """CW-B93 PRIVATE counterpart of `SparseCharacteristicStep` for the paired production accounting path: the
    private transport record plus the kernel's remaining pool and CSR phase (what reconciliation reads) and the
    step's scalar records. `mobile_before_kg`, `arrival_deposition_kg` and `crossing_kg` (return-only) are absent."""

    transport: _TransportAccountingStep
    phase_after: SparsePhaseState
    mobile_remaining_kg: Any
    n_bins: int
    implementation: str
    nnz_after: int
    peak_nnz: int
    device_report: dict[str, Any]


def _sparse_characteristic_accounting_step(network, mobile_before_kg, phase, pickup_kg, sediment_velocity_m_s,
                                           deposition_rate_per_m, settle_mask, dt_s, *, n_substeps: int = 1,
                                           courant_max: float = DEFAULT_COURANT_MAX, max_nnz: int | None = None):
    """CW-B93 PRIVATE: `sparse_characteristic_step(..., implementation="cuda")` with private accounting retention --
    the SAME `characteristic_cuda_csr` core (validation, kernels, sums, flags, packet order, reads), returning
    `_SparseAccountingStep`. CUDA only; every other implementation uses the public step."""
    from maple_syrup import characteristic_cuda_csr

    return characteristic_cuda_csr._cuda_sparse_accounting_step(
        network, mobile_before_kg, phase, pickup_kg, sediment_velocity_m_s, deposition_rate_per_m, settle_mask,
        dt_s, n_substeps=n_substeps, courant_max=courant_max, max_nnz=max_nnz)


# --- step -----------------------------------------------------------------------------------------------
def sparse_characteristic_step(
    network: TransportNetwork,
    mobile_before_kg: Any,
    phase: SparsePhaseState,
    pickup_kg: Any,
    sediment_velocity_m_s: Any,
    deposition_rate_per_m: Any,
    settle_mask: Any,
    dt_s: float,
    *,
    n_substeps: int = 1,
    courant_max: float = DEFAULT_COURANT_MAX,
    implementation: str = DEFAULT_IMPLEMENTATION,
    max_nnz: int | None = None,
    max_launched_threads: int | None = None,
    block_threads: int | None = None,
) -> SparseCharacteristicStep:
    """Apply the characteristic operator for `dt_s` on the CSR state (module docstring).

    `implementation="cuda"` delegates to `characteristic_cuda_csr.cuda_sparse_step` (CuPy network, arrays and
    phase; `max_launched_threads` / `block_threads` bound the persistent grid-stride launch, defaults
    `characteristic_cuda_csr.DEFAULT_MAX_LAUNCHED_THREADS` / `BLOCK_THREADS`). Those two options are CUDA-only
    and refused on the Numba path."""
    from maple.core.backend import (
        freeze,
        pairwise_bound_scale_factor,
        pairwise_sum_over_leading_axes,
        scatter_add,
    )

    from maple_syrup import characteristic_numba_csr as kernels

    if implementation not in IMPLEMENTATIONS:
        raise TransportError(f"implementation must be one of {IMPLEMENTATIONS} (no array path exists for the sparse "
                             f"CSR reference), got {implementation!r}")
    if implementation == "cuda":
        from maple_syrup import characteristic_cuda_csr

        options: dict[str, Any] = {}
        if max_launched_threads is not None:
            options["max_launched_threads"] = max_launched_threads
        if block_threads is not None:
            options["block_threads"] = block_threads
        return characteristic_cuda_csr.cuda_sparse_step(
            network, mobile_before_kg, phase, pickup_kg, sediment_velocity_m_s, deposition_rate_per_m, settle_mask,
            dt_s, n_substeps=n_substeps, courant_max=courant_max, max_nnz=max_nnz, **options)
    if max_launched_threads is not None or block_threads is not None:
        raise TransportError("max_launched_threads / block_threads are CUDA launch options; implementation 'numba' "
                             "does not accept them")
    _require_host_network(network)
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
    cap = None if max_nnz is None else _strict_int(max_nnz, "max_nnz", 0, 1 << 62)
    nc = int(mobile_before_kg.shape[-1]) if isinstance(mobile_before_kg, np.ndarray) and mobile_before_kg.ndim == 3 \
        else 0
    if nc < 1:
        raise TransportError("mobile_before_kg must be a host (ny, nx, nc) array with nc >= 1, got "
                             f"{getattr(mobile_before_kg, 'shape', type(mobile_before_kg).__name__)}")
    shape3 = (ny, nx, nc)
    floats = {"mobile_before_kg": mobile_before_kg, "pickup_kg": pickup_kg,
              "sediment_velocity_m_s": sediment_velocity_m_s, "deposition_rate_per_m": deposition_rate_per_m}
    _require_host_arrays(floats, shape3, np.float64)
    _require_host_arrays({"settle_mask": settle_mask}, shape3, np.bool_)
    for name, array in floats.items():
        if not np.all(np.isfinite(array)):
            raise TransportError(f"{name} must be finite everywhere")
        if np.any(array < 0.0):
            raise TransportError(f"{name} must be >= 0 everywhere")
    maps = slot_maps(network)
    _check_structure(phase, network, nc, maps)
    nb = int(phase.n_bins)
    active = np.asarray(network.active_host, dtype=np.bool_).reshape(-1)
    outlet = np.asarray(network.outlet_flat, dtype=np.bool_).reshape(-1)
    inactive3 = (~active)[:, None]
    M = mobile_before_kg.reshape(n, nc)
    P = pickup_kg.reshape(n, nc)
    V = sediment_velocity_m_s.reshape(n, nc)
    R = deposition_rate_per_m.reshape(n, nc)
    S = settle_mask.reshape(n, nc)
    if np.any(inactive3 & (M != 0.0)):
        raise TransportError("mobile_before_kg must be 0 on inactive cells")
    if np.any(inactive3 & (P != 0.0)):
        raise TransportError("pickup_kg must be 0 on inactive cells")
    if np.any(inactive3 & (V != 0.0)):
        raise TransportError("sediment_velocity_m_s must be 0 on inactive cells")
    idx = maps.cell_of_slot
    n_active = idx.size
    n_rows = n_active * nc
    Mc = np.ascontiguousarray(M[idx])
    _check_values(phase, Mc)

    dx = network.dx_m
    dt_sub = dt / n_sub
    with np.errstate(all="ignore"):
        a = V * (dt_sub / dx)
        max_courant = np.max(np.where(active[:, None], a, 0.0))
        if not np.all(np.isfinite(a)):
            raise TransportError("characteristic sediment Courant number overflowed FP64")
        if np.any(active[:, None] & (a > cr_max)):
            raise TransportStepRejected(f"{_COURANT_REJECTION} courant_max = {cr_max}; step rejected (retry with a "
                                        "smaller dt or more substeps)")
        exponent = (V * R) * dt_sub
        max_exponent = np.max(np.where(active[:, None], exponent, 0.0))
        if not np.all(np.isfinite(exponent)):
            raise TransportError("deposition hazard v r dt overflowed FP64")

    # Input validation is complete; the kernels may now assume the invariants.
    Pc = np.ascontiguousarray(P[idx])
    Vc = np.ascontiguousarray(V[idx])
    Rc = np.ascontiguousarray(R[idx])
    Sc = np.ascontiguousarray(S[idx])
    # Terminal rule (networks declaring terminal storage only), the same explicit refusal as the C2 operators: the
    # caller's settle mask must be True for EVERY grain class at a terminal cell, so the whole mobile load (own
    # pool, pickup, arrivals) settles through the deposition request and nothing is sent (its receiver is itself).
    # A mask missing any class is refused before any kernel work; it is never coerced. Strict networks: unchanged.
    terminal_c = _host_terminal_compact(network, idx)
    if terminal_c is not None and np.any(terminal_c[:, None] & ~Sc):
        raise TransportError("settle_mask must be True for every grain class at terminal-storage cells (the mobile "
                             "load settles there and nothing is sent); a mask missing a class is refused, not coerced")
    no_pickup = np.zeros((n_active, nc), dtype=np.float64)
    count_pass = kernels.run_count_pass  # compiles lazily; NumbaUnavailableError without Numba
    kernels.compiled_kernels()
    # Production row sums are the compiled emulation; it must have reproduced
    # the running NumPy's reduction bytes in this process (no fallback).
    row_sum_emulation_self_check()
    with np.errstate(all="ignore"):
        row_ptr = np.ascontiguousarray(phase.row_ptr, dtype=np.int64)
        bins = np.ascontiguousarray(phase.bin, dtype=np.uint8)
        # Component masses from the authoritative total (dense: W = M[..., None] * F).
        masses = Mc.reshape(-1)[_row_of_entry(row_ptr)] * phase.fraction
        positions = np.ascontiguousarray(phase.position_m, dtype=np.float64)
        dep_decay = np.zeros((n_active, nc), dtype=np.float64)
        settled = np.zeros((n_active, nc), dtype=np.float64)
        export_c = np.zeros((n_active, nc), dtype=np.float64)
        out_internal = np.zeros((n_active, nc), dtype=np.float64)
        in_internal = np.zeros((n_active, nc), dtype=np.float64)
        arrival = np.zeros((n_active, nc), dtype=np.float64)
        peak_nnz = 0
        # CW-B46 exact empty-transport path. Every check above has run (structure, values, inactive cells, Courant,
        # exponent, terminal settle mask, Numba compilation, row-sum self-check). With no stored bin (validated
        # `nnz == 0`) and no compact pickup packet (`Pc != 0.0` nowhere; the kernels skip `m == 0.0`, so -0.0
        # too), `_source_row` and `_row_candidates` skip every packet, so each substep's count pass returns
        # all-zero counts and +0.0 accounts, the write pass writes nothing, and `_row_sums` leaves every empty row
        # at +0.0. Those exact results are substituted (fresh buffers, as the last substep would have produced);
        # the accumulators stay the +0.0 arrays above. Everything below is unchanged.
        empty_transport = phase.nnz == 0 and not bool(np.any(Pc != 0.0))
        if empty_transport:
            packet = Pc
            row_ptr = np.zeros(n_rows + 1, dtype=np.int64)
            bins = np.empty(0, dtype=np.uint8)
            masses = np.empty(0, dtype=np.float64)
            positions = np.empty(0, dtype=np.float64)
        for k in range(0 if empty_transport else n_sub):
            packet = Pc if k == 0 else no_pickup  # pickup injected ONCE, at x = 0
            (counts, d_decay, d_settled, d_export, d_out, d_in, d_arr_decay, d_arr_settled) = count_pass(
                row_ptr, masses, positions, packet, Vc, Rc, Sc, maps.cell_of_slot, maps.slot_of_cell,
                maps.receiver_slot, maps.outlet, ny, nx, nb, dx, dt_sub)
            row_ptr_new = np.zeros(n_rows + 1, dtype=np.int64)
            np.cumsum(counts, out=row_ptr_new[1:])
            nnz_new = int(row_ptr_new[-1])
            peak_nnz = max(peak_nnz, nnz_new)
            if cap is not None and nnz_new > cap:
                raise SparseCapacityError(nnz_new, cap, k, n_sub)
            bins_new = np.empty(nnz_new, dtype=np.uint8)
            mass_new = np.empty(nnz_new, dtype=np.float64)
            pos_new = np.empty(nnz_new, dtype=np.float64)
            kernels.run_write_pass(row_ptr, masses, positions, packet, Vc, Rc, Sc, maps.cell_of_slot,
                                   maps.slot_of_cell, maps.receiver_slot, ny, nx, nb, dx, dt_sub, NARROW_SPREAD,
                                   row_ptr_new, bins_new, mass_new, pos_new)
            dep_decay += d_decay + d_arr_decay
            settled += d_settled + d_arr_settled
            export_c += d_export
            out_internal += d_out
            in_internal += d_in
            arrival += d_arr_decay + d_arr_settled
            row_ptr, bins, masses, positions = row_ptr_new, bins_new, mass_new, pos_new
            del d_decay, d_settled, d_export, d_out, d_in, d_arr_decay, d_arr_settled, counts
        del Mc, Pc, Vc, Rc, Sc, no_pickup, packet, a, exponent

        # Remaining pool per row (dense: W.sum(axis=-1)): the compiled emulation,
        # self-checked above against the running NumPy; `row_sums_numpy` is the reference.
        if empty_transport:
            remaining_c = np.zeros(n_rows, dtype=np.float64).reshape(n_active, nc)  # `_row_sums` of empty rows
        else:
            remaining_c = np.asarray(kernels.run_row_sums(row_ptr, bins, masses, nb)).reshape(n_active, nc)
        rows_after = _row_of_entry(row_ptr)
        # Fractions of the remaining pool (dense: W / remaining on occupied rows).
        fraction_new = masses / remaining_c.reshape(-1)[rows_after]

        def full(compact):
            out = np.zeros((n, nc), dtype=np.float64)
            out[idx] = compact
            return out

        remaining = full(remaining_c)
        dep_decay_f = full(dep_decay)
        del dep_decay
        settled_f = full(settled)
        del settled
        export = full(export_c)
        del export_c
        out_internal_f = full(out_internal)
        del out_internal
        in_internal_f = full(in_internal)
        del in_internal
        arrival_f = full(arrival)
        del arrival
        deposition = dep_decay_f + settled_f
        T = (remaining + deposition) + export
        before = M + P
        divergence = in_internal_f - out_internal_f
        cell_balance = (T - before) - divergence
        cell_scale = np.maximum(np.maximum(before, T), np.maximum(in_internal_f, out_internal_f))
        n_ops = operation_count(n_sub, nb)
        cell_tol = summation_error_bound_kg(n_ops, cell_scale)

        before_total = pairwise_sum_over_leading_axes(before)
        after_total = pairwise_sum_over_leading_axes(T)
        dep_total = pairwise_sum_over_leading_axes(deposition)
        exp_total = pairwise_sum_over_leading_axes(export)
        in_total = pairwise_sum_over_leading_axes(in_internal_f)
        out_total = pairwise_sum_over_leading_axes(out_internal_f)
        residual = after_total - before_total
        pairwise = pairwise_bound_scale_factor(n)
        tolerance = summation_error_bound_kg(
            n_ops, np.maximum(np.maximum(before_total, after_total), np.maximum(in_total, out_total))) \
            + 2.0 * pairwise * (before_total + after_total)

        crossing = out_internal_f + export
        x_gross = np.zeros((ny, nx + 1, nc), dtype=np.float64)
        y_gross = np.zeros((ny + 1, nx, nc), dtype=np.float64)
        x_net = np.zeros((ny, nx + 1, nc), dtype=np.float64)
        y_net = np.zeros((ny + 1, nx, nc), dtype=np.float64)
        if network.cells_x.size:
            vx = crossing[network.cells_x]
            scatter_add(x_gross, (network.rows_x, network.faces_x), vx)
            scatter_add(x_net, (network.rows_x, network.faces_x), vx * network.sign_x[:, None])
        if network.cells_y.size:
            vy = crossing[network.cells_y]
            scatter_add(y_gross, (network.faces_y, network.cols_y), vy)
            scatter_add(y_net, (network.faces_y, network.cols_y), vy * network.sign_y[:, None])

    # Output validation before anything is returned (same identities as the dense step).
    for name, array in (("mobile_after_transfer", T), ("deposition_request", deposition),
                        ("export_request", export), ("internal_transfer_in", in_internal_f),
                        ("internal_transfer_out", out_internal_f), ("x_face_gross", x_gross),
                        ("y_face_gross", y_gross), ("x_face_net", x_net), ("y_face_net", y_net),
                        ("decay_deposition", dep_decay_f), ("settled", settled_f), ("phase fraction", fraction_new),
                        ("phase position", positions), ("remaining mass", masses)):
        if not np.all(np.isfinite(array)):
            raise TransportError(f"sparse characteristic transport produced non-finite {name}")
    for name, array in (("mobile_after_transfer", T), ("deposition_request", deposition),
                        ("export_request", export), ("internal_transfer_in", in_internal_f),
                        ("internal_transfer_out", out_internal_f), ("x_face_gross", x_gross),
                        ("y_face_gross", y_gross), ("decay_deposition", dep_decay_f), ("settled", settled_f),
                        ("phase fraction", fraction_new), ("phase position", positions), ("remaining mass", masses)):
        if np.any(array < 0.0):
            raise TransportError(f"sparse characteristic transport produced negative {name}")
    if np.any(export > T):
        raise TransportError("export request exceeds the post-transfer pool")
    if np.any(deposition > T):
        raise TransportError("deposition request exceeds the post-transfer pool")
    if np.any((deposition + export) > T * (1.0 + 4.0 * _EPS)):
        raise TransportError("deposition plus export requests exceed the post-transfer pool")
    if np.any(np.abs(cell_balance) > cell_tol):
        raise TransportError("per-cell mobile balance T - (M + P) = in - out violated beyond FP64 tolerance")
    if not np.all(np.isfinite(residual)):
        raise TransportError("per-class budget residual is non-finite")
    if np.any(np.abs(residual) > tolerance):
        raise TransportError("per-class mobile mass not conserved by the transfer beyond the declared FP64 bound")
    if masses.size:
        if np.any(masses <= 0.0):
            raise TransportError("sparse characteristic transport stored a bin without mass")
        sums = np.bincount(rows_after, weights=fraction_new, minlength=n_rows)
        if np.any((np.diff(row_ptr) > 0) & (np.abs(sums - 1.0) > fraction_sum_tolerance(nb))):
            raise TransportError("phase fractions after the step do not sum to 1 within the declared bound")
        if np.any(positions >= dx):
            raise TransportError("phase position after the step reached dx")
        if np.any(bin_index(positions, dx, nb, np) != bins.astype(np.int64)):
            raise TransportError("phase position after the step left its bin")
        same_row = rows_after[1:] == rows_after[:-1]
        if np.any(same_row & ~(bins[1:] > bins[:-1])):
            raise TransportError("sparse characteristic transport wrote bins out of order")
    if np.any((np.diff(row_ptr) > 0) != (remaining_c.reshape(-1) > 0.0)):
        raise TransportError("sparse characteristic transport left a row inconsistent with its remaining mass")
    if np.any(export[~outlet] != 0.0):
        raise TransportError("export requested at a non-outlet cell")

    def grid(array):
        return array.reshape(shape3)

    transport = TransportStep(
        dt_s=dt, n_substeps=n_sub,
        mobile_after_transfer_kg=grid(T), deposition_request_kg=grid(deposition), export_request_kg=grid(export),
        decay_deposition_kg=grid(dep_decay_f), settled_kg=grid(settled_f),
        internal_transfer_in_kg=grid(in_internal_f), internal_transfer_out_kg=grid(out_internal_f),
        divergence_kg=grid(divergence),
        x_face_gross_kg=x_gross, y_face_gross_kg=y_gross, x_face_net_kg=x_net, y_face_net_kg=y_net,
        mobile_before_by_class_kg=before_total, mobile_after_by_class_kg=after_total,
        deposition_request_by_class_kg=dep_total, export_request_by_class_kg=exp_total,
        budget_residual_by_class_kg=residual, budget_tolerance_by_class_kg=tolerance,
        max_courant=max_courant, max_decay_exponent=max_exponent,
        max_cell_balance_residual_kg=np.max(np.abs(cell_balance)),
    )
    phase_after = SparsePhaseState(
        row_ptr=freeze(row_ptr), bin=freeze(bins), fraction=freeze(fraction_new), position_m=freeze(positions),
        dx_m=dx, n_bins=nb, n_classes=nc, shape=(int(ny), int(nx)),
        cell_of_slot=phase.cell_of_slot, slot_of_cell=phase.slot_of_cell,
        network_sha256=network.graph_input_sha256, xp=np,
    )
    return SparseCharacteristicStep(
        transport=transport, phase_after=phase_after, mobile_before_kg=grid(before),
        mobile_remaining_kg=grid(remaining), arrival_deposition_kg=grid(arrival_f), crossing_kg=grid(crossing),
        n_bins=nb, implementation=implementation, nnz_after=int(masses.size), peak_nnz=int(peak_nnz),
    )
