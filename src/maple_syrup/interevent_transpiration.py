"""Explicit REQUESTED-transpiration general interevent owner (Phase 1, task phase1_transpiration_owner_20261110).

SYRUP owns and debits all physical water. A caller (later: the LEAF bridge) supplies an aggregate transpiration REQUEST
`(L, N)` of water-depth AMOUNTS in metres (not rates: no dt/PET conversion, no plant model here). This module accepts it
at the actual interevent instant through the EXISTING zero-time native `subsurface_transpiration.withdraw_transpiration`
(available = max(w - residual, 0), actual = min(request, available), unmet and the common proportional scale returned;
no compensation, no privileged plants, no second soil authority) and then lets genuine hydrology/E continue:

    gop  = prepare_general_interevent(base_prep)                        # base: iw.InterEventWaterOperator (T0)
    s0   = enable_requested_transpiration(gop, t0_state, expected_t_s=..., expected_generation=...)
    req  = prepare_withdrawal_request(gop, s0, depth_m, identity="...")
    cand = request_transpiration(gop, s0, req, expected_revision=s0.revision)       # zero time
    s1   = cand.state                                                    # the CALLER adopts the whole candidate or drops it
    h    = general_interevent_water_step(gop, s1, pet, dt, phase="interevent")      # real E/hydrology, same shared body

Why a distinct typed owner (docs/subsurface_hydrology/interevent_transpiration.md): the T0 `InterEventState`/
`WaterWindowLedger` keep their exact rule `transpiration_m3 == 0`; soil that a T debit changed must never be wrapped in a
T0 record. The general ledger therefore COMPOSES the genuine unchanged hydrology ledger (T permanently 0) with an explicit
T account. Counters (all Python ints, normalised before arithmetic):

    hydro_generation == ledger.hydrology.n_steps
    revision         == (hydro_generation - entry_hydro_generation) + n_requests
    next_sequence    == initial_sequence + n_requests

A hydro step advances time/generation/n_steps/revision once; a T request is ZERO TIME (clock, hydro count/elapsed,
surface, faces, wetting history, bed untouched) and advances revision/request count/sequence once (an all-zero request
too: it consumes its transaction, accepts 0 water and invents no plant growth).

Purity and replay: every function is pure. The scalar `digest` is a canonical SCALAR identity (not a physical parent identity):
requests also bind the actual parent INSTANCE by a non-owning weakref (in-memory only; a future codec must rebind). Re-evaluating a request against the SAME old state may reproduce the same
proposal; this module has no global registry/compare-and-swap and does not claim crash-atomic publication. A request is
bound to (profile, parent token, revision, hydro generation, clock, sequence); against the NEW returned state it refuses as
stale/duplicate. The opaque caller identity is a bounded label, NOT proof of LEAF/weather/scenario provenance.

Scope: native Dense (exact `VoxelColumnState`) and `bed=None`, NumPy/Numba by default; Windowed/foreign storage and CUDA are fenced
early (CUDA-compatible guards authored, nothing was run on a GPU). One EXPLICIT opt-in extends this to the actual native39
`WindowedVoxelColumnState` bed (`prepare_general_interevent(..., bed_storage="windowed")`, CPU, fixed-material water only, published
full/v1 store, the pinned MAPLE39 package only; see `WINDOWED_PROFILE`/`WINDOWED_T_SOURCE_DIGESTS` and the doc). No codec, calendar/scheduler,
LEAF import, adoption or catchment claim.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import weakref
from dataclasses import asdict, dataclass, replace
from typing import Any, NamedTuple

import numpy as np
from maple.core.types.voxel import VoxelColumnState

from maple_syrup import interevent_water as iw
from maple_syrup import layered_event as ev
from maple_syrup import layered_storm as ls
from maple_syrup import subsurface as ss
from maple_syrup import provenance as prov
from maple_syrup import sediment_bed as sb
from maple_syrup import subsurface_transpiration as st
from maple_syrup.conservation import volume_roundoff_bound_m3
from maple_syrup.surface_routing import (
    ExperimentalHydrologyError,
    SurfaceStep,
    advance_time,
)

__all__ = [
    "BED_STORAGES",
    "CUDA9F_PROFILE",
    "CUDA_PROFILE",
    "CudaOwnerMemory",
    "estimate_cuda_owner_memory",
    "GENERAL_PROFILE",
    "WINDOWED_PROFILE",
    "WINDOWED_T_SOURCE_DIGESTS",
    "WindowedTProfile",
    "GeneralIntereventCandidate",
    "GeneralIntereventError",
    "GeneralIntereventMemory",
    "GeneralIntereventOperator",
    "GeneralIntereventState",
    "GeneralWaterLedger",
    "TranspirationCandidate",
    "TranspirationLedger",
    "TranspirationReceipt",
    "TranspirationRequestError",
    "WithdrawalRequest",
    "audit_general_interevent",
    "describe",
    "enable_requested_transpiration",
    "estimate_general_memory",
    "general_interevent_recharge_step",
    "general_interevent_water_step",
    "general_metadata",
    "prepare_general_interevent",
    "prepare_withdrawal_request",
    "request_transpiration",
]

GENERAL_PROFILE = "maple_syrup.interevent_transpiration/requested_amount/1"
BED_STORAGES = ("dense", "windowed")
#: The EXPLICIT, UNQUALIFIED CUDA capability (genuine native39 Dense or bedless fixed material, requested T). Never selected by default.
CUDA_PROFILE = GENERAL_PROFILE + "/native39_cuda_dense_or_bedless_fixed_material_UNQUALIFIED"
#: The separate EXPLICIT, UNQUALIFIED native9f counterpart (genuine native9f Dense or bedless fixed material). A distinct literal: it seeds its own operator
#: identities and is verified through the narrow dense-CUDA source helper, never through the Windowed pin set.
CUDA9F_PROFILE = GENERAL_PROFILE + "/native9f_cuda_dense_or_bedless_fixed_material_UNQUALIFIED"
#: Retained-record charge of one `dense_cuda_source.DenseCudaSource` (a literal so the pure plans import nothing; the cold 9f verification and a test tie it to
#: `dense_cuda_source.RECORD_RETAINED_BYTES`). Added beside every owner/capability that holds such a record; the native39 record is inside `OBJECT_BYTES`.
DENSE_SOURCE_RECORD_BYTES = 6144
_NATIVE9F_EXCLUDED = ("LEAF import/growth, calendar/scheduler, codec, Windowed, the native39 and any other MAPLE source, 9f GPU case setup/transfer factory, "
                      "moving material/remap, sediment transport, wind, speed/scale claims, adoption")
#: Declared host scratch of the hot scalar packets (primitive 8, owner 1, audit/hydro tens): 64 doubles.
CUDA_HOST_PACKET_BYTES = 64 * 8
#: The EXPLICIT Windowed capability: a distinct profile, never selected by default or by a digest alone.
WINDOWED_PROFILE = GENERAL_PROFILE + "/native39_windowed_cpu_full_v1_fixed_material"
#: The ONLY MAPLE package digest this capability is qualified for (the immutable merged MAPLE `39bcf79e` bundle). Deliberately a
#: literal local to this feature: it is not derived from, and never extends, any other source list (native9f is not in it).
WINDOWED_T_SOURCE_DIGESTS = frozenset({"af54626ab432670539ed71480fd884f13581dfdfcfec63d7c85053ca215adfec"})
#: Named, code-derived counts of the temporaries of the cold native NON-SCAN validators (see `_entry_terms`); D = 8 N K bytes, b = N K bytes.
#: availability validator: total, |active|, scale, two bound temporaries, gap (+ one bool mask)  -> 6 D + b
ENTRY_AVAILABILITY_DOUBLE_ARRAYS = 6
#: reconciliation: reference R, fold accumulator pair, physical change, gross activity, |R|, local scale, touches, fp64 tolerance,
#: tolerance, mismatch and one live temporary are at most 9 live (N, K) arrays; the (P, N, K) gross sum adds P+3 at its own peak.
ENTRY_RECONCILE_LIVE_ARRAYS = 9
ENTRY_RECONCILE_GROSS_EXTRA = 3
#: Fixed Python-object overhead of the validators: at most 200 transient ndarray/tuple/str/list headers of <= 128 B each (13 ledger field
#: records, check/violation lists, shape tuples, views, f-strings); independent of N and K.
ENTRY_OBJECT_BYTES = 200 * 128
#: Source-hash stage (see `_source_stage_bytes`): bounds are the provenance reader's own limits and the OS path limit.
SOURCE_PATH_MAX_BYTES = 4096
SOURCE_NAME_MAX_BYTES = 255
SOURCE_RECORD_FIXED_BYTES = 56 + 8 + 49 + 113 + 8 + 8 + 49 + 8  # entry tuple, list slot, two str headers, hex digest, sort/copy/walk slots
SOURCE_STAGE_FIXED_BYTES = 1 << 16  # Path/stat_result/hasher/os.walk iteration objects, declared
INTEREVENT = iw.INTEREVENT
_F8 = 8
_SEAL = object()
MAX_IDENTITY_CHARS = 128
MAX_COUNTER = 2 ** 53
#: Owner per-cell validation after a withdrawal: a handful of N-vectors (flags, masks, row differences), declared.
OWNER_CHECK_SCRATCH_DOUBLES_PER_CELL = 12
#: Plus the late audit's inventory temporaries: one masked soil copy (L) and the two surface temporaries per cell.
AUDIT_TEMPORARY_DOUBLES_PER_CELL_PER_LAYER = 1
AUDIT_TEMPORARY_DOUBLES_PER_CELL = 2
OBJECT_BYTES = iw.OBJECT_BYTES
_REQUEST_SCALARS = ("identity", "sequence", "parent_token", "expected_revision", "expected_t_s", "expected_generation",
                    "profile_sha256", "payload_sha256")

MEMORY_SCOPE = (
    "General requested-transpiration owner under the EXISTING column convention. Components (all ADMITTED against the "
    "column's available_bytes_fn on its side before any allocation/read): resident = the base owner's accepted aggregate + "
    "static arrays + the T operator's two persistent layer vectors; reserves = the storm's extra and native reserves, the "
    "base owner's evaporation/extra reserve and this wrapper's extra reserve; a request preparation adds ONE owned (L, N) "
    "copy (+ CUDA host staging chunk); a withdrawal admits the native `estimate_transpiration_memory` as the primitive "
    "itself does (accepted soil counted once there, owned request, four (L, N) outputs, row reductions, backend scratch) "
    "with (resident - that soil) + reserves + the owner per-cell check scratch passed as the primitive's extra reserve "
    "(no double count, no hidden copy); the OLD and NEW soil are both live while a candidate is evaluated, the unchanged surface/faces/"
    "history/bed leaves are shared. A hydro step admits the base owner's step requirement + this wrapper's extra + the "
    "caller-declared `retained_feedback_bytes` (actual/unmet/scale and the request that the caller still holds; 0 = "
    "released); a returned candidate may hold those arrays after its state is adopted. EXCLUDED: allocator pools, CUDA "
    "context/NVRTC/Numba JIT, native voxel/ledger material outside the storm operator's native reserve, the "
    "release lifetime of the base owner's native `bed.water` reference (inherited, separately audited), checkpoint "
    "staging, LEAF and the process. A reading is not a reservation; no whole-process fit is claimed."
)


# --- errors / scalars --------------------------------------------------------------------------------------------------
class GeneralIntereventError(iw.InterEventWaterError):
    """Invalid binding, state, ledger, request, phase or option; raised before any result exists (inputs untouched)."""


class TranspirationRequestError(GeneralIntereventError):
    """A request was refused (stale/duplicate/unbound payload, native validation or late audit failure). No partial result."""


def _real(value: Any, name: str) -> float:
    try:
        return iw._real(value, name)
    except iw.InterEventWaterError as exc:
        raise GeneralIntereventError(str(exc)) from None


def _nonneg(value: Any, name: str) -> float:
    out = _real(value, name)
    if out < 0.0:
        raise GeneralIntereventError(f"{name} must be >= 0, got {out!r}")
    return out


def _count(value: Any, name: str) -> int:
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or int(value) < 0
            or int(value) > MAX_COUNTER):
        raise GeneralIntereventError(f"{name} must be a non-Boolean integer in [0, 2**53], got {type(value).__name__}")
    return int(value)


def _ident(value: Any, name: str) -> str:
    if (type(value) is not str or not 1 <= len(value) <= MAX_IDENTITY_CHARS or not value.isascii()
            or not value.isprintable()):
        raise GeneralIntereventError(f"{name} must be a printable ASCII string of 1..{MAX_IDENTITY_CHARS} characters "
                                     "(a bounded opaque caller label, not verified LEAF provenance)")
    return value


def _sha(value: Any, name: str) -> str:
    try:
        return iw._sha(value, name)
    except iw.InterEventWaterError as exc:
        raise GeneralIntereventError(str(exc)) from None


def _digest(tag: str, body: dict) -> str:
    return hashlib.sha256(tag.encode() + b"\0" + json.dumps(body, sort_keys=True, allow_nan=False).encode()).hexdigest()


# --- memory ------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class GeneralIntereventMemory:
    """Composed requirements (bytes on the admitted side, see MEMORY_SCOPE). `*_required_bytes` are admitted per call;
    `step_required_bytes` excludes the caller's `retained_feedback_bytes`; `feedback_bytes` is the size of the four (L, N)
    arrays (request, actual, unmet, scale) a caller may still hold."""

    backend: str
    n_layers: int
    n_cells: int
    resident_bytes: int
    reserves_bytes: int
    column_reserve_bytes: int
    request_bytes: int
    feedback_bytes: int
    owner_check_scratch_bytes: int
    transition_required_bytes: int
    request_prepare_required_bytes: int
    request_prepare_host_bytes: int
    withdraw_required_bytes: int
    withdraw_extra_reserve_bytes: int
    step_required_bytes: int
    audit_required_bytes: int
    owned_parameter_bytes: int
    cold_prepare_required_bytes: int
    cold_prepare_extra_reserve_bytes: int
    supplied_prepare_required_bytes: int
    transpiration: dict
    scope: str = MEMORY_SCOPE

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def estimate_general_memory(base: iw.InterEventWaterOperator, t_op: st.TranspirationOperator | None = None, *,
                            extra_reserve_bytes: int = 0) -> GeneralIntereventMemory:
    """Pure arithmetic from the base owner's and the native transpiration estimates; no allocation, no device query.
    `t_op` is accepted for call compatibility only (the figures depend on L, N, the base owner and the reserves)."""
    del t_op
    extra = ss._checked_int(extra_reserve_bytes, "extra_reserve_bytes")
    op = base.op
    L, N, backend = base.n_layers, base.n_cells, base.backend
    column_reserve = int(op.column.reserve_bytes)
    t_params = 2 * _F8 * L
    base_resident = int(base.memory.resident_bytes)  # includes the inherited native reference depth H0 when a bed is bound
    resident = base_resident + t_params
    reserves = (int(op.extra_reserve_bytes) + int(op.native_reserve_bytes) + int(base.memory.extra_reserve_bytes) + extra)
    ln = _F8 * L * N
    scratch = (OWNER_CHECK_SCRATCH_DOUBLES_PER_CELL + AUDIT_TEMPORARY_DOUBLES_PER_CELL_PER_LAYER * L
               + AUDIT_TEMPORARY_DOUBLES_PER_CELL) * N * _F8
    host_chunk = min(ls.HASH_CHUNK_BYTES, ln) if backend == "cuda" else 0
    # the primitive ALWAYS counts the accepted soil (L, N) itself, so that one array is removed from the resident extra
    withdraw_extra = resident - t_params - ln + reserves + scratch
    tm = st.estimate_transpiration_memory(L, N, backend=backend, reserve_bytes=column_reserve,
                                          extra_reserve_bytes=withdraw_extra, include_state=True)
    # COLD preparation: the owned immutable (L,) residual + saturation copies (2 L doubles) and their detached temporaries
    # (2 L doubles); the native prepare then adds its own host vectors, parameter vectors, warm-up arrays and one byte.
    owned = 4 * _F8 * L
    warm = st.estimate_transpiration_memory(1, 1, backend=backend)
    native_prepare = 6 * _F8 * L + 2 * _F8 * L + int(warm.total_bytes) + 1  # == TranspirationOperator.prepare_bytes
    cold_extra = base_resident + reserves + owned + OBJECT_BYTES  # live beside the native cold work (no accepted soil copy)
    supplied = cold_extra + column_reserve
    audit = int(base.memory.required_bytes) + t_params + extra  # the whole owner + declared extras: never less than a step
    return GeneralIntereventMemory(
        backend=backend, n_layers=L, n_cells=N, resident_bytes=resident, reserves_bytes=reserves,
        column_reserve_bytes=column_reserve, request_bytes=ln, feedback_bytes=4 * ln, owner_check_scratch_bytes=scratch,
        transition_required_bytes=max(resident + reserves + column_reserve + OBJECT_BYTES, audit),
        request_prepare_required_bytes=resident + ln + reserves + column_reserve + OBJECT_BYTES,
        request_prepare_host_bytes=host_chunk,
        withdraw_required_bytes=int(tm.required_bytes), withdraw_extra_reserve_bytes=withdraw_extra,
        step_required_bytes=int(base.memory.required_bytes) + t_params + extra, audit_required_bytes=audit,
        owned_parameter_bytes=owned, cold_prepare_required_bytes=native_prepare + column_reserve + cold_extra,
        cold_prepare_extra_reserve_bytes=cold_extra, supplied_prepare_required_bytes=supplied, transpiration=tm.as_dict())


def _admit(gop: Any, need: int, what: str) -> None:
    if isinstance(gop, GeneralIntereventOperator) and gop.cuda is not None:
        need = int(need) + int(gop.cuda.memory.device_param_packet_bytes)  # the (2L) device check packet (CUDA owner only)
    try:
        iw._admit(gop.base.op if isinstance(gop, GeneralIntereventOperator) else gop.op, int(need), what)
    except iw.InterEventMemoryError:
        raise
    except iw.InterEventWaterError as exc:
        raise GeneralIntereventError(str(exc)) from None


# --- CUDA owner (explicit, UNQUALIFIED opt-in; see docs) ----------------------------------------------------------------
@dataclass(frozen=True)
class CudaOwnerMemory:
    """HOST-side requirements (bytes) of the CUDA owner, separate from the device-side `GeneralIntereventMemory` (which is admitted
    against the column's `available_bytes_fn`). Pure arithmetic of (L, N); `device_param_packet_bytes` is the one device-side
    addition (the stacked (2L) parameter check packet) that `_admit` adds for a CUDA owner. A reading is not a reservation."""

    n_layers: int
    n_cells: int
    host_reserve_bytes: int
    persistent_host_bytes: int
    cold_vectors_bytes: int
    source_stage_bytes: int
    cold_required_bytes: int
    hot_required_bytes: int
    request_capture_host_bytes: int
    request_prepare_required_bytes: int
    device_param_packet_bytes: int

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def _cuda_source_record_extra_bytes(profile: Any) -> int:
    """Additional retained host bytes of the profile's source-association record (0 for the native39 profile: its record is within `OBJECT_BYTES`)."""
    if profile == CUDA_PROFILE:
        return 0
    if profile == CUDA9F_PROFILE:
        return DENSE_SOURCE_RECORD_BYTES
    raise GeneralIntereventError(f"cuda_profile must be exactly CUDA_PROFILE or CUDA9F_PROFILE, got {profile!r}")


def estimate_cuda_owner_memory(n_layers: int, n_cells: int, *, host_reserve_bytes: int = 0,
                               source_cold: bool = True, source_record_bytes: int = 0) -> CudaOwnerMemory:
    """Metadata-only host figures (no allocation, no device or host query, no source read). `source_cold=True` includes the
    streamed source-tree stage (`_source_stage_bytes`); the cold figure is the MAXIMUM of the sequential stages (source stage;
    native host vectors 6*8L + canonical copies/detached temporaries 4*8L), not their sum. `source_record_bytes` (default 0: the unchanged native39 figures)
    is the ADDITIONAL retained source-association record of a native9f owner (`DENSE_SOURCE_RECORD_BYTES`); it is alive beside the vectors, the hot
    packets and every request preparation, so it is added to those figures and to the persistent host bytes (never to the source stage, which precedes it)."""
    L, N = ss._checked_int(n_layers, "n_layers"), ss._checked_int(n_cells, "n_cells")
    reserve = ss._checked_int(host_reserve_bytes, "host_reserve_bytes")
    rec = ss._checked_int(source_record_bytes, "source_record_bytes")
    if L < 1 or N < 1:
        raise GeneralIntereventError("n_layers and n_cells must be >= 1")
    stage = _source_stage_bytes() if source_cold else 0
    vectors = (6 + 4) * _F8 * L
    hot = 4 * 2 * _F8 * L + CUDA_HOST_PACKET_BYTES + OBJECT_BYTES
    chunk = min(ls.HASH_CHUNK_BYTES, _F8 * L * N)
    return CudaOwnerMemory(
        n_layers=L, n_cells=N, host_reserve_bytes=reserve, persistent_host_bytes=2 * _F8 * L + rec, cold_vectors_bytes=vectors,
        source_stage_bytes=stage, cold_required_bytes=reserve + OBJECT_BYTES + rec + max(stage, vectors),
        hot_required_bytes=reserve + hot + rec, request_capture_host_bytes=chunk,
        request_prepare_required_bytes=reserve + hot + rec + chunk, device_param_packet_bytes=2 * _F8 * L)


@dataclass(frozen=True, eq=False)
class _CudaOwner:
    """Private evidence of the CUDA capability: immutable HOST canonical bounds, the factory-produced kernel object and its source
    text, the verified native39 source association, the host quota controls and the host memory record."""

    host_residual: np.ndarray
    host_saturation: np.ndarray
    kernel: Any
    kernel_source: str
    source: Any  # `_WindowedSource` (native39 profile) or `dense_cuda_source.DenseCudaSource` (native9f profile); dispatched by `profile`, never by type alone
    host_available_bytes_fn: Any
    host_reserve_bytes: int
    memory: CudaOwnerMemory
    profile: str = CUDA_PROFILE


def _device_gate(column: Any) -> None:
    """Current-device / preflight gate BEFORE any allocation or quota query (same rule as the native `check_device`)."""
    try:
        if callable(getattr(column, "preflight", None)):
            column.preflight()
    except ss.SubsurfaceError as exc:
        raise GeneralIntereventError(str(exc)) from None


def _admit_host(gop: GeneralIntereventOperator, need: int, what: str) -> None:
    """Separate HOST quota admission for the CUDA owner (RAM, never device free memory); a CPU owner has none."""
    cuda = gop.cuda
    if cuda is None:
        return
    try:
        ls._admit_host_build(int(need), cuda.host_available_bytes_fn)
    except ls.LayeredMemoryError as exc:
        raise iw.InterEventMemoryError(f"{what}: {exc}") from None
    except ls.LayeredStormError as exc:
        raise GeneralIntereventError(str(exc)) from None


def _check_cuda_meta(gop: GeneralIntereventOperator) -> None:
    """Metadata-only CUDA guards (no value read): record type, genuine factory kernel binding, device-resident bound vectors, hot
    source association. Nothing is hashed or compiled."""
    cuda, t, column = gop.cuda, gop.t_op, gop.base.op.column
    if type(cuda) is not _CudaOwner or gop.windowed is not None or gop.windowed_source is not None:
        raise GeneralIntereventError("the CUDA capability record is invalid or combined with the Windowed profile; refused")
    xp = gop.xp
    if gop.base.backend != "cuda" or xp is np or column.backend != "cuda":
        raise GeneralIntereventError("the CUDA capability needs a CUDA column/owner")
    if (t.kernel is not cuda.kernel or type(cuda.kernel) is not xp.RawKernel
            or getattr(cuda.kernel, "name", None) != "withdraw_fused" or st._CUDA_SRC != cuda.kernel_source):
        raise GeneralIntereventError("the transpiration kernel is not the genuine factory-prepared withdrawal kernel; refused")
    for name in ("residual_xp", "saturation_xp"):
        arr = getattr(t, name)
        if (type(arr) is not xp.ndarray or arr.dtype != np.float64 or tuple(arr.shape) != (gop.n_layers,)
                or not arr.flags.c_contiguous or int(arr.device.id) != int(column.device_id)):
            raise GeneralIntereventError(f"the owned transpiration {name} bound is not the exact device (L,) float64 vector "
                                         "of the column's device; refused")
    for name in ("host_residual", "host_saturation"):
        arr = getattr(cuda, name)
        if type(arr) is not np.ndarray or arr.dtype != np.float64 or arr.shape != (gop.n_layers,) or arr.flags.writeable:
            raise GeneralIntereventError(f"the CUDA owner's host canonical {name} lost its exact immutable (L,) form; refused")
    if cuda.profile not in (CUDA_PROFILE, CUDA9F_PROFILE):
        raise GeneralIntereventError("the CUDA capability record names an unknown profile; refused")
    _check_cuda_source_hot(cuda.source, cuda.profile)


def _check_cuda_values(gop: GeneralIntereventOperator) -> None:
    """Canonical VALUE evidence for the CUDA owner, run at every physical use (zero demand included): the column's own canonical
    vectors must equal the owner's immutable host canonical vectors, and the device bounds must equal them bitwise. The device
    side is ONE stacked (2L) D2H packet (no per-scalar/layer fetch); the primitive remains the only physical computation."""
    cuda, t, xp = gop.cuda, gop.t_op, gop.xp
    residual, satb = _canonical_bounds(gop.base.op.column)
    if not (np.array_equal(residual, cuda.host_residual) and np.array_equal(satb, cuda.host_saturation)):
        raise GeneralIntereventError("the column's canonical layer properties changed after the CUDA owner was prepared; refused")
    from maple.core.backend import to_host

    packet = np.asarray(to_host(xp.concatenate([t.residual_xp, t.saturation_xp])), dtype=np.float64)
    L = gop.n_layers
    if not (np.array_equal(packet[:L], cuda.host_residual) and np.array_equal(packet[L:], cuda.host_saturation)):
        raise GeneralIntereventError("the device transpiration bounds differ from the canonical host evidence; refused "
                                     "(they would change the protected-residual physics)")


# --- ledgers -----------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class TranspirationLedger:
    """Cumulative host scalars of accepted T since the explicit T anchor (m3, active cells). `actual_m3` is the ONLY physical
    loss; requested/unmet are diagnostics. `soil_bound_m3` accumulates the native per-transaction FP64 bounds of the soil
    change `after - before + actual`. Every addition is normalised to Python float/int BEFORE arithmetic."""

    n_requests: int = 0
    requested_m3: float = 0.0
    actual_m3: float = 0.0
    unmet_m3: float = 0.0
    soil_bound_m3: float = 0.0
    max_abs_soil_residual_m3: float = 0.0
    max_abs_request_residual_m3: float = 0.0

    def accepted(self, budget: st.TranspirationBudget) -> TranspirationLedger:
        _check_tledger(self)
        res = TranspirationLedger(
            int(self.n_requests) + 1, float(self.requested_m3) + float(budget.requested_m3),
            float(self.actual_m3) + float(budget.actual_m3), float(self.unmet_m3) + float(budget.unmet_m3),
            float(self.soil_bound_m3) + float(budget.bound_m3),
            max(float(self.max_abs_soil_residual_m3), abs(float(budget.residual_m3))),
            max(float(self.max_abs_request_residual_m3), abs(float(budget.request_residual_m3))))
        _check_tledger(res)
        return res

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def _check_tledger(led: Any) -> None:
    if type(led) is not TranspirationLedger:
        raise GeneralIntereventError("transpiration ledger must be a genuine TranspirationLedger")
    n = _count(led.n_requests, "tledger.n_requests")
    vals = {k: _nonneg(getattr(led, k), "tledger." + k) for k in
            ("requested_m3", "actual_m3", "unmet_m3", "soil_bound_m3", "max_abs_soil_residual_m3",
             "max_abs_request_residual_m3")}
    if vals["max_abs_soil_residual_m3"] > vals["soil_bound_m3"] or vals["max_abs_request_residual_m3"] > vals["soil_bound_m3"]:
        raise GeneralIntereventError("a T residual exceeds the accumulated native bound")
    if n == 0 and any(vals[k] != 0.0 for k in vals):
        raise GeneralIntereventError("a T ledger with no accepted request must be all zero")
    cum = volume_roundoff_bound_m3(n + 1, max(vals["requested_m3"], vals["actual_m3"], vals["unmet_m3"]))
    if abs(vals["requested_m3"] - vals["actual_m3"] - vals["unmet_m3"]) > vals["soil_bound_m3"] + cum:
        raise GeneralIntereventError("cumulative request != actual + unmet beyond the derived bound")


@dataclass(frozen=True)
class GeneralWaterLedger:
    """Whole-water window ledger: the GENUINE unchanged hydrology-only `WaterWindowLedger` (its T stays exactly 0) plus the
    explicit `TranspirationLedger`. The actual T is counted ONCE, here, in `actual_transpiration_m3`."""

    hydrology: iw.WaterWindowLedger
    transpiration: TranspirationLedger

    @property
    def actual_transpiration_m3(self) -> float:
        return float(self.transpiration.actual_m3)

    def as_dict(self) -> dict[str, Any]:
        return {"hydrology": self.hydrology.as_dict(), "transpiration": self.transpiration.as_dict()}


# --- the operator ------------------------------------------------------------------------------------------------------
class GeneralIntereventOperator:
    """Immutable factory-only binding of the base `InterEventWaterOperator` and the genuine `TranspirationOperator`
    prepared on THE SAME column, with the composed memory estimate and a cold profile SHA (distinct from the T0 identity;
    the base SHA is retained separately because the genuine storm receipt is bound to it)."""

    __slots__ = ("_seal", "backend", "base", "cuda", "extra_reserve_bytes", "memory", "n_cells", "n_layers", "sha256", "t_op", "windowed", "windowed_source", "xp")

    def __init__(self, seal: object, **fields: Any) -> None:
        if seal is not _SEAL:
            raise GeneralIntereventError("GeneralIntereventOperator is constructed only by prepare_general_interevent()")
        for name in self.__slots__:
            if name != "_seal":
                object.__setattr__(self, name, fields.pop(name))
        if fields:
            raise GeneralIntereventError(f"internal: unexpected operator fields {sorted(fields)}")
        object.__setattr__(self, "_seal", seal)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("GeneralIntereventOperator is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("GeneralIntereventOperator is immutable")

    def __repr__(self) -> str:
        return f"GeneralIntereventOperator(backend={self.backend!r}, sha256={self.sha256[:12]}...)"


def _check_gop_meta(gop: Any) -> None:
    """TRUE metadata-only revalidation (callable BEFORE admission): types, seals, identities, backend/extents, kernel and the
    owned bounds' type/dtype/shape/immutability flag. It forms and reads no canonical vector and allocates nothing."""
    if type(gop) is not GeneralIntereventOperator or getattr(gop, "_seal", None) is not _SEAL:
        raise GeneralIntereventError(f"operator must be a genuine GeneralIntereventOperator, got {type(gop).__name__}")
    try:
        iw._check_prepared(gop.base)
    except iw.InterEventWaterError as exc:
        raise GeneralIntereventError(f"the bound base owner is no longer valid: {exc}") from None
    column, t = gop.base.op.column, gop.t_op
    if (type(t) is not st.TranspirationOperator or t.column is not column or t.xp is not column.xp
            or t.backend != column.backend or t.n_layers != gop.base.n_layers or t.n_cells != gop.base.n_cells):
        raise GeneralIntereventError("the transpiration operator is no longer the one prepared on this column")
    try:
        t.check_device()
    except ss.SubsurfaceError as exc:
        raise GeneralIntereventError(str(exc)) from None
    _check_t_metadata(column, t, gop.cuda)
    if gop.xp is not gop.base.xp or gop.backend != gop.base.backend:
        raise GeneralIntereventError("operator record disagrees with its base owner; refused")
    if gop.backend == "cuda":
        _check_cuda_meta(gop)  # device-resident bounds + genuine kernel + hot source association (metadata only)
        return
    if gop.cuda is not None:
        raise GeneralIntereventError("a CUDA capability record is bound to a CPU owner; refused")
    for name in ("residual_xp", "saturation_xp"):
        arr = getattr(t, name)
        if type(arr) is not np.ndarray or arr.dtype != np.float64 or arr.shape != (gop.n_layers,) or arr.flags.writeable:
            raise GeneralIntereventError(f"the owned transpiration {name} bound lost its exact immutable (L,) host form; refused")
    if gop.windowed is None:
        return
    # explicit Windowed capability: the bound deep identity/source/publication must still be exactly the prepared one (metadata only;
    # no voxel read, deep read, re-hash, flush or publication)
    if type(gop.windowed) is not WindowedTProfile:
        raise GeneralIntereventError("the Windowed profile record is invalid; refused")
    _check_windowed_source_hot(gop.windowed_source)
    fields = _windowed_metadata(gop.base, gop.windowed.header_digest)
    if (gop.windowed.maple_package_sha256 != gop.windowed_source.package_sha256
            or any(getattr(gop.windowed, k) != v for k, v in fields.items())):
        raise GeneralIntereventError("the Windowed bed/store no longer matches the prepared Windowed profile; refused")


def _check_gop_values(gop: Any) -> None:
    """Canonical-bounds VALUE verification (reads the column's (L,) property vectors and forms comparison temporaries): call
    only AFTER the entry's composed admission."""
    if gop.cuda is not None:
        _check_cuda_values(gop)
        return
    _check_t_parameters(gop.base.op.column, gop.t_op, owned=True)


def _check_gop(gop: Any) -> None:
    """Full revalidation for internal use after admission: metadata plus canonical values."""
    _check_gop_meta(gop)
    _check_gop_values(gop)


def _canonical_bounds(column: Any) -> tuple[np.ndarray, np.ndarray]:
    """The column's own canonical residual and saturation(+tolerance) vectors, exactly as the native preparation derives them."""
    lay = column.layers
    residual = np.ascontiguousarray(np.asarray(lay.residual_m, dtype=np.float64))
    satb = np.ascontiguousarray(np.asarray(lay.saturation_m, dtype=np.float64)
                                + np.asarray(lay.capacity_tolerance_m, dtype=np.float64))
    return residual, satb


def _check_t_metadata(column: Any, t: Any, cuda: Any = None) -> None:
    """Metadata-only identity of a native operator (no value read): exact class, same column/namespace/backend, kernel. A CUDA
    kernel is accepted ONLY when it is the factory-produced one recorded in the owner's `_CudaOwner` (never by type/name alone)."""
    if type(t) is not st.TranspirationOperator or t.column is not column or t.xp is not column.xp:
        raise GeneralIntereventError("t_op must be an exact TranspirationOperator prepared on the base owner's column")
    if t.backend != column.backend or t.n_layers != column.n_layers or t.n_cells != column.n_cells:
        raise GeneralIntereventError("the transpiration operator's backend/extents differ from its column; refused")
    if t.backend == "cuda" and cuda is not None and type(cuda) is _CudaOwner:
        if t.kernel is not cuda.kernel:
            raise GeneralIntereventError("the transpiration operator carries a foreign kernel; refused")
        return
    expected_kernel = st._fused_numba if t.backend == "numba" else None
    if t.kernel is not expected_kernel:
        raise GeneralIntereventError("the transpiration operator carries a foreign kernel; refused")


def _check_t_parameters(column: Any, t: Any, *, owned: bool) -> None:
    """Compare the (L,) bound vectors with the SAME column's canonical properties (L is tiny: not a hot grid read). An owned
    operator must additionally hold immutable host arrays (a caller cannot make them writable)."""
    _check_t_metadata(column, t)
    residual, satb = _canonical_bounds(column)
    for name, got, want in (("residual", t.residual_xp, residual), ("saturation", t.saturation_xp, satb)):
        if type(got) is not np.ndarray or got.dtype != np.float64 or got.shape != want.shape:
            raise GeneralIntereventError(f"the transpiration {name} bound must be an exact float64 ({want.shape[0]},) host array")
        if owned and got.flags.writeable:
            raise GeneralIntereventError(f"the owned transpiration {name} bound lost its immutability; refused")
        if not np.array_equal(got, want):
            raise GeneralIntereventError(f"the transpiration {name} bound differs from the column's canonical layer "
                                         "property; refused (it would change the protected-residual physics)")


def _own_t_operator(column: Any, t: st.TranspirationOperator) -> st.TranspirationOperator:
    """Bind a DETACHED copy: the actual native dataclass (same column/kernel/memory) holding immutable owned canonical bounds.
    The caller's operator is never changed and a later caller mutation cannot reach the bound physics."""
    _check_t_parameters(column, t, owned=False)
    residual, satb = _canonical_bounds(column)
    return replace(t, residual_xp=ev._immutable_host(residual), saturation_xp=ev._immutable_host(satb))


@dataclass(frozen=True)
class WindowedTProfile:
    """Scalar identity of the EXPLICIT Windowed capability bound at preparation (metadata only; no deep data). `entry_scratch_bytes`
    is the declared extra the one-time cold native validation at `enable_requested_transpiration` is admitted with."""

    profile: str
    storage: str
    maple_package_sha256: str
    store_format_version: int
    physical_manifest_digest: str
    logical_manifest_digest: str
    header_digest: str
    store_generation: int
    n_voxels: int
    window_voxels: int
    n_classes: int
    packet_terms: int
    entry_scratch_bytes: int
    entry_terms: tuple

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True, eq=False)
class _WindowedSource:
    """The GENUINE source association recorded at preparation (not an identity claim by itself): the imported `maple` package module
    object and file, the Windowed class object and the verified package digest. Hot calls compare these cheaply (and the process cache
    when it happens to be warm); they NEVER hash the source tree. A changed association is a typed refusal: re-prepare."""

    module: Any
    module_file: str
    windowed_class: type
    package_sha256: str


def _fence_storage(base: iw.InterEventWaterOperator, bed_storage: str = "dense") -> dict | None:
    """Default (`"dense"`): Dense native bed or `bed=None`; Windowed/subclass/foreign storage is refused here, before any work.
    Explicit `"windowed"`: ONLY the actual native39 Windowed bed (see `_windowed_metadata`); returns its metadata fields. Neither branch
    reads a voxel, deep byte or array value, and NEITHER hashes any source tree (that cold stage is admitted separately)."""
    if type(bed_storage) is not str or bed_storage not in BED_STORAGES:
        raise GeneralIntereventError(f"bed_storage must be exactly one of {BED_STORAGES}, got {bed_storage!r}")
    if bed_storage == "windowed":
        return _windowed_metadata(base)
    bed = base.op.bed
    if bed is not None and type(bed.voxel_column) is not VoxelColumnState:
        raise GeneralIntereventError("only a native DENSE voxel column (exact VoxelColumnState) is qualified by default; Windowed or "
                                     f"foreign storage ({type(bed.voxel_column).__name__}) is refused (the Windowed bed has the "
                                     "separate explicit opt-in bed_storage='windowed')")
    return None


def _entry_terms(n_cells: int, k: int, ledger_axis: int) -> tuple[tuple[str, int], ...]:
    """Named sequential stages of the cold native non-scan validators (each frees its temporaries before the next; the entry scratch
    is their MAXIMUM). D = 8 N K, b = N K bytes; P = the ledger's process axis (7 Dense, fewer Compact), read from metadata.
      availability  : `check_sediment_availability_state`: total, |active|, scale, bound temporaries, gap + bool mask
      ledger        : `validate_ledger_state`: one finite/negative bool mask of the largest (P, N, K) field at a time
      committed     : `validate_committed_topography_state`: finite/negative bool masks, twice (also inside the reference resolver)
      water         : `validate_water_state`: finite/negative masks of the (N, K) mobile array
      reconciliation: reference R (retained) + window fold + physical change + (P, N, K) gross sum + local scale/touches/tolerance/
                      mismatch (see ENTRY_* constants); the resident H0 is the base owner's, counted once there.
    Each stage also carries the fixed `ENTRY_OBJECT_BYTES`. No deep chunk table, manifest or voxel is read by any of them."""
    d, b, p = _F8 * n_cells * k, n_cells * k, int(ledger_axis)
    o = ENTRY_OBJECT_BYTES
    return (("availability", ENTRY_AVAILABILITY_DOUBLE_ARRAYS * d + b + o), ("ledger", 2 * p * b + b + o),
            ("committed", 2 * 2 * b + 2 * _F8 * n_cells + o), ("water", 2 * b + 2 * n_cells + o),
            ("reconciliation", max(p + ENTRY_RECONCILE_GROSS_EXTRA, ENTRY_RECONCILE_LIVE_ARRAYS) * d + b + o))


def _source_stage_bytes() -> int:
    """Declared peak of the cold source-tree verification (`provenance.source_tree_digest`, streamed): one 1 MiB read chunk, at most
    `DEFAULT_MAX_FILES` bounded (path, hash) records with the sort/final-tuple copies and walk name lists, and fixed iteration objects.
    The whole tree is never resident. Excludes the interpreter, allocator pools and the file system."""
    record = SOURCE_RECORD_FIXED_BYTES + SOURCE_PATH_MAX_BYTES + SOURCE_NAME_MAX_BYTES
    return int(prov._READ_CHUNK_BYTES + prov.DEFAULT_MAX_FILES * record + SOURCE_STAGE_FIXED_BYTES)


def _windowed_source_cold() -> bool:
    return "digest" not in sb._MAPLE_DIGEST_CACHE


def _windowed_metadata(base: iw.InterEventWaterOperator, expected_digest: str | None = None) -> dict:
    """Every refusal that must precede ANY native/deep operation, from TYPES and METADATA only, and WITHOUT reading/hashing the source
    tree: genuine bound `BedState`/`BedContext`, the strict class identity of `is_windowed_bed` (subclass/unknown/duck refused; an
    import failure propagates), CPU namespace, fixed capacity, NO unpublished deep journal, the published full/v1 store, genuine prefix
    packets. No facade: a Dense or bedless owner is refused. The MAPLE source pin is verified separately (`_verify_windowed_source`)."""
    op = base.op
    bed, ctx = op.bed, op.context
    if bed is None or ctx is None:
        raise GeneralIntereventError("bed_storage='windowed' needs an actual native Windowed bed; a bedless owner is refused")
    if type(bed) is not sb.BedState or type(ctx) is not sb.BedContext:
        raise GeneralIntereventError("bed_storage='windowed' needs the genuine BedState/BedContext (no subclass or stand-in)")
    try:
        windowed = sb.is_windowed_bed(bed)
    except sb.BedIntegrationError as exc:
        raise GeneralIntereventError(f"unsupported bed representation for the Windowed profile: {exc}") from None
    if not windowed:
        raise GeneralIntereventError("bed_storage='windowed' needs an actual native Windowed bed (no Dense facade or conversion)")
    if base.backend not in ("numpy", "numba") or base.xp is not np:
        raise GeneralIntereventError("the Windowed profile is CPU-only (NumPy/Numba); CUDA is refused")
    from maple.surface.voxels import windowed_pending, windowed_prefix
    from maple.surface.voxels.windowed_inventory import require_fixed_windowed_capacity

    column = bed.voxel_column
    try:
        require_fixed_windowed_capacity(ctx.geometry, operation="SYRUP Windowed requested transpiration")
        windowed_pending.require_published_column(column, operation="SYRUP Windowed requested transpiration")
        packets = windowed_prefix.require_prefix_packets(column, operation="SYRUP Windowed requested transpiration")
    except (ValueError, TypeError) as exc:
        raise GeneralIntereventError(f"{exc}; this profile never publishes, canonicalises or discards a pending deep journal "
                                     "(publish the accepted storage state explicitly first)") from None
    store = column.deep_store
    if int(getattr(store, "format_version", 0)) != 1 or (ctx.deep_chunk_encoding, ctx.deep_overlay_max_bytes) != ("full", None):
        raise GeneralIntereventError("only the published full/v1 deep store is supported (overlay v2 refused before any deep work)")
    n_cells, k = int(base.n_cells), int(op.n_classes)
    try:
        axis = int(bed.ledger.actual_mass_positive_kg.shape[0])  # metadata only: Dense 7 / Compact present-process axis
    except (AttributeError, IndexError, TypeError):
        raise GeneralIntereventError("the native ledger exposes no process axis; refused") from None
    terms = _entry_terms(n_cells, k, axis)
    header = store.header
    return dict(profile=WINDOWED_PROFILE, storage="windowed", store_format_version=int(store.format_version),
                physical_manifest_digest=str(store.manifest_digest), logical_manifest_digest=str(store.logical_manifest_digest),
                header_digest=str(header.digest()) if expected_digest is None else expected_digest,
                store_generation=int(store.version_id), n_voxels=int(column.n_voxels), window_voxels=int(header.window_voxels),
                n_classes=k, packet_terms=int(packets.packet_terms), entry_scratch_bytes=int(max(v for _, v in terms)),
                entry_terms=terms)


def _verify_windowed_source() -> _WindowedSource:
    """COLD source qualification, only after its stage was admitted: the real (cached once per process) streamed package digest must be
    in the local literal pin set; records the genuine association for hot guards. A foreign digest is a typed refusal."""
    digest = sb.bound_maple_package_digest()
    if digest not in WINDOWED_T_SOURCE_DIGESTS:
        raise GeneralIntereventError(f"the bound MAPLE package {digest[:12]}... is not a source this Windowed T capability is "
                                     "qualified for (only the pinned merged MAPLE39 package; refused before any native work)")
    import maple

    voxel = __import__(sb.MAPLE_VOXEL_MODULE, fromlist=["WindowedVoxelColumnState"])
    return _WindowedSource(maple, str(getattr(maple, "__file__", "")), voxel.WindowedVoxelColumnState, digest)


def _check_windowed_source_hot(src: Any) -> None:
    """Cheap continuing identity guards; NEVER hashes. The recorded association must still be the imported `maple` (same module object
    and file, same Windowed class object) and, only when the process cache happens to be warm, the cached digest must equal the
    verified one. If something changed, the operator must be re-prepared (which re-verifies the source explicitly)."""
    if type(src) is not _WindowedSource:
        raise GeneralIntereventError("the Windowed source association is missing; re-prepare the operator")
    voxel = sys.modules.get(sb.MAPLE_VOXEL_MODULE)
    if (sys.modules.get("maple") is not src.module or str(getattr(src.module, "__file__", "")) != src.module_file
            or voxel is None or getattr(voxel, "WindowedVoxelColumnState", None) is not src.windowed_class):
        raise GeneralIntereventError("the imported MAPLE package/Windowed class is no longer the one verified at preparation; "
                                     "re-prepare the operator (no source re-hash happens in a hot call)")
    cached = sb._MAPLE_DIGEST_CACHE.get("digest")
    if cached is not None and cached != src.package_sha256:
        raise GeneralIntereventError("the process source digest no longer equals the verified Windowed source; re-prepare")


def _verify_cuda_source(profile: Any, *, available_bytes_fn: Any, reserve_bytes: int) -> Any:
    """COLD source qualification selected by the EXPECTED profile (never by which genuine record happens to be importable). The native39 profile is exactly
    the unchanged `_verify_windowed_source()`. The native9f profile verifies through the narrow dense-CUDA helper (imported here, lazily: the helper imports this
    module at its own load) with the fresh host reading and the caller reserve (which already contains every live wrapper leaf). Typed refusal otherwise."""
    if profile == CUDA_PROFILE:
        return _verify_windowed_source()
    if profile != CUDA9F_PROFILE:
        raise GeneralIntereventError(f"cuda_profile must be exactly CUDA_PROFILE or CUDA9F_PROFILE, got {profile!r}")
    from maple_syrup import dense_cuda_source as dcs

    if dcs.RECORD_RETAINED_BYTES != DENSE_SOURCE_RECORD_BYTES:
        raise GeneralIntereventError("the declared dense-source record charge differs from the helper's; refused before any hash")
    try:
        return dcs.verify_dense_cuda_source(dcs.PROFILE_NATIVE9F, available_bytes_fn=available_bytes_fn, reserve_bytes=reserve_bytes)
    except dcs.DenseCudaSourceMemoryError as exc:
        raise iw.InterEventMemoryError(f"native9f source verification: {exc}") from None
    except dcs.DenseCudaSourceError as exc:
        raise GeneralIntereventError(f"the bound MAPLE package is not the native9f source this profile requires: {exc}") from None


def _check_cuda_source_hot(src: Any, profile: Any) -> None:
    """Cheap continuing identity guard selected by the EXPECTED profile; NEVER hashes. A record of the other family, or a Windowed/foreign object under either
    label, is refused (native39: `_check_windowed_source_hot` requires the exact `_WindowedSource`; native9f: the helper requires its own sealed record named for
    the native9f profile)."""
    if profile == CUDA_PROFILE:
        _check_windowed_source_hot(src)
        return
    if profile != CUDA9F_PROFILE:
        raise GeneralIntereventError("the CUDA capability record names an unknown profile; refused")
    dcs = sys.modules.get("maple_syrup.dense_cuda_source")
    if dcs is None:
        raise GeneralIntereventError("the native9f source association cannot exist without its verifier module; re-prepare the operator")
    try:
        dcs.check_dense_cuda_source_hot(src, dcs.PROFILE_NATIVE9F)
    except dcs.DenseCudaSourceError as exc:
        raise GeneralIntereventError(f"the native9f source association is no longer valid: {exc}") from None


def _transition_bytes(gop: GeneralIntereventOperator) -> int:
    """The cold-entry admission figure: the unchanged Dense/bedless one, or (Windowed) at least the declared cold native validation
    temporaries beside the resident owner and reserves."""
    m = gop.memory
    if gop.windowed is None:
        return int(m.transition_required_bytes)
    return max(int(m.transition_required_bytes), int(m.resident_bytes + m.reserves_bytes + m.column_reserve_bytes + OBJECT_BYTES
                                                     + gop.windowed.entry_scratch_bytes))


def prepare_general_interevent(base: iw.InterEventWaterOperator, t_op: st.TranspirationOperator | None = None, *,
                               extra_reserve_bytes: int = 0, bed_storage: str = "dense", cuda_profile: Any = None,
                               host_available_bytes_fn: Any = None,
                               host_reserve_bytes: int = 0) -> GeneralIntereventOperator:
    """Bind the base owner and a native `TranspirationOperator` on the SAME column. Order: type/storage/backend fences, the
    pure composed requirement ADMITTED (before any allocation, copy, hash or read of a supplied operator's values), then
    either the native factory with the composed extra reserve (t_op None) or cold validation of the supplied operator against
    the column's canonical layer properties. The bound operator is a detached copy holding immutable owned bounds.
    `bed_storage="dense"` (default) is the unchanged Dense/bedless behaviour; `"windowed"` is the explicit opt-in CPU profile for an
    actual published full/v1 native39 Windowed bed (fixed material, water only; see `WINDOWED_PROFILE`).

    `cuda_profile=CUDA_PROFILE` is the separate EXPLICIT, UNQUALIFIED opt-in for a CUDA base (genuine native39 Dense or bedless fixed
    material only; Windowed and any other MAPLE package are refused; a supplied `t_op` is unsupported - the owner mints the genuine
    primitive itself). `host_available_bytes_fn`/`host_reserve_bytes` are CUDA-only host-RAM controls (CPU refuses them)."""
    try:
        iw._check_prepared(base)
    except iw.InterEventWaterError as exc:
        raise GeneralIntereventError(f"base must be a genuine, valid InterEventWaterOperator: {exc}") from None
    windowed = _fence_storage(base, bed_storage)  # metadata only: no source tree is read or hashed here
    host_reserve = ss._checked_int(host_reserve_bytes, "host_reserve_bytes")
    if host_available_bytes_fn is not None and not callable(host_available_bytes_fn):
        raise GeneralIntereventError("host_available_bytes_fn must be callable or None")
    if base.backend not in ("numpy", "numba"):
        if cuda_profile is None:
            raise GeneralIntereventError("this milestone fences in the NumPy/Numba backends; CUDA is authored but unqualified "
                                         "(the separate explicit cuda_profile opt-in selects the UNQUALIFIED CUDA capability)")
        if type(cuda_profile) is not str or cuda_profile not in (CUDA_PROFILE, CUDA9F_PROFILE):
            raise GeneralIntereventError(f"cuda_profile must be exactly CUDA_PROFILE or CUDA9F_PROFILE, got {cuda_profile!r}")
        if base.backend != "cuda" or base.xp is np or windowed is not None:
            raise GeneralIntereventError("the CUDA capability needs a CUDA owner and is never combined with the Windowed profile")
        if t_op is not None:
            raise GeneralIntereventError("a supplied transpiration operator is unsupported on CUDA (the owner prepares the genuine "
                                         "primitive itself; an arbitrary RawKernel is never accepted by type or name)")
        return _prepare_cuda(base, ss._checked_int(extra_reserve_bytes, "extra_reserve_bytes"), host_available_bytes_fn,
                             host_reserve, cuda_profile)
    if cuda_profile is not None or host_available_bytes_fn is not None or host_reserve != 0:
        raise GeneralIntereventError("cuda_profile and the host staging controls are CUDA-only; CPU uses the column quota")
    extra = ss._checked_int(extra_reserve_bytes, "extra_reserve_bytes")
    column = base.op.column
    memory = estimate_general_memory(base, extra_reserve_bytes=extra)
    source = None
    if windowed is not None:
        # the COLD source qualification stage is admitted (with the resident owner and reserves) BEFORE any file is read or hashed; a
        # warm process cache needs no stage. The stage's temporaries are freed before the native T preparation below runs.
        stage = _source_stage_bytes() if _windowed_source_cold() else 0
        need = memory.cold_prepare_required_bytes if t_op is None else memory.supplied_prepare_required_bytes
        source_need = (memory.resident_bytes + memory.reserves_bytes + memory.column_reserve_bytes + OBJECT_BYTES + stage
                       if stage else 0)
        _admit(base, max(need, source_need), "Windowed preparation (cold source stage included)" if stage else "Windowed preparation")
        source = _verify_windowed_source()
        windowed = WindowedTProfile(maple_package_sha256=source.package_sha256, **windowed)
    if t_op is None:
        _admit(base, memory.cold_prepare_required_bytes, "preparation")
        try:
            t_op = st.prepare_transpiration(column, extra_reserve_bytes=memory.cold_prepare_extra_reserve_bytes)
        except ss.SubsurfaceMemoryError as exc:
            raise iw.InterEventMemoryError(str(exc)) from None
        except ss.SubsurfaceError as exc:
            raise GeneralIntereventError(str(exc)) from None
    else:
        _check_t_metadata(column, t_op)  # metadata first
        _admit(base, memory.supplied_prepare_required_bytes, "preparation with a supplied operator")
    owned = _own_t_operator(column, t_op)
    h = hashlib.sha256(b"maple_syrup.interevent_transpiration.operator/2\0")
    pieces = [GENERAL_PROFILE, base.sha256, str(extra), str(base.n_layers), str(base.n_cells),
              repr(tuple(float(v) for v in owned.residual_xp)), repr(tuple(float(v) for v in owned.saturation_xp)),
              st.VERSION, type(base.op.bed).__name__]
    if windowed is not None:  # a Windowed operator is a different identity; the default digest above is unchanged
        pieces += [WINDOWED_PROFILE, json.dumps(windowed.as_dict(), sort_keys=True)]
    for piece in pieces:
        h.update(piece.encode())
        h.update(b"\0")
    return GeneralIntereventOperator(_SEAL, backend=base.backend, base=base, extra_reserve_bytes=extra, memory=memory,
                                     n_cells=base.n_cells, n_layers=base.n_layers, sha256=h.hexdigest(), t_op=owned,
                                     xp=base.xp, windowed=windowed, windowed_source=source, cuda=None)


def _prepare_cuda(base: iw.InterEventWaterOperator, extra: int, host_fn: Any, host_reserve: int,
                  profile: str = CUDA_PROFILE) -> GeneralIntereventOperator:
    """Cold CUDA preparation. Order: current-device gate, device admission (metadata arithmetic), HOST admission including the cold
    source stage, ONLY THEN the streamed native39 source verification, the genuine native factory (compile/warm-up is setup), the
    immutable host canonical bounds and fresh detached device copies, and a first parameter-packet check."""
    column, xp = base.op.column, base.xp
    _device_gate(column)
    memory = estimate_general_memory(base, extra_reserve_bytes=extra)
    _admit(base, memory.cold_prepare_required_bytes, "CUDA preparation")
    if host_fn is None:  # a caller-supplied callable is honoured whatever its truthiness (a zero-budget callable may be False-valued)
        host_fn = ss.default_host_available_bytes
    hm = estimate_cuda_owner_memory(base.n_layers, base.n_cells, host_reserve_bytes=host_reserve,
                                    source_cold=_windowed_source_cold(), source_record_bytes=_cuda_source_record_extra_bytes(profile))
    try:
        ls._admit_host_build(hm.cold_required_bytes, host_fn)
    except ls.LayeredMemoryError as exc:
        raise iw.InterEventMemoryError(f"CUDA preparation (cold host stage included): {exc}") from None
    except ls.LayeredStormError as exc:
        raise GeneralIntereventError(str(exc)) from None
    source = _verify_cuda_source(profile, available_bytes_fn=host_fn, reserve_bytes=host_reserve)  # genuine pinned source, recorded for cheap hot guards
    try:
        prepared = st.prepare_transpiration(column, extra_reserve_bytes=memory.cold_prepare_extra_reserve_bytes)
    except ss.SubsurfaceMemoryError as exc:
        raise iw.InterEventMemoryError(str(exc)) from None
    except ss.SubsurfaceError as exc:
        raise GeneralIntereventError(str(exc)) from None
    kernel = prepared.kernel
    if (type(prepared) is not st.TranspirationOperator or prepared.column is not column or prepared.xp is not xp
            or prepared.backend != "cuda" or type(kernel) is not xp.RawKernel or getattr(kernel, "name", None) != "withdraw_fused"):
        raise GeneralIntereventError("the native factory did not return the genuine CUDA withdrawal operator for this column")
    residual, satb = _canonical_bounds(column)
    host_res, host_sat = ev._immutable_host(residual), ev._immutable_host(satb)
    # fresh DETACHED device copies of the canonical host vectors; nothing the primitive's own preparation allocated is retained
    owned = replace(prepared, residual_xp=xp.ascontiguousarray(xp.array(host_res, dtype=np.float64)),
                    saturation_xp=xp.ascontiguousarray(xp.array(host_sat, dtype=np.float64)))
    record = _CudaOwner(host_residual=host_res, host_saturation=host_sat, kernel=kernel, kernel_source=st._CUDA_SRC, source=source,
                        host_available_bytes_fn=host_fn, host_reserve_bytes=host_reserve, memory=hm, profile=profile)
    h = hashlib.sha256(b"maple_syrup.interevent_transpiration.operator/2\0")
    pieces = [GENERAL_PROFILE, base.sha256, str(extra), str(base.n_layers), str(base.n_cells),
              repr(tuple(float(v) for v in host_res)), repr(tuple(float(v) for v in host_sat)),
              st.VERSION, type(base.op.bed).__name__, profile, source.package_sha256]
    for piece in pieces:
        h.update(piece.encode())
        h.update(b"\0")
    gop = GeneralIntereventOperator(_SEAL, backend=base.backend, base=base, extra_reserve_bytes=extra, memory=memory,
                                    n_cells=base.n_cells, n_layers=base.n_layers, sha256=h.hexdigest(), t_op=owned, xp=xp,
                                    windowed=None, windowed_source=None, cuda=record)
    _check_gop_meta(gop)
    _check_cuda_values(gop)  # the detached device copies equal the canonical evidence (one (2L) packet)
    return gop


def _host(a: Any) -> Any:
    return a.get() if hasattr(a, "get") else a


# --- state -------------------------------------------------------------------------------------------------------------
_MINTED_STATES = weakref.WeakSet()
_MINTED_REQUESTS = weakref.WeakSet()
_STATE_SCALARS = ("t_s", "generation", "revision", "entry_generation", "initial_sequence", "next_request_sequence",
                  "profile_sha256", "base_prepared_sha256", "origin_s", "anchor_surface_m3", "anchor_soil_m3",
                  "last_request_identity")


class GeneralIntereventState:
    """One accepted whole-water interevent instant under the requested-transpiration profile. Arrays are the actual physical
    arrays (shared by reference with the parent where unchanged): `surface_m`, `qx_m2_s`, `qy_m2_s`, `soil`, `history`,
    `bed` (None in this milestone), the composed `ledger`, the window `anchor_*` (the source storm's ORIGINAL anchors and
    anchor ledger), the immutable T anchor, the genuine `StormEndReceipt` and the counters. Factory-only and immutable: its
    canonical scalar `digest` (also the parent token a request binds) detects an in-place edit of any scalar/ledger."""

    __slots__ = ("__weakref__", "anchor_ledger", "bed", "digest", "history", "ledger", "qx_m2_s", "qy_m2_s", "soil",
                 "source", "surface_m", "t_anchor", *_STATE_SCALARS)

    def __init__(self, seal: object, **fields: Any) -> None:
        if seal is not _SEAL:
            raise GeneralIntereventError("GeneralIntereventState is minted only by enable_requested_transpiration()/"
                                         "request_transpiration()/general_interevent_water_step()")
        for name in self.__slots__:
            if name in ("__weakref__", "digest"):
                continue
            object.__setattr__(self, name, fields.pop(name))
        if fields:
            raise GeneralIntereventError(f"internal: unexpected state fields {sorted(fields)}")
        object.__setattr__(self, "digest", _state_digest(self))
        _MINTED_STATES.add(self)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("GeneralIntereventState is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("GeneralIntereventState is immutable")

    def __repr__(self) -> str:
        return (f"GeneralIntereventState(t_s={self.t_s!r}, generation={self.generation}, revision={self.revision}, "
                f"next_request_sequence={self.next_request_sequence})")

    @property
    def hydro_generation(self) -> int:
        return self.generation


def _state_digest(s: Any) -> str:
    body = {name: getattr(s, name) for name in _STATE_SCALARS}
    body["ledger"] = {"hydrology": asdict(s.ledger.hydrology), "transpiration": asdict(s.ledger.transpiration)}
    body["anchor_ledger"] = asdict(s.anchor_ledger)
    body["t_anchor"] = asdict(s.t_anchor)
    body["source"] = s.source.digest
    return _digest("maple_syrup.interevent_transpiration.state/1", body)


def _mint(gop: GeneralIntereventOperator, parent_fields: dict, **over: Any) -> GeneralIntereventState:
    fields = dict(parent_fields)
    fields.update(over)
    fields["profile_sha256"] = gop.sha256
    fields["base_prepared_sha256"] = gop.base.sha256
    return GeneralIntereventState(_SEAL, **fields)


def _fields_of(s: GeneralIntereventState) -> dict:
    return {name: getattr(s, name) for name in s.__slots__ if name not in ("__weakref__", "digest")}


def _check_general_state(gop: GeneralIntereventOperator, state: Any) -> float:
    """Scalar authenticity, counters, T ledger and the shared physical/source validation; returns `t0`."""
    if type(state) is not GeneralIntereventState or state not in _MINTED_STATES:
        raise GeneralIntereventError(f"state must be a GeneralIntereventState minted by this module, got {type(state).__name__}")
    try:
        if _state_digest(state) != state.digest:
            raise GeneralIntereventError("the state's scalars/ledgers were altered after it was minted; refused")
    except (ValueError, TypeError, AttributeError) as exc:
        raise GeneralIntereventError(f"the state carries a blank, non-finite or foreign-typed scalar: {exc}") from None
    if state.profile_sha256 != gop.sha256 or state.base_prepared_sha256 != gop.base.sha256:
        raise GeneralIntereventError("the state is bound to a different requested-transpiration profile; refused")
    if type(state.ledger) is not GeneralWaterLedger:
        raise GeneralIntereventError("state.ledger must be a genuine GeneralWaterLedger")
    gen = _count(state.generation, "state.generation")
    entry = _count(state.entry_generation, "state.entry_generation")
    revision = _count(state.revision, "state.revision")
    seq0 = _count(state.initial_sequence, "state.initial_sequence")
    seq = _count(state.next_request_sequence, "state.next_request_sequence")
    _check_tledger(state.ledger.transpiration)
    _check_tledger(state.t_anchor)
    n_req = int(state.ledger.transpiration.n_requests)
    if gen < entry or revision != (gen - entry) + n_req or seq != seq0 + n_req or n_req < int(state.t_anchor.n_requests):
        raise GeneralIntereventError("revision/generation/sequence counters disagree with the ledger; refused")
    for name in ("requested_m3", "actual_m3", "unmet_m3", "soil_bound_m3"):
        if getattr(state.ledger.transpiration, name) < getattr(state.t_anchor, name):
            raise GeneralIntereventError(f"T ledger {name} predates its anchor")
    if state.last_request_identity is not None:
        _ident(state.last_request_identity, "state.last_request_identity")
    if (state.last_request_identity is None) != (n_req == int(state.t_anchor.n_requests)):
        raise GeneralIntereventError("last request identity disagrees with the accepted request count")
    try:
        return iw._check_state_core(
            gop.base, t_s=state.t_s, surface_m=state.surface_m, qx_m2_s=state.qx_m2_s, qy_m2_s=state.qy_m2_s,
            soil=state.soil, history=state.history, bed=state.bed, ledger=state.ledger.hydrology, generation=gen,
            prepared_sha256=state.base_prepared_sha256, origin_s=state.origin_s,
            anchor_surface_m3=state.anchor_surface_m3, anchor_soil_m3=state.anchor_soil_m3,
            anchor_ledger=state.anchor_ledger, source=state.source)
    except iw.InterEventWaterError as exc:
        raise GeneralIntereventError(str(exc)) from None


def audit_general_interevent(gop: GeneralIntereventOperator, state: GeneralIntereventState) -> dict:
    """Actual inventories plus the anchored WHOLE-water window closure (never a saved pass bit):

        (V - V_anchor) + hydro export + drainage + surfaceE + soilE + accepted T - rain = 0

    The MAPLE FP64 bound combines the hydrology bound of the base audit, the native T-soil bounds accumulated since the T
    anchor, rounding of the new additions into the large cumulative requested/actual/unmet/bound (n + 1 operations) and the
    one extra algebra term at the largest scale. Hydro operations and T transactions are counted separately."""
    _check_gop_meta(gop)
    _admit(gop, gop.memory.audit_required_bytes, "audit")  # whole owner + wrapper residents/reserves, not the base alone
    _admit_host(gop, gop.cuda.memory.hot_required_bytes if gop.cuda is not None else 0, "audit (host)")
    return _audit(gop, state)


def _audit(gop: GeneralIntereventOperator, state: GeneralIntereventState) -> dict:
    """The audit body WITHOUT its own admission, for callers that already admitted a larger composed requirement (the
    transition and the late audit of a transaction; its temporaries are in `owner_check_scratch_bytes`)."""
    _check_gop(gop)
    _check_general_state(gop, state)
    h, soil = iw._fresh_inventories(gop.base, state.t_s, state.surface_m, state.qx_m2_s, state.qy_m2_s, state.soil,
                                    state.history)
    led, anc = state.ledger.hydrology, state.anchor_ledger
    t, ta = state.ledger.transpiration, state.t_anchor
    rain = float(led.rain_m3) - float(anc.rain_m3)
    export = float(led.export_m3) - float(anc.export_m3)
    drain = float(led.drainage_m3) - float(anc.drainage_m3)
    es = float(led.surface_evaporation_m3) - float(anc.surface_evaporation_m3)
    eso = float(led.soil_evaporation_m3) - float(anc.soil_evaporation_m3)
    tr = float(t.actual_m3) - float(ta.actual_m3)
    req = float(t.requested_m3) - float(ta.requested_m3)
    unmet = float(t.unmet_m3) - float(ta.unmet_m3)
    residual = ((((h - float(state.anchor_surface_m3)) + (soil - float(state.anchor_soil_m3))) + export) + drain
                + es + eso + tr) - rain
    n_t = int(t.n_requests) - int(ta.n_requests)
    bound = iw._window_bound_core(gop.base, led, anc, state.anchor_surface_m3, state.anchor_soil_m3, h, soil, rain, export,
                                  drain, es, eso, extra_scale=max(tr, 0.0))
    bound += (float(t.soil_bound_m3) - float(ta.soil_bound_m3))
    for name in ("requested_m3", "actual_m3", "unmet_m3", "soil_bound_m3"):
        bound += volume_roundoff_bound_m3(n_t + 1, max(abs(float(getattr(t, name))), abs(float(getattr(ta, name)))))
    bound += volume_roundoff_bound_m3(2, max(h, soil, tr, float(state.anchor_surface_m3), float(state.anchor_soil_m3)))
    if not math.isfinite(bound) or bound < 0.0:
        raise GeneralIntereventError("non-finite anchored closure bound")
    if not math.isfinite(residual) or abs(residual) > bound:
        raise GeneralIntereventError(f"anchored whole-water closure failed: {residual} m3 against {bound} m3")
    return dict(surface_m3=h, soil_m3=soil, rain_m3=rain, export_m3=export, drainage_m3=drain,
                surface_evaporation_m3=es, soil_evaporation_m3=eso, transpiration_m3=tr, requested_m3=req,
                unmet_m3=unmet, n_requests=n_t, residual_m3=residual, bound_m3=bound)


# --- transition --------------------------------------------------------------------------------------------------------
def _windowed_cold_entry(gop: GeneralIntereventOperator, t0_state: iw.InterEventState) -> None:
    """The ONE cold native validation of the Windowed profile (admitted with `entry_scratch_bytes` before it runs): the entered bed
    must be the operator-bound genuine Windowed bed (same voxel column/store object, no pending journal) and pass the native
    NON-SCAN validators (bed/context support, availability, ledger/committed, water, zero mobile, ledger reconciliation) already
    qualified for the Windowed layered codec. It reads no deep chunk, hashes nothing, flushes/commits/publishes nothing; no later T
    or hydro step revisits it (the storm operator's identity bindings guard the unchanged material)."""
    bed, ref = t0_state.bed, gop.base.op.bed
    if type(bed) is not sb.BedState or bed.voxel_column is not ref.voxel_column or bed.voxel_column.pending_deep is not None:
        raise GeneralIntereventError("the entered bed is not the operator-bound published Windowed bed; refused")
    from maple_syrup import layered_checkpoint as lc

    try:
        lc._native_checks_windowed(bed, gop.base.op.context, float(t0_state.t_s), type(ref.ledger))
    except lc.LayeredCheckpointError as exc:
        raise GeneralIntereventError(f"cold native Windowed validation failed: {exc}") from None


def enable_requested_transpiration(gop: GeneralIntereventOperator, t0_state: iw.InterEventState, *, expected_t_s: Any,
                                   expected_generation: Any, initial_sequence: Any = 0) -> GeneralIntereventState:
    """Cold controlled transition from a GENUINE, already-entered, valid T0 `InterEventState` (source end or after real E
    steps) to the general profile. The actual `audit_interevent(base, t0_state)` passes first; every physical array, the
    actual `StormEndReceipt`, the original anchors and the hydrology ledger are preserved EXACTLY (shared by reference; no
    storm re-entry, no soil/history reset, no new initial-water anchor, no forced topography commit). The T account starts
    at an all-zero immutable anchor. The result is bound to the NEW profile, which every T0 public API refuses."""
    _check_gop_meta(gop)
    _admit(gop, _transition_bytes(gop), "transition")
    _admit_host(gop, gop.cuda.memory.hot_required_bytes if gop.cuda is not None else 0, "transition (host)")
    _check_gop_values(gop)
    t_exp, gen_exp = _real(expected_t_s, "expected_t_s"), _count(expected_generation, "expected_generation")
    seq0 = _count(initial_sequence, "initial_sequence")
    if type(t0_state) is not iw.InterEventState:
        raise GeneralIntereventError(f"t0_state must be a genuine T0 InterEventState, got {type(t0_state).__name__}")
    if t0_state.t_s != t_exp or t0_state.generation != gen_exp:
        raise GeneralIntereventError("stale parent: the T0 state's clock/generation differ from the expected values; refused")
    if gop.windowed is not None:
        # Windowed profile: its qualification is the CPU storm-origin lane only. The authenticated boundary-origin source
        # (metadata only, before any audit or native work) is refused explicitly, never silently accepted or rewritten.
        try:
            iw.require_storm_origin(t0_state.source, "the Windowed requested-transpiration profile")
        except iw.InterEventWaterError as exc:
            raise GeneralIntereventError(str(exc)) from None
    try:
        iw.audit_interevent(gop.base, t0_state)  # the real audit on the genuine state, before anything is built
    except iw.InterEventWaterError as exc:
        raise GeneralIntereventError(f"the T0 state failed its genuine audit: {exc}") from None
    if gop.windowed is not None:
        _windowed_cold_entry(gop, t0_state)
    zero = TranspirationLedger()
    state = GeneralIntereventState(
        _SEAL, t_s=float(t0_state.t_s), surface_m=t0_state.surface_m, qx_m2_s=t0_state.qx_m2_s, qy_m2_s=t0_state.qy_m2_s,
        soil=t0_state.soil, history=t0_state.history, bed=t0_state.bed,
        ledger=GeneralWaterLedger(t0_state.ledger, zero), anchor_ledger=t0_state.anchor_ledger, t_anchor=zero,
        source=t0_state.source, generation=int(t0_state.generation), revision=0,
        entry_generation=int(t0_state.generation), initial_sequence=seq0, next_request_sequence=seq0,
        profile_sha256=gop.sha256, base_prepared_sha256=gop.base.sha256, origin_s=float(t0_state.origin_s),
        anchor_surface_m3=float(t0_state.anchor_surface_m3), anchor_soil_m3=float(t0_state.anchor_soil_m3),
        last_request_identity=None)
    _audit(gop, state)  # full physical/source/window audit of the product before it is returned (admitted above)
    return state


# --- the request -------------------------------------------------------------------------------------------------------
class WithdrawalRequest:
    """Factory-prepared immutable aggregate request: OWNED float64 `(L, N)` water-DEPTH AMOUNTS in metres (not rates), a
    bounded opaque identity, the request sequence and the exact parent (token, revision, clock, hydro generation) it was
    prepared for, plus the payload SHA-256. The values are validated natively on EVERY cell at withdrawal (finite,
    nonnegative, 0 on inactive cells); preparation checks type/shape/device only. Host payloads are immutable bytes; a CUDA
    payload is private and inspected only through a detached admitted copy (`payload()`)."""

    __slots__ = ("__weakref__", "_column", "_depth", "_parent", *_REQUEST_SCALARS)

    def __init__(self, seal: object, **fields: Any) -> None:
        if seal is not _SEAL:
            raise GeneralIntereventError("WithdrawalRequest is minted only by prepare_withdrawal_request()")
        for name in self.__slots__:
            if name != "__weakref__":
                object.__setattr__(self, name, fields.pop(name))
        if fields:
            raise GeneralIntereventError(f"internal: unexpected request fields {sorted(fields)}")
        _MINTED_REQUESTS.add(self)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("WithdrawalRequest is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("WithdrawalRequest is immutable")

    def __repr__(self) -> str:
        return f"WithdrawalRequest(identity={self.identity!r}, sequence={self.sequence}, payload={self.payload_sha256[:12]}...)"

    def payload(self) -> Any:
        """The immutable host array, or on a device a fresh DETACHED (admitted) copy."""
        if isinstance(self._depth, np.ndarray):
            return self._depth
        column = self._column
        _device_gate(column)  # the current device must be the column's before any quota query or allocation
        avail = column.available_bytes_fn()
        need = int(self._depth.nbytes) + int(column.reserve_bytes)
        if need > int(avail):
            raise iw.InterEventMemoryError(f"request inspection needs {need} B but {int(avail)} B are available")
        return self._depth.copy()

    def scalars(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in _REQUEST_SCALARS}


def _request_header(gop: GeneralIntereventOperator, identity: str, sequence: int, parent_token: str, revision: int,
                    t_s: float, generation: int) -> dict:
    return {"identity": identity, "sequence": sequence, "parent_token": parent_token, "expected_revision": revision,
            "expected_t_s": t_s, "expected_generation": generation, "profile_sha256": gop.sha256,
            "shape": [gop.n_layers, gop.n_cells], "units": "m_water_depth_per_cell_area_amount"}


def prepare_withdrawal_request(gop: GeneralIntereventOperator, parent: GeneralIntereventState, depth_m: Any, *,
                               identity: Any) -> WithdrawalRequest:
    """Copy and hash ONCE a caller aggregate depth `(L, N)` (a float64 array of the column namespace/device) into an owned
    request for `parent`'s exact token/revision/clock/sequence. Admission precedes the copy/hash; no value is read except by
    the cold hash. No dt/PET conversion happens and an all-zero request is a valid, sequence-consuming request."""
    _check_gop_meta(gop)
    ident = _ident(identity, "identity")
    _admit(gop, gop.memory.request_prepare_required_bytes, "request preparation")
    if gop.cuda is not None:  # host quota (parameter packet + hash staging chunk) BEFORE the packet, copy and hash
        _admit_host(gop, gop.cuda.memory.request_prepare_required_bytes, "request preparation (host staging)")
    _check_gop_values(gop)
    if gop.base.op.backend == "cuda" and gop.cuda is None:
        try:
            ls._admit_host_build(gop.memory.request_prepare_host_bytes + OBJECT_BYTES, ss.default_host_available_bytes)
        except ls.LayeredMemoryError as exc:
            raise iw.InterEventMemoryError(str(exc)) from None
    _check_general_state(gop, parent)
    column, xp = gop.base.op.column, gop.xp
    try:
        ss._check_dynamic(column, "depth_m", depth_m, (gop.n_layers, gop.n_cells))
    except ss.SubsurfaceError as exc:
        raise TranspirationRequestError(str(exc)) from None
    owned = (ev._immutable_host(np.asarray(depth_m, dtype=np.float64)) if xp is np
             else xp.ascontiguousarray(xp.array(depth_m, dtype=np.float64, copy=True)))
    header = _request_header(gop, ident, int(parent.next_request_sequence), parent.digest, int(parent.revision),
                             float(parent.t_s), int(parent.generation))
    h = hashlib.sha256(b"maple_syrup.interevent_transpiration.request/1\0" + json.dumps(header, sort_keys=True).encode())
    ls._stream_hash(h, owned)
    return WithdrawalRequest(_SEAL, _column=column, _depth=owned, _parent=weakref.ref(parent), identity=ident, sequence=header["sequence"],
                             parent_token=parent.digest, expected_revision=header["expected_revision"],
                             expected_t_s=header["expected_t_s"], expected_generation=header["expected_generation"],
                             profile_sha256=gop.sha256, payload_sha256=h.hexdigest())


def _check_request(gop: GeneralIntereventOperator, request: Any) -> None:
    if type(request) is not WithdrawalRequest or request not in _MINTED_REQUESTS:
        raise TranspirationRequestError("request must be a WithdrawalRequest prepared by prepare_withdrawal_request()")
    _ident(request.identity, "request.identity")
    _sha(request.payload_sha256, "request.payload_sha256")
    _sha(request.parent_token, "request.parent_token")
    _count(request.sequence, "request.sequence")
    _count(request.expected_revision, "request.expected_revision")
    _count(request.expected_generation, "request.expected_generation")
    _nonneg(request.expected_t_s, "request.expected_t_s")
    if request.profile_sha256 != gop.sha256:
        raise TranspirationRequestError("the request was prepared for a different requested-transpiration profile; refused")
    if request._column is not gop.base.op.column:
        raise TranspirationRequestError("the request was prepared on a different column; refused")
    depth = request._depth
    try:
        ss._check_dynamic(gop.base.op.column, "request.depth_m", depth, (gop.n_layers, gop.n_cells))
    except ss.SubsurfaceError as exc:
        raise TranspirationRequestError(str(exc)) from None
    if isinstance(depth, np.ndarray) and depth.flags.writeable:
        raise TranspirationRequestError("the request payload lost its immutability; refused")


# --- the zero-time withdrawal ------------------------------------------------------------------------------------------
class TranspirationReceipt(NamedTuple):
    """Bounded scalar record of one accepted request (no arrays)."""

    identity: str
    sequence: int
    payload_sha256: str
    parent_token: str
    revision_before: int
    revision_after: int
    t_s: float
    requested_m3: float
    actual_m3: float
    unmet_m3: float


@dataclass(frozen=True, eq=False)
class TranspirationCandidate:
    """One proposed T transaction: the fresh whole `state`, the native `actual_m`/`unmet_m`/`scale` `(L, N)` feedback arrays,
    the native `TranspirationBudget`, its memory record and the scalar `receipt`. The caller adopts `state` whole or drops the
    candidate; adoption is external (this function holds no registry). The returned arrays may be retained by the caller
    after adopting the state (declare them as `retained_feedback_bytes` on the next hydro step)."""

    state: GeneralIntereventState
    actual_m: Any
    unmet_m: Any
    scale: Any
    budget: st.TranspirationBudget
    memory: st.TranspirationMemory
    receipt: TranspirationReceipt


def request_transpiration(gop: GeneralIntereventOperator, accepted: GeneralIntereventState, request: WithdrawalRequest, *,
                          expected_revision: Any, phase: str = INTEREVENT) -> TranspirationCandidate:
    """ONE pure ZERO-TIME T transaction at the accepted instant. Order: metadata/profile/phase/request binding (before any
    trial), whole-owner admission, the REAL native `withdraw_transpiration` on the SAME soil once (availability, residual
    floor, actual, unmet and scale all from the primitive; the clock must be returned unchanged), an owner per-cell packet,
    the new state and the full whole-water audit. Any failure (including the late audit) discards the proposed soil and
    leaves every accepted array, ledger, counter and the request untouched. There is no CFL/retry: it has no dt."""
    _check_gop_meta(gop)
    if type(phase) is not str or phase != INTEREVENT:
        raise TranspirationRequestError(f"phase must be exactly {INTEREVENT!r}; wind/rain/recession spend no T demand "
                                        f"(got {phase!r})")
    exp_rev = _count(expected_revision, "expected_revision")
    _check_request(gop, request)
    _admit(gop, gop.memory.withdraw_required_bytes, "withdrawal")  # whole owner + native T terms, before any state array read
    _admit_host(gop, gop.cuda.memory.hot_required_bytes if gop.cuda is not None else 0, "withdrawal (host)")
    _check_gop_values(gop)
    t0 = _check_general_state(gop, accepted)
    if exp_rev != accepted.revision:
        raise TranspirationRequestError(f"stale parent: expected revision {exp_rev} but the accepted state is at "
                                        f"{accepted.revision}; refused before any trial")
    if request.sequence < accepted.next_request_sequence:
        raise TranspirationRequestError(f"duplicate/stale request: sequence {request.sequence} was already consumed "
                                        f"(next is {accepted.next_request_sequence}); refused before any trial")
    if request._parent() is not accepted:
        raise TranspirationRequestError("the request was prepared for a different physical parent instance (a distinct "
                                        "branch can share every scalar and label, or the parent is gone); refused before "
                                        "any trial")
    if (request.sequence != accepted.next_request_sequence or request.parent_token != accepted.digest
            or request.expected_revision != accepted.revision or request.expected_generation != accepted.generation
            or request.expected_t_s != t0):
        raise TranspirationRequestError("the request is not bound to this accepted state (token/revision/generation/"
                                        "clock/sequence); refused before any trial")
    try:
        res = st.withdraw_transpiration(gop.t_op, accepted.soil, request._depth,
                                        extra_reserve_bytes=gop.memory.withdraw_extra_reserve_bytes)
    except ss.SubsurfaceMemoryError as exc:
        raise iw.InterEventMemoryError(str(exc)) from None
    except ss.SubsurfaceError as exc:
        raise TranspirationRequestError(f"native withdrawal refused: {exc}") from None
    except (ValueError, OverflowError, FloatingPointError) as exc:
        # the unmodified primitive lets a finite request whose SUM overflows reach `volume_roundoff_bound_m3` as a bare
        # ValueError (reported to the root as a native seam); nothing was published, so it is a typed refusal here
        raise TranspirationRequestError(f"native withdrawal refused (non-finite aggregate volume): {exc}") from None
    if res.state.t_s != t0 or type(res.state) is not ss.SoilColumnState:
        raise TranspirationRequestError(f"the native withdrawal returned clock {res.state.t_s!r}, required {t0!r}; refused")
    _owner_check(gop, accepted.soil.water_m, res, request._depth)
    t_led = accepted.ledger.transpiration.accepted(res.budget)
    fields = _fields_of(accepted)
    new = _mint(gop, fields, soil=res.state, ledger=GeneralWaterLedger(accepted.ledger.hydrology, t_led),
                revision=int(accepted.revision) + 1, next_request_sequence=int(accepted.next_request_sequence) + 1,
                last_request_identity=request.identity)
    try:
        _audit(gop, new)  # late whole-water audit (admitted by the withdrawal); failure discards `new` and the proposed soil
    except GeneralIntereventError as exc:
        raise TranspirationRequestError(f"late whole-water audit refused the candidate: {exc}") from None
    b = res.budget
    receipt = TranspirationReceipt(request.identity, int(request.sequence), request.payload_sha256, request.parent_token,
                                   int(accepted.revision), int(new.revision), float(t0), float(b.requested_m3),
                                   float(b.actual_m3), float(b.unmet_m3))
    return TranspirationCandidate(state=new, actual_m=res.actual_m, unmet_m=res.unmet_m, scale=res.scale, budget=b,
                                  memory=res.memory, receipt=receipt)


def _owner_check(gop: GeneralIntereventOperator, w_old: Any, res: st.TranspirationResult, q: Any) -> None:
    """Independent per-layer/cell VALIDATION of the primitive's raw outputs against its own published law (a second solver is
    NOT introduced: the primitive remains the only physical computation). A total-volume closure cannot see a swapped layer,
    an offsetting local error or a replaced feedback array, so each layer/cell must satisfy, with the canonical residual
    depth of the bound operator: everything finite; actual == min(request, max(w - residual, 0)) (so 0 <= actual <= request
    and <= the withdrawable depth); unmet == request - actual >= 0; new == old - actual; scale == actual / request for a
    positive request and 0 otherwise (the common fulfilment fraction LEAF must reuse); inactive water unchanged. One host
    packet; the per-layer loop bounds the scratch (`owner_check_scratch_bytes`)."""
    xp = gop.xp
    inactive = gop.base.op._inactive
    # CUDA: the immutable HOST canonical vector (Python floats), never `float(device[layer])` (one sync per layer)
    residual = gop.cuda.host_residual if gop.cuda is not None else gop.t_op.residual_xp
    from maple.core.backend import errstate

    bad = xp.zeros(gop.n_cells, dtype=np.int64)
    new, act, unm, scl = res.state.water_m, res.actual_m, res.unmet_m, res.scale
    with errstate(xp=xp, all="ignore"):
        for layer in range(gop.n_layers):
            ql, wl, al = q[layer], w_old[layer], act[layer]
            expect = xp.minimum(ql, xp.maximum(wl - float(residual[layer]), 0.0))
            fraction = xp.where(ql > 0.0, al / ql, 0.0)
            ok = (xp.isfinite(al) & xp.isfinite(unm[layer]) & xp.isfinite(scl[layer]) & xp.isfinite(new[layer])
                  & (al == expect) & (unm[layer] == ql - al) & (unm[layer] >= 0.0) & (new[layer] == wl - al)
                  & (scl[layer] == fraction) & ~(inactive & (new[layer] != wl)))
            xp.add(bad, (~ok).astype(np.int64), out=bad)
        count = xp.count_nonzero(bad)
    n_bad = int(ls._packet(xp, [count])[0])
    if n_bad != 0:
        raise TranspirationRequestError(f"owner per-cell check failed on {n_bad} cell(s): actual/unmet/scale/new water "
                                        "disagree with the request, the canonical availability or each other")


# --- further genuine hydrology/E --------------------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class GeneralIntereventCandidate:
    """One proposed real hydrology/E step: the fresh whole `state`, the shared physical `receipt`, the router's step and the
    stage `vertical` arrays (the same objects the T0 candidate carries)."""

    state: GeneralIntereventState
    receipt: iw.InterEventStepReceipt
    dt_s: float
    surface_step: SurfaceStep
    vertical: iw.InterEventVertical


def general_interevent_water_step(gop: GeneralIntereventOperator, accepted: GeneralIntereventState, pet_terrain_m_s: Any,
                                  dt_s: Any, *, phase: str, retained_feedback_bytes: Any = 0) -> GeneralIntereventCandidate:
    """ONE pure real hydrology/E trial through the SAME private physical body as the T0 `interevent_water_step`, on the
    already-T-debited physical arrays. It advances time/elapsed/n_steps/generation and revision once and carries the T
    account, request sequence and last identity unchanged. `retained_feedback_bytes` names the bytes of the prior
    transaction's request/actual/unmet/scale arrays the caller STILL holds (0 = released; see `memory.feedback_bytes`); it is
    admitted with the whole owner. `HydraulicStepRejected` is the only recoverable failure (retry the SAME state at a
    smaller dt)."""
    return _general_step_engine(gop, accepted, pet_terrain_m_s, dt_s, phase, retained_feedback_bytes, None)


def general_interevent_recharge_step(gop: GeneralIntereventOperator, accepted: GeneralIntereventState, pet_terrain_m_s: Any,
                                     dt_s: Any, *, recharge_depth_m: Any, phase: str,
                                     retained_feedback_bytes: Any = 0) -> GeneralIntereventCandidate:
    """The Phase 1.5 daily step: the SAME single trial as `general_interevent_water_step` (one admission, one `_physical_trial`, one
    router stage, one E/diffusion) whose one column movement carries the uniform `recharge_depth_m` (m, finite, >= 0) through
    `subsurface.step_with_recharge` (no second pass, no fake dt). The candidate's receipt is a `RechargeStepReceipt`; the window ledger
    credits the applied pulse once as rain. A column that cannot hold the pulse raises `subsurface.SubsurfaceFitFailure` (nothing is
    adopted); `HydraulicStepRejected` keeps its existing meaning. The accepted record is never written. `recharge_depth_m` must be ONE real host
    scalar (not None, bool, string, array, non-finite or negative): None is the engine's private no-recharge dispatch sentinel and is never a
    valid depth here; the argument is validated before any admission, entry or trial."""
    if (isinstance(recharge_depth_m, (bool, np.bool_)) or not isinstance(recharge_depth_m, (int, float, np.integer, np.floating))
            or not math.isfinite(float(recharge_depth_m)) or float(recharge_depth_m) < 0.0):
        raise GeneralIntereventError(f"recharge_depth_m must be one finite real host number >= 0 (metres of water), got {recharge_depth_m!r}")
    return _general_step_engine(gop, accepted, pet_terrain_m_s, dt_s, phase, retained_feedback_bytes, float(recharge_depth_m))


def _general_step_engine(gop: GeneralIntereventOperator, accepted: GeneralIntereventState, pet_terrain_m_s: Any, dt_s: Any,
                         phase: str, retained_feedback_bytes: Any, recharge_depth_m: Any) -> GeneralIntereventCandidate:
    _check_gop_meta(gop)
    if type(phase) is not str or phase != INTEREVENT:
        raise GeneralIntereventError(f"phase must be exactly {INTEREVENT!r}; wind/rain/recession spend no E/T demand "
                                     f"(got {phase!r})")
    feedback = _count(retained_feedback_bytes, "retained_feedback_bytes")
    _admit(gop, gop.memory.step_required_bytes + feedback, "step")
    _admit_host(gop, gop.cuda.memory.hot_required_bytes if gop.cuda is not None else 0, "step (host)")
    _check_gop_values(gop)
    t0 = _check_general_state(gop, accepted)
    dt = _real(dt_s, "dt_s")
    if not dt > 0.0:
        raise GeneralIntereventError(f"dt_s must be > 0 (the interevent step has no dt = 0 identity), got {dt!r}")
    try:
        t1 = advance_time(t0, dt)
    except ExperimentalHydrologyError as exc:
        raise GeneralIntereventError(str(exc)) from None
    try:
        ss._check_dynamic(gop.base.op.column, "pet_terrain_m_s", pet_terrain_m_s, (gop.n_cells,))
    except ss.SubsurfaceError as exc:
        raise GeneralIntereventError(str(exc)) from None
    if recharge_depth_m is None:
        trial = iw._physical_trial(gop.base, t0, dt, t1, accepted.surface_m, accepted.qx_m2_s, accepted.qy_m2_s, accepted.soil,
                                   accepted.history, accepted.bed, pet_terrain_m_s)
    else:
        trial = iw._physical_trial(gop.base, t0, dt, t1, accepted.surface_m, accepted.qx_m2_s, accepted.qy_m2_s, accepted.soil,
                                   accepted.history, accepted.bed, pet_terrain_m_s, recharge_depth_m=recharge_depth_m)
    receipt, p, m, e, c, s, bed1 = trial
    new = _mint(gop, _fields_of(accepted), t_s=float(t1), surface_m=s.state.depth_m, qx_m2_s=s.state.qx_m2_s,
                qy_m2_s=s.state.qy_m2_s, soil=c.state, history=m.history, bed=bed1,
                ledger=GeneralWaterLedger(accepted.ledger.hydrology.advanced(receipt), accepted.ledger.transpiration),
                generation=int(accepted.generation) + 1, revision=int(accepted.revision) + 1)
    return GeneralIntereventCandidate(
        state=new, receipt=receipt, dt_s=dt, surface_step=s,
        vertical=iw.InterEventVertical(p.intake_m, m.intake_m, e.surface_evaporation_m, c.evaporation_m,
                                       c.surface_return_m, c.drainage_m))


# --- metadata / provenance --------------------------------------------------------------------------------------------
def general_metadata(gop: GeneralIntereventOperator, state: GeneralIntereventState) -> dict[str, Any]:
    """Bounded scalar metadata a later exact codec needs (no arrays): profile/base/source identities, the hydrology and T
    ledgers and anchor, counters, the original window anchors and the genuine source receipt. Not a codec."""
    _check_gop_meta(gop)  # metadata only: no canonical vector is read here (use an audit for the value check)
    _check_general_state(gop, state)
    meta = {"profile": GENERAL_PROFILE, "profile_sha256": gop.sha256, "base_prepared_sha256": gop.base.sha256,
            "storm_operator_sha256": gop.base.op.sha256, "state_digest": state.digest, "t_s": state.t_s,
            "hydro_generation": state.generation, "entry_hydro_generation": state.entry_generation,
            "revision": state.revision, "initial_sequence": state.initial_sequence,
            "next_request_sequence": state.next_request_sequence, "last_request_identity": state.last_request_identity,
            "ledger": state.ledger.as_dict(), "t_anchor": state.t_anchor.as_dict(),
            "anchor_ledger": state.anchor_ledger.as_dict(), "origin_s": state.origin_s,
            "anchor_surface_m3": state.anchor_surface_m3, "anchor_soil_m3": state.anchor_soil_m3,
            "source_kind": iw.source_kind(state.source), "source_receipt": state.source.as_dict()}
    if gop.windowed is not None:  # additive: the Dense/bedless metadata above is unchanged
        meta["profile"] = WINDOWED_PROFILE
        meta["windowed"] = gop.windowed.as_dict()
    if gop.cuda is not None:  # additive: explicit UNQUALIFIED CUDA capability
        meta["profile"] = gop.cuda.profile
        meta["cuda"] = {"maple_package_sha256": gop.cuda.source.package_sha256, "host_memory": gop.cuda.memory.as_dict()}
    return meta


def describe(gop: GeneralIntereventOperator) -> dict[str, Any]:
    _check_gop_meta(gop)
    out = {
        "operation": "explicit requested-transpiration general interevent owner (zero-time T amounts + genuine hydro/E)",
        "profile": GENERAL_PROFILE, "backend": gop.backend, "n_layers": gop.n_layers, "n_cells": gop.n_cells,
        "identity": {"sha256": gop.sha256, "base_sha256": gop.base.sha256},
        "counters": ["hydro_generation == hydrology.n_steps",
                     "revision == (hydro_generation - entry_hydro_generation) + n_requests",
                     "next_request_sequence == initial_sequence + n_requests"],
        "request": "aggregate (L, N) water-depth AMOUNTS in m; common proportional scale, no compensation, no second store",
        "replay": "pure; no global registry/CAS; stale/duplicate refused against a NEWER accepted state only",
        "memory": gop.memory.as_dict(),
        "excluded": "LEAF import/growth, calendar/scheduler, codec, Windowed/native bed, CUDA qualification, adoption",
    }
    if gop.windowed is not None:
        out["profile"] = WINDOWED_PROFILE
        out["windowed"] = gop.windowed.as_dict()
        out["excluded"] = ("LEAF import/growth, calendar/scheduler, codec, CUDA, overlay-v2/unpublished journal, native9f and any "
                           "other MAPLE source, moving material/remap, sediment transport, Windowed wind, adoption")
    if gop.cuda is not None:
        out["profile"] = gop.cuda.profile
        out["cuda"] = {"status": "UNQUALIFIED until a guarded real-device test",
                       "maple_package_sha256": gop.cuda.source.package_sha256, "host_memory": gop.cuda.memory.as_dict()}
        out["excluded"] = (_NATIVE9F_EXCLUDED if gop.cuda.profile == CUDA9F_PROFILE else "LEAF import/growth, calendar/scheduler, codec, Windowed, native9f and any other MAPLE source, GPU "
                           "lifecycle, moving material/remap, sediment transport, wind, speed/scale claims, adoption")
    return out


# --- checkpoint restoration seam (owner-controlled; used only by the verified lifecycle checkpoint codec) ------------------------
_STATE_RECORD_FLOATS = ("t_s", "origin_s", "anchor_surface_m3", "anchor_soil_m3")
_STATE_RECORD_COUNTS = ("generation", "revision", "entry_generation", "initial_sequence", "next_request_sequence")
_STATE_RECORD_KEYS = frozenset({*_STATE_RECORD_FLOATS, *_STATE_RECORD_COUNTS, "profile_sha256", "base_prepared_sha256",
                                "last_request_identity", "hydrology", "transpiration", "anchor_ledger", "t_anchor", "digest"})


def checkpoint_state_record(state: GeneralIntereventState) -> dict[str, Any]:
    """Bounded scalar record (an ENCODED tree) of a GENUINE accepted general state: no array, no source receipt (that is a
    separate record, `iw.checkpoint_receipt_record`). Refuses anything not minted by this module."""
    from maple_syrup import checkpoint_capability as cc

    if type(state) is not GeneralIntereventState or state not in _MINTED_STATES:
        raise GeneralIntereventError("only a GeneralIntereventState minted by this module can be recorded; refused")
    body = {name: getattr(state, name) for name in (*_STATE_RECORD_FLOATS, *_STATE_RECORD_COUNTS, "profile_sha256",
                                                    "base_prepared_sha256", "last_request_identity")}
    body["hydrology"] = asdict(state.ledger.hydrology)
    body["transpiration"] = asdict(state.ledger.transpiration)
    body["anchor_ledger"] = asdict(state.anchor_ledger)
    body["t_anchor"] = asdict(state.t_anchor)
    body["digest"] = state.digest
    return cc.encode_tree(body)


def _typed_ledger(cls: Any, record: Any, name: str) -> Any:
    """Exact closed-key, exact-type reconstruction of one of the cumulative ledger dataclasses (defaults define the types)."""
    import dataclasses

    fields = dataclasses.fields(cls)
    if type(record) is not dict or set(record) != {f.name for f in fields}:
        raise GeneralIntereventError(f"{name} ledger record keys mismatch (closed key set)")
    kwargs = {}
    for f in fields:
        want = type(f.default)
        value = record[f.name]
        if type(value) is not want:
            raise GeneralIntereventError(f"{name}.{f.name} must be an exact {want.__name__}")
        kwargs[f.name] = value
    return cls(**kwargs)


def _restore_checkpoint_general_state(gop: GeneralIntereventOperator, cap: Any, record: Any, receipt_record: Any, *,
                                      arrays: tuple, _source_restorer: Any = None) -> GeneralIntereventState:
    """Owner-controlled restoration of the CURRENT accepted general state from a verified checkpoint. `arrays` is the exact
    decoded `(surface_m, qx, qy, soil, history, bed)` tuple the capability bound. The scalar source receipt is restored by the
    IW owner (real mint + `_check_receipt_core`), the real `GeneralIntereventState` is minted here with this module's own seal,
    its recomputed digest must equal the saved one, and the ACTUAL `audit_general_interevent` runs before it is returned. The
    enablement/T0 path is not replayed: counters, T anchor and ledgers are the saved, accepted ones (never zeroed)."""
    from maple_syrup import checkpoint_capability as cc

    _check_gop_meta(gop)
    try:
        cc.require(cap, "t_state", record, arrays)
        tree = cc.decode_tree(record)
    except cc.CheckpointCapabilityError as exc:
        raise GeneralIntereventError(f"checkpoint state refused: {exc}") from None
    if type(tree) is not dict or set(tree) != _STATE_RECORD_KEYS:
        raise GeneralIntereventError("checkpoint state record keys mismatch (closed key set)")
    for key in _STATE_RECORD_FLOATS:
        if type(tree[key]) is not float or tree[key] < 0.0:
            raise GeneralIntereventError(f"state.{key} must be an exact non-negative float")
    for key in _STATE_RECORD_COUNTS:
        if type(tree[key]) is not int:
            raise GeneralIntereventError(f"state.{key} must be an exact integer")
        _count(tree[key], "state." + key)
    ident = tree["last_request_identity"]
    if ident is not None:
        _ident(ident, "state.last_request_identity")
    if tree["profile_sha256"] != gop.sha256 or tree["base_prepared_sha256"] != gop.base.sha256:
        raise GeneralIntereventError("the saved state is bound to a different requested-transpiration profile; refused")
    hydro = _typed_ledger(iw.WaterWindowLedger, tree["hydrology"], "hydrology")
    tled = _typed_ledger(TranspirationLedger, tree["transpiration"], "transpiration")
    anchor = _typed_ledger(iw.WaterWindowLedger, tree["anchor_ledger"], "anchor_ledger")
    t_anchor = _typed_ledger(TranspirationLedger, tree["t_anchor"], "t_anchor")
    if _source_restorer is None:  # v1 default: the storm-end receipt vocabulary only
        _source_restorer = iw._restore_checkpoint_source_receipt
    elif _source_restorer is not iw._restore_checkpoint_source_v2:  # the only other versioned restorer (closed selector, identity)
        raise GeneralIntereventError("unknown checkpoint source restorer; refused")
    try:
        source = _source_restorer(
            gop.base, cap, receipt_record, t_s=tree["t_s"], ledger=hydro, generation=tree["generation"],
            origin_s=tree["origin_s"], anchor_surface_m3=tree["anchor_surface_m3"], anchor_soil_m3=tree["anchor_soil_m3"],
            anchor_ledger=anchor)
    except iw.InterEventWaterError as exc:
        raise GeneralIntereventError(str(exc)) from None
    surface, qx, qy, soil, history, bed = arrays
    state = GeneralIntereventState(
        _SEAL, t_s=tree["t_s"], surface_m=surface, qx_m2_s=qx, qy_m2_s=qy, soil=soil, history=history, bed=bed,
        ledger=GeneralWaterLedger(hydro, tled), anchor_ledger=anchor, t_anchor=t_anchor, source=source,
        generation=tree["generation"], revision=tree["revision"], entry_generation=tree["entry_generation"],
        initial_sequence=tree["initial_sequence"], next_request_sequence=tree["next_request_sequence"],
        profile_sha256=gop.sha256, base_prepared_sha256=gop.base.sha256, origin_s=tree["origin_s"],
        anchor_surface_m3=tree["anchor_surface_m3"], anchor_soil_m3=tree["anchor_soil_m3"], last_request_identity=ident)
    if state.digest != iw._sha(tree["digest"], "state.digest"):
        raise GeneralIntereventError("the saved state digest differs from its recomputed digest; refused")
    audit_general_interevent(gop, state)  # the actual whole-water anchored audit (admitted) on the minted product
    return state
