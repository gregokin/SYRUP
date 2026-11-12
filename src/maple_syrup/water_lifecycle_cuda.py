"""CUDA capability helper for the fixed-property water lifecycle (UNQUALIFIED; written without any device access).

This module holds ONLY (a) the explicit CUDA profile literal and the private capability record, (b) metadata-only host/device
memory arithmetic, and (c) bounded namespace-generic validation helpers (one host packet per reduction). It contains no physical
law, no kernel and no scalar lifecycle logic: those remain the existing components and `water_lifecycle`'s unchanged `_fold`,
`_lifetime_bound` and watermark code. Importing it imports neither CuPy nor touches a device; every device operation happens only
inside functions that receive a genuine CUDA column/array from the caller.

CPU numbers, identities and call shapes of `water_lifecycle` are unchanged: nothing here is used by a CPU lifecycle.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np

from maple_syrup import interevent_transpiration as it
from maple_syrup import layered_storm as ls
from maple_syrup import subsurface as ss

__all__ = [
    "ALL_CUDA_PROFILES",
    "CUDA9F_BOUNDARY_PROFILE",
    "CUDA9F_PROFILE",
    "CUDA9F_PROFILES",
    "CUDA_BOUNDARY_PROFILE",
    "CUDA_PROFILE",
    "CUDA_PROFILES",
    "CudaLifecycleCapability",
    "CudaLifecycleError",
    "CudaLifecycleMemory",
    "cold_cover_ownership",
    "compose_entry_plan",
    "general_canonical_host_bytes",
    "wrapper_retained_host_bytes",
    "estimate_cuda_lifecycle_memory",
    "general_profile_of",
    "inventories_xp",
    "validate_inventories_xp",
]

CUDA_PROFILE = "maple_syrup.water_lifecycle/native39_cuda_dense_or_bedless_fixed_material_UNQUALIFIED"
#: The separate explicit UNQUALIFIED opt-in that ADDS genuine initialized/closed-boundary interevent entry and the controlled J reset to the storm-first
#: capability (never an alias of it: the literal seeds the operator hash).
CUDA_BOUNDARY_PROFILE = "maple_syrup.water_lifecycle/native39_cuda_dense_or_bedless_boundary_entry_reset_UNQUALIFIED"
CUDA_PROFILES = (CUDA_PROFILE, CUDA_BOUNDARY_PROFILE)  # the native39 pair: unchanged value/order (never extended)
#: The explicit UNQUALIFIED native9f pair (distinct literals seed distinct operator hashes; verified through the dense-CUDA source helper, never the Windowed pin set).
CUDA9F_PROFILE = "maple_syrup.water_lifecycle/native9f_cuda_dense_or_bedless_fixed_material_UNQUALIFIED"
CUDA9F_BOUNDARY_PROFILE = "maple_syrup.water_lifecycle/native9f_cuda_dense_or_bedless_boundary_entry_reset_UNQUALIFIED"
CUDA9F_PROFILES = (CUDA9F_PROFILE, CUDA9F_BOUNDARY_PROFILE)
ALL_CUDA_PROFILES = CUDA_PROFILES + CUDA9F_PROFILES
_BOUNDARY_PROFILES = (CUDA_BOUNDARY_PROFILE, CUDA9F_BOUNDARY_PROFILE)


def general_profile_of(profile: Any) -> str:
    """The general (requested-T) owner profile a lifecycle profile requires: the native39 pair -> `it.CUDA_PROFILE`, the native9f pair -> `it.CUDA9F_PROFILE`."""
    if profile in CUDA_PROFILES:
        return it.CUDA_PROFILE
    if profile in CUDA9F_PROFILES:
        return it.CUDA9F_PROFILE
    raise CudaLifecycleError(f"unknown CUDA lifecycle profile {profile!r}")

OBJECT_BYTES = 4096
_F8 = 8
#: Host bytes of the bounded validation packets (flag packet 8 doubles, volume packet 2 doubles) plus one canonical (L,) vector copy.
PACKET_HOST_BYTES = 16 * _F8


class CudaLifecycleError(ValueError):
    """Typed CUDA-lifecycle refusal (the lifecycle re-raises it as its own `LifecycleError`)."""


@dataclass(frozen=True)
class CudaLifecycleMemory:
    """Metadata-only CUDA supplementary figures of one lifecycle (never one count used as both host and device)."""

    n_layers: int
    n_cells: int
    host_reserve_bytes: int
    persistent_device_bytes: int  # private cover 8N + capacity bound 8L
    persistent_host_bytes: int  # canonical (L,) capacity evidence
    cold_device_bytes: int  # staged cover copy beside the caller's cover + capacity + packet
    cold_host_bytes: int  # reserve + object + max(source stage (cold), hash chunk + canonical vector)
    source_stage_bytes: int
    hot_host_bytes: int  # reserve + bounded packets
    cover_inspection_device_bytes: int  # detached device copy of the cover
    nested_general_packet_bytes: int  # the public T wrapper's own device parameter packet (16 L) beyond raw general memory


def estimate_cuda_lifecycle_memory(n_layers: int, n_cells: int, *, host_reserve_bytes: int = 0, source_cold: bool = True,
                                   source_record_bytes: int = 0) -> CudaLifecycleMemory:
    """Pure arithmetic: no allocation, no file read/hash, no device/host query, no cache warming. `source_cold=True` (the conservative
    default when no valid warm association exists) includes the streamed source-tree stage of the actual T source verifier. `source_record_bytes` (default 0 =
    the unchanged native39 figures) is the ADDITIONAL retained native9f association record of this capability: a persistent/hot retained leaf, and beside the
    canonical vector in the cold stage (never added to the stage that precedes it)."""
    L, N = ss._checked_int(n_layers, "n_layers"), ss._checked_int(n_cells, "n_cells")
    reserve = ss._checked_int(host_reserve_bytes, "host_reserve_bytes")
    rec = ss._checked_int(source_record_bytes, "source_record_bytes")
    if L < 1 or N < 1:
        raise CudaLifecycleError("n_layers and n_cells must be >= 1")
    stage = it._source_stage_bytes() if source_cold else 0
    chunk = min(ls.HASH_CHUNK_BYTES, _F8 * N)
    return CudaLifecycleMemory(
        n_layers=L, n_cells=N, host_reserve_bytes=reserve, persistent_device_bytes=_F8 * N + _F8 * L, persistent_host_bytes=_F8 * L + rec,
        cold_device_bytes=_F8 * N + 4 * _F8 * L + OBJECT_BYTES, cold_host_bytes=reserve + OBJECT_BYTES + max(stage, chunk, _F8 * L + 2 * OBJECT_BYTES + rec),
        source_stage_bytes=stage, hot_host_bytes=reserve + PACKET_HOST_BYTES + OBJECT_BYTES + rec, cover_inspection_device_bytes=_F8 * N,
        nested_general_packet_bytes=2 * _F8 * L)


@dataclass(frozen=True, eq=False)
class CudaLifecycleCapability:
    """Private factory-minted record of the CUDA lifecycle capability (never user constructed): the verified native39 source
    association, the explicit Dense class binding, the device-resident private cover and capacity bound, the immutable HOST canonical
    capacity vector, the host quota controls and the memory record. Private-Python mutation is outside the trusted-caller contract."""

    profile: str
    device_id: Any
    source: Any  # it._WindowedSource (native39 pair) or DenseCudaSource (native9f pair): package pin + module/file association (cached evidence, NOT a fresh hash)
    dense_class: type
    voxel_module: str
    cover: Any  # (N,) float64 contiguous device array, private
    saturation_xp: Any  # (L,) float64 device capacity bound, private
    host_saturation: np.ndarray  # (L,) immutable canonical
    host_available_bytes_fn: Any
    host_reserve_bytes: int
    memory: CudaLifecycleMemory
    seal: Any

    @property
    def boundary_capable(self) -> bool:
        return self.profile in _BOUNDARY_PROFILES


def _packet(xp: Any, values: list) -> np.ndarray:
    return ls._packet(xp, values)


def cold_cover_ownership(xp: Any, cover: Any, n_cells: int) -> tuple[Any, str]:
    """One validation packet (finite, min, max) then a private contiguous device copy and ONE streamed hash (bounded host chunk).
    Never converts the cover to a full host array."""
    pk = _packet(xp, [xp.all(xp.isfinite(cover)), xp.min(cover), xp.max(cover)])
    if pk[0] != 1.0 or not (pk[1] >= 0.0 and pk[2] <= 1.0):
        raise CudaLifecycleError("cover must be finite and within [0, 1] on every cell")
    owned = xp.ascontiguousarray(xp.array(cover, dtype=np.float64, copy=True))
    h = hashlib.sha256()
    ls._stream_hash(h, owned)
    return owned, h.hexdigest()


def validate_inventories_xp(lop: Any, a: Any, error: type) -> None:
    """Namespace-generic mirror of the CPU `_validate_inventories` checks, every cell (inactive included): soil finite, >= 0 and
    <= saturation + capacity tolerance; J and lifetime intake finite and >= 0. ONE bounded host packet; the surface/faces go through
    the router's own device validator (called by the caller). Nothing is clipped or reset."""
    xp = lop.xp
    w = a.soil.water_m
    bound = lop._cuda.saturation_xp
    wet, cum = a.history.wetting_m, a.history.cumulative_intake_m
    pk = _packet(xp, [xp.all(xp.isfinite(w)), xp.any(w < 0.0), xp.any(w > bound[:, None]), xp.all(xp.isfinite(wet)), xp.any(wet < 0.0),
                      xp.all(xp.isfinite(cum)), xp.any(cum < 0.0)])
    if pk[0] != 1.0 or pk[1] != 0.0:
        raise error("soil water must be finite and >= 0 on every cell (inactive included)")
    if pk[2] != 0.0:
        raise error("soil water exceeds the layer saturation depth (+ MAPLE capacity tolerance) on a cell")
    if pk[3] != 1.0 or pk[4] != 0.0:
        raise error("history wetting_m must be finite and >= 0 on every cell")
    if pk[5] != 1.0 or pk[6] != 0.0:
        raise error("history cumulative_intake_m must be finite and >= 0 on every cell")


def inventories_xp(lop: Any, a: Any, error: type) -> tuple[float, float]:
    """Active-only surface and soil volumes in ONE two-scalar packet (same expression as the event's `_volumes`)."""
    xp = lop.xp
    area = lop.column.cell_area_m2
    active = lop.column.active
    pk = _packet(xp, [xp.sum(xp.where(active.reshape(lop.shape), a.surface_m, 0.0)) * area,
                      xp.sum(xp.where(active.reshape(1, -1), a.soil.water_m, 0.0)) * area])
    h, soil = float(pk[0]), float(pk[1])
    if not (math.isfinite(h) and math.isfinite(soil)) or h < 0.0 or soil < 0.0:
        raise error("non-finite/negative physical inventory volumes")
    return h, soil


def wrapper_retained_host_bytes(n_layers: int, *, source_record_bytes: int = 0) -> int:
    """Host leaves alive for the whole life of a CUDA lifecycle that every entry and every nested child admission must see: the private canonical
    saturation vector (8 L) plus the capability record and the cached source association record (one bounded object each). Metadata only.
    `source_record_bytes` (default 0) is the ADDITIONAL native9f association record charge, alive for the whole life of the capability."""
    return _F8 * ss._checked_int(n_layers, "n_layers") + 2 * OBJECT_BYTES + ss._checked_int(source_record_bytes, "source_record_bytes")


def general_canonical_host_bytes(n_layers: int) -> int:
    """The general (requested-T) owner's OWN canonical host vectors (residual + saturation = 16 L; `CudaOwnerMemory.persistent_host_bytes`); a different
    record from the lifecycle's 8 L, live beside every interevent-role entry (its hot figure does not contain it)."""
    return 2 * _F8 * ss._checked_int(n_layers, "n_layers")


# Entries whose WHOLE raw figure receives the wrapper 16 L packet. `enter_interevent` is NOT here: its packet is counted per stage inside its own stage table
# (only the transition and audit stages have a genuine prepared general owner), so it is never added a second time.
_GENERAL = frozenset({"evolve_interevent", "request", "close", "audit_interevent"})
_ENTRIES = frozenset({"prepare", "start", "begin_storm", "evolve_storm", "enter_interevent", "evolve_interevent", "request", "close", "audit_boundary",
                      "audit_storm", "audit_interevent", "cover_inspection", "enter_from_boundary", "reset"})


def compose_entry_plan(n_layers: int, n_cells: int, entry: str, *, device_raw_bytes: int, host_reserve_bytes: int = 0, source_cold: bool = True,
                       schedule_bytes: tuple = (), n_faces: int = 0, storm_host_stage_bytes: int = 0, interevent_build_host_bytes: int = 0,
                       source_record_bytes: int = 0) -> dict[str, Any]:
    """PURE composer of the complete separate host/device figures of one public CUDA lifecycle entry (no lop, no array, no read/hash/query/cache warming).
    `device_raw_bytes` is the entry's raw device figure (already containing accepted/static/native/call-parent/request/feedback/post-audit overlaps once);
    this adds the nested T wrapper packet (16 L) for general-owner entries and builds the host figure from real retained leaves + the entry's stage terms.
    `child_host_reserve_bytes` is what each nested cold factory must be told (caller reserve + wrapper-retained leaves, never a whole-call peak).
    `source_record_bytes` (default 0 = the unchanged native39 plan) is the native9f association record: counted once as a retained wrapper leaf AND inside each
    nested general-owner figure (the owner mints its own record), so both are charged wherever both are alive."""
    if entry not in _ENTRIES:
        raise CudaLifecycleError(f"unknown CUDA lifecycle entry {entry!r}")
    L, N = ss._checked_int(n_layers, "n_layers"), ss._checked_int(n_cells, "n_cells")
    reserve = ss._checked_int(host_reserve_bytes, "host_reserve_bytes")
    rec = ss._checked_int(source_record_bytes, "source_record_bytes")
    owner = it.estimate_cuda_owner_memory(L, N, host_reserve_bytes=0, source_cold=source_cold, source_record_bytes=rec)
    general = entry in _GENERAL
    device_terms = {"raw_entry_figure": int(device_raw_bytes), "nested_T_wrapper_packet": 2 * _F8 * L if general else 0}
    if entry == "cover_inspection":
        device_terms["detached_cover_copy"] = _F8 * N + OBJECT_BYTES
    retained = wrapper_retained_host_bytes(L, source_record_bytes=rec)
    host_terms: dict[str, int] = {"caller_host_reserve": reserve, "wrapper_retained_host": retained, "object": OBJECT_BYTES}
    if entry in ("evolve_interevent", "request", "close", "audit_interevent"):
        host_terms["general_owner_canonical_host"] = general_canonical_host_bytes(L)  # live beside every interevent-role entry
    if entry == "prepare":
        # sequential cold stages whose buffers die before the next (source tree walk, streamed cover hash chunk) and the retained leaves, which only
        # come to exist at the end of preparation: the peak is their maximum, so the retained leaves are NOT also added beside the stages
        host_terms["wrapper_retained_host"] = 0
        host_terms["cold_stages_max"] = max(it._source_stage_bytes() if source_cold else 0, min(ls.HASH_CHUNK_BYTES, _F8 * N), retained)
    elif entry == "start":
        host_terms["packets"] = PACKET_HOST_BYTES
    elif entry == "begin_storm":
        edges, intensity, rate = (tuple(schedule_bytes) + (0, 0, 0))[:3]
        host_terms["event_forcing_stage"] = 2 * edges + intensity + rate + min(ls.HASH_CHUNK_BYTES, _F8 * N) + 2 * N * _F8 + 2 * int(n_faces) + N + 4096
        host_terms["storm_operator_host_stage"] = int(storm_host_stage_bytes)
    elif entry == "enter_interevent":
        host_terms["interevent_build_host_stage"] = int(interevent_build_host_bytes)
        host_terms["general_cuda_cold_stage"] = owner.cold_required_bytes  # contains the owner's own cold vectors (no second canonical count here)
    elif entry == "enter_from_boundary":
        # fresh storm operator (native diagnostic stage) + IW build stage + the actual general CUDA cold stage (no supplied primitive) + bounded packets
        # (inventory, quiet evidence); the escape/bed maps are existing topology metadata uploaded to the device (counted in the device stage) but the two outlet-type masks are
        # built on the HOST first, so they are a named host term below
        host_terms["storm_operator_host_stage"] = int(storm_host_stage_bytes)
        host_terms["interevent_build_host_stage"] = int(interevent_build_host_bytes)
        host_terms["general_cuda_cold_stage"] = owner.cold_required_bytes
        host_terms["quiet_mask_host_construction"] = int(n_faces)  # `fx_type == 2` / `fy_type == 2` are built on the HOST (1 B per face) before upload
        host_terms["packets"] = PACKET_HOST_BYTES
    elif entry == "reset":
        host_terms["hash_chunk"] = min(ls.HASH_CHUNK_BYTES, _F8 * N)  # old J, new J and the aliased intake are hashed one after another, streamed
        host_terms["quiet_mask_host_construction"] = int(n_faces)  # both outlet-type masks built on the HOST by the shared quiet evidence (F_x + F_y bytes, simultaneously)
        host_terms["packets"] = PACKET_HOST_BYTES
    elif entry in ("evolve_storm", "audit_boundary", "audit_storm", "cover_inspection"):
        host_terms["packets"] = PACKET_HOST_BYTES
    elif entry in ("evolve_interevent", "close", "audit_interevent"):
        host_terms["packets"] = PACKET_HOST_BYTES + owner.hot_required_bytes
    elif entry == "request":
        host_terms["packets"] = PACKET_HOST_BYTES + owner.request_prepare_required_bytes  # retained caller feedback is DEVICE memory, not host
    return {"entry": entry, "device_bytes": sum(device_terms.values()), "host_bytes": sum(host_terms.values()), "device_terms": device_terms,
            "host_terms": host_terms, "child_host_reserve_bytes": reserve + retained,
            "scope": "metadata-only; separate host/device; excludes JIT/NVRTC/pools/context/import/OS/native material beyond declared reserve; "
                     "cached source evidence is a retained record, not a fresh hash"}
