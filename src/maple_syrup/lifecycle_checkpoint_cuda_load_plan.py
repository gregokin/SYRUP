"""PURE metadata plan of the native9f CUDA lifecycle physical LOAD (A6 G4a; UNQUALIFIED, written without any device access).

Never imports CuPy, queries a device, allocates an array, hashes a file or reads a source. `plan_cuda_load` is arithmetic over a `CudaLoadMetadata` of validated primitives. The codec fills the owner figures AFTER its
current-device gate from real pure metadata functions (`wlc.compose_entry_plan`, `wl._cuda_boundary_enter_stage_table`, `lop.memory`); this module never calls the device-gated owner accessor, never builds a parent/duck/capability.

Memory model (Root corrections): the default availability readers report FREE bytes after current allocations, so the borrowed current lifecycle/bed/state/context are REPORTED (`resident_*`), never charged again, never required inside the
caller reserve. The loader's OWN candidate allocations persist across stages: every device stage therefore has `live` (what the loader already holds) and `need` (its own incremental/owner requirement); the whole-call device requirement is
`max(live + need)` over the stage PREFIXES, not the largest isolated temporary, and a per-stage fresh reading (already net of `live`) is compared with `need + reserve`. The candidate bed is the FULL BedState (nonmemoizing leaf OCCURRENCES incl.
ledger, committed, mobile). Host: the native reader returns whole host arrays and the companion decode is simultaneous with them (`S_nat_host + C`), through host validation and the host-to-device transfer. Nested owners keep their own
admissions (their figures contain resident leaves; they read free memory after our candidate exists) — documented as conservative, nothing is invented to bypass them. Device validator scratch is the save plan's named, unmeasured coefficient set.
Exclusions: CuPy import, CUDA context, NVRTC/JIT, allocator pool/fragmentation/retention, pinned pool (assumed one largest leaf, unmeasured), OS page cache, interpreter overhead, whole-process peak.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from maple_syrup import layered_checkpoint as lc
from maple_syrup import lifecycle_checkpoint as lk
from maple_syrup import lifecycle_checkpoint_cuda_plan as sp

__all__ = ["CudaLoadBudget", "CudaLoadMetadata", "CudaLoadPlanError", "LOAD_STAGES", "plan_cuda_load"]

F8 = 8
LOAD_STAGES = ("pre", "stable_copy", "native_read", "companion_decode", "host_validate", "transfer", "device_validate", "arrays_soil", "arrays_history", "arrays_faces",
               "new_lop", "owners", "restore_audit")


class CudaLoadPlanError(ValueError):
    """Typed refusal of the pure load plan."""


@dataclass(frozen=True)
class CudaLoadMetadata:
    """Validated primitives of one planned load. `save` is the pure save-plan metadata of the REFERENCE bed (its leaf occurrences equal the candidate's: the manifest table is compared with them before any payload is read)."""

    save: sp.CudaSaveMetadata
    boundary_capable: bool
    prepare_device_bytes: int       # wlc.compose_entry_plan(.., "prepare").device_bytes (owner's own cold admission incl. its reserves)
    prepare_host_bytes: int         # same plan, host
    newlop_persistent_device_bytes: int  # private device cover + (L,) bound = wlc persistent_device_bytes (8N + 8L)
    cover_copy_bytes: int           # `lop.cover` detached device copy (8N), alive beside the owner's own copy
    owner_stage_table: tuple        # ((name, device bytes), ...) of wl._cuda_boundary_enter_stage_table(lop); empty for a boundary role
    owner_host_bytes: int           # compose_entry_plan(.., "enter_from_boundary").host_bytes (interevent), else 0
    audit_device_bytes: int         # lop.memory.audit_bytes (boundary role audit figure)
    reset_device_bytes: int         # immediate-reset recheck raw figure (0 when not applicable)
    reset_host_bytes: int


def _nn(v: Any, name: str) -> int:
    if isinstance(v, bool) or not isinstance(v, int) or v < 0:
        raise CudaLoadPlanError(f"{name} must be a non-Boolean integer >= 0, got {v!r}")
    return int(v)


@dataclass(frozen=True)
class CudaLoadBudget:
    profile: str
    role: str
    source_kind: str | None
    n_leaf_occurrences: int
    candidate_bed_bytes: int
    candidate_array_bytes: int
    stages: tuple
    stage_requirements: dict
    host_peak_bytes: int
    host_required_with_reserves: int
    device_peak_whole_call_bytes: int
    device_required_with_reserves: int
    serialized_bytes: int
    disk_bytes: int
    scratch_bytes: int
    manifest_cap_bytes: int
    manifest_max_nodes: int
    metadata_bytes: int
    companion_members: int
    native_members: int
    native_reader_census_keys: int
    caller_host_reserve_bytes: int
    caller_device_reserve_bytes: int
    extra_reserve_bytes: int
    expected_h2d: dict
    resident: dict
    host_terms: dict
    device_terms: dict
    exclusions: tuple
    measurements_needed: tuple

    def as_dict(self) -> dict:
        return {k: (dict(v) if isinstance(v, dict) else v) for k, v in self.__dict__.items()}


def plan_cuda_load(meta: CudaLoadMetadata, *, caller_host_reserve_bytes: int, caller_device_reserve_bytes: int, extra_reserve_bytes: int = 0,
                   profile: str = "native9f_dense_lifecycle_cuda_v1") -> CudaLoadBudget:
    if type(meta) is not CudaLoadMetadata or type(meta.save) is not sp.CudaSaveMetadata:
        raise CudaLoadPlanError("meta must be a CudaLoadMetadata over a CudaSaveMetadata")
    hr, dr, ex = (_nn(caller_host_reserve_bytes, "caller_host_reserve_bytes"), _nn(caller_device_reserve_bytes, "caller_device_reserve_bytes"),
                  _nn(extra_reserve_bytes, "extra_reserve_bytes"))
    for f in ("prepare_device_bytes", "prepare_host_bytes", "newlop_persistent_device_bytes", "cover_copy_bytes", "owner_host_bytes", "audit_device_bytes", "reset_device_bytes",
              "reset_host_bytes"):
        _nn(getattr(meta, f), f)
    sm = meta.save
    sb = sp.plan_cuda_save(sm, caller_host_reserve_bytes=0, caller_device_reserve_bytes=0)  # reuse the audited members/metadata/serialized arithmetic (reserves added below)
    N, L, K, V, F, T = sm.n_cells, sm.n_layers, sm.n_classes, sm.n_voxels, sm.n_faces, sm.n_topo_cells
    nat = [l for l in sm.leaves if l.group == "native_writer"]
    led = [l for l in sm.leaves if l.group == "ledger"]
    com = [l for l in sm.leaves if l.group == "committed"]
    S_cand = sum(l.nbytes for l in sm.leaves)            # FULL BedState occurrences (ledger/committed/mobile/alias per occurrence)
    S_nat = sum(l.nbytes for l in nat)
    largest_nat = max([l.nbytes for l in nat] + [0])
    largest_led = max([l.nbytes for l in led + com] + [0])
    C = 8 * L * N + 16 * N + 8 * F + 8 * T + sum(l.nbytes for l in led + com)
    largest_comp = max([8 * L * N, 8 * N, 8 * F, 8 * T, largest_led])
    Cstate = 8 * L * N + 16 * N + 8 * F
    md, io = sb.metadata_bytes, lc.IO_CHUNK_BYTES + lc.HASH_CHUNK_BYTES
    fp = lc.HASH_CHUNK_BYTES + lk.OBJECT_BYTES
    ctx = 3 * sm.context_fraction_bytes
    npy = 2 * lc.NPY_READ_BUFFER_BYTES
    native_objects = sp.NATIVE_MEMBERS * (lc.ZIP_INFO_OBJECT_BYTES + 3 * lc.NATIVE_STRING_MAX) + 3 * lc.NATIVE_PROVENANCE_MAX
    # CPU reader figures (lifecycle_checkpoint._estimate_budget native_load): active-mass copy + (N*K + N) flags/validation + objects
    reader_extra = 8 * N * K + N * K + N + native_objects
    c = sp.DEVICE_SCRATCH_COEFFICIENTS
    scratch_v = V * N * (8 * c["voxel_f8_per_NV"] + 8 * c["voxel_i8_per_NV"] + c["voxel_bool_per_NV"]) + 8 * N * K * c["class_f8_per_NK"] + c["ledger_leaf_temporaries"] * largest_led + c["flag_packet_bytes"]
    # ---- host stage totals (the loader's OWN host allocations; native tree + companion simultaneous)
    host = {
        "pre": meta.save.source_cold_bytes + ctx + fp + md + io,
        "stable_copy": lc.IO_CHUNK_BYTES + lc.HASH_CHUNK_BYTES + md,
        "native_read": S_nat + reader_extra + npy + md,
        "companion_decode": S_nat + C + 2 * largest_comp + npy + md,
        "host_validate": S_nat + C + scratch_v + lc.HASH_CHUNK_BYTES + md,
        "transfer": S_nat + C + largest_nat + lc.HASH_CHUNK_BYTES + md,
        "device_validate": S_nat + C + lc.HASH_CHUNK_BYTES + md,           # native tree still held for the device-vs-host fingerprint compare
        "arrays_soil": C + largest_comp + md, "arrays_history": C + largest_comp + md, "arrays_faces": C + largest_comp + md,
        "new_lop": meta.prepare_host_bytes, "owners": max(meta.owner_host_bytes, meta.reset_host_bytes), "restore_audit": max(meta.reset_host_bytes, md),
    }
    # ---- device stage prefixes: (live, need); `live` = what the loader already holds when the stage starts
    owner_need = max([b for _, b in meta.owner_stage_table] + [0])
    owners_live = S_cand + Cstate + meta.newlop_persistent_device_bytes
    dev = {
        "pre": (0, 0), "stable_copy": (0, 0), "native_read": (0, 0), "companion_decode": (0, 0), "host_validate": (0, 0),
        "transfer": (0, S_cand),
        "device_validate": (S_cand, scratch_v),
        "arrays_soil": (S_cand, 2 * 8 * L * N),                       # staged (L,N) + the constructor copy, staging released after
        "arrays_history": (S_cand + 8 * L * N, 2 * 16 * N),           # two staged (N,) + two constructor copies
        "arrays_faces": (S_cand + 8 * L * N + 16 * N, 8 * F),         # staged faces ARE the final arrays (`ascontiguousarray` aliases)
        "new_lop": (S_cand + Cstate, meta.cover_copy_bytes + meta.prepare_device_bytes),
        "owners": (owners_live, owner_need + meta.cover_copy_bytes),
        "restore_audit": (owners_live, max(meta.audit_device_bytes, meta.reset_device_bytes)),
    }
    if sm.role == "boundary":
        dev["owners"] = (owners_live, 0)
    stages = tuple((s, host[s], dev[s][0], dev[s][1]) for s in LOAD_STAGES)
    host_peak = max(host.values())
    dev_peak = max(l + n for l, n in dev.values())
    req = {s: {"host_incremental_bytes": h, "device_live_bytes": l, "device_need_bytes": n, "device_peak_prefix_bytes": l + n, "host_with_reserves": h + hr + ex,
               "device_need_with_reserve": n + dr} for s, h, l, n in stages}
    scratch = md + io + max(sm.source_cold_bytes, fp + ctx, reader_extra + npy, 2 * largest_comp + npy, scratch_v, largest_nat)
    resident = {"device_reference_bed_leaf_occurrence_bytes": S_cand, "device_context_fraction_bytes": sm.context_fraction_bytes, "device_current_state_arrays_bytes": sm.n_state_arrays_resident_bytes,
                "host_lifecycle_retained_bytes": sm.wrapper_retained_host_bytes,
                "note": "REPORTED ONLY: borrowed and already allocated, hence already absent from a free-bytes reading; not charged again, not required inside the caller reserve"}
    return CudaLoadBudget(
        profile=profile, role=sm.role, source_kind=sm.source_kind, n_leaf_occurrences=len(sm.leaves), candidate_bed_bytes=S_cand, candidate_array_bytes=Cstate, stages=stages,
        stage_requirements=req, host_peak_bytes=host_peak, host_required_with_reserves=host_peak + hr + ex, device_peak_whole_call_bytes=dev_peak, device_required_with_reserves=dev_peak + dr,
        serialized_bytes=sb.serialized_bytes, disk_bytes=sb.serialized_bytes, scratch_bytes=scratch, manifest_cap_bytes=sb.manifest_cap_bytes, manifest_max_nodes=sb.manifest_max_nodes,
        metadata_bytes=md, companion_members=sb.companion_members, native_members=sp.NATIVE_MEMBERS, native_reader_census_keys=sp.NATIVE_READER_CENSUS_KEYS, caller_host_reserve_bytes=hr,
        caller_device_reserve_bytes=dr, extra_reserve_bytes=ex,
        expected_h2d={"bed": {"transfers": len(sm.leaves), "bytes": S_cand}, "arrays": {"transfers": 5, "bytes": 8 * L * N + 16 * N + 8 * F}},
        resident=resident, host_terms={"native_host_leaves": S_nat, "companion": C, "reader_extra": reader_extra, "npy_buffers": npy, "host_validator_scratch": scratch_v, "pinned_staging_assumed": largest_nat,
                                       "context_digest_stage": ctx, "source_audit": sm.source_cold_bytes, "metadata": md, "io": io},
        device_terms={"candidate_bed": S_cand, "candidate_arrays": Cstate, "validator_scratch": scratch_v, "new_lop_prepare": meta.prepare_device_bytes, "cover_copy": meta.cover_copy_bytes,
                      "new_lop_persistent": meta.newlop_persistent_device_bytes, "owner_stage_max": owner_need, "audit": meta.audit_device_bytes, "reset_recheck": meta.reset_device_bytes},
        exclusions=sp.EXCLUSIONS, measurements_needed=sp.MEASUREMENTS_NEEDED + ("candidate-bed H2D pinned staging", "nested owner admissions vs the loader's prefix figures"))
