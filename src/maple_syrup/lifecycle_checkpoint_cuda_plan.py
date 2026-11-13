"""PURE metadata plan of the native9f CUDA lifecycle physical SAVE (A6 G2; UNQUALIFIED, written without any device access).

This module never imports CuPy, queries a device, allocates an array, hashes a file or reads any source. `plan_cuda_save` is arithmetic over a sealed
`CudaSaveMetadata` of validated primitives. The metadata is produced by `extract_save_metadata`, a metadata-only walk (reads `.shape/.dtype/.nbytes` only) that the
codec calls AFTER its current-device gate; the device-gated owner accessor (`water_lifecycle._cuda_checkpoint_owner`) is NEVER called here.

Memory semantics (Root correction): the default availability readers report FREE bytes after current allocations. The borrowed current bed / state / context and the
lifecycle's retained host leaves are therefore REPORTED (`resident_*`) and never charged again; the plan's requirements are INCREMENTAL and are compared, together with the
caller's outside reserves, with a fresh free reading. The native MAPLE writer stages every host leaf simultaneously (`snapshot.py` one `to_host` boundary + `np.savez`), so the
whole native host set is charged before the first device-to-host transfer; there is no chunked-NPZ claim. Device scratch coefficients are derived from the read call graph of the
MAPLE validators and are NAMED and flagged unmeasured: they are not a proof, and this is an algorithmic model, not a whole-process memory fit (CUDA context, NVRTC/JIT, CuPy
import, allocator pool/fragmentation, pinned pool, OS page cache and interpreter overhead are excluded).
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from maple_syrup import layered_checkpoint as lc
from maple_syrup import lifecycle_checkpoint as lk

__all__ = [
    "CudaSaveBudget", "CudaSaveMetadata", "CudaSavePlanError", "DEVICE_SCRATCH_COEFFICIENTS", "EXCLUSIONS", "LeafOcc", "MEASUREMENTS_NEEDED", "NATIVE_MEMBERS",
    "NATIVE_READER_CENSUS_KEYS", "extract_save_metadata", "leaf_occurrences", "plan_cuda_save",
]

NATIVE_MEMBERS = 22  # len(lc._native_snapshot_specs): 10 numeric/array + 5 bool + 7 string members (verified at import below)
NATIVE_READER_CENSUS_KEYS = len(lk.NATIVE9F_SNAPSHOT_CENSUS)  # 34: the exact ordinary 9f reader return keys
ROLES = ("boundary", "interevent")
SOURCE_KINDS = ("storm_end", "boundary_origin")
#: The six numeric native members the genuine writer passes through its ONE `to_host` boundary, by leaf path suffix.
NATIVE_WRITER_LEAVES = {"voxel_column.mass_kg": "voxel_mass_kg", "active_layer.mass_kg": "active_mass_kg",
                        "sediment_availability.available_mass_kg": "sediment_availability_available_mass_kg",
                        "sediment_availability.bound_mass_kg": "sediment_availability_bound_mass_kg",
                        "water.depth_m": "water_depth_m", "water.mobile_mass_by_cell_class_kg": "water_mobile_mass_by_cell_class_kg"}
#: Named device-scratch coefficients, derived from READING the MAPLE call graph (`check_voxel_column_state`: voxel_totals, local_scale, capacity-totals and two assumed
#: tolerance temporaries = 5 f8 per (N,V); code/inner-where/diff = 3 i8; empty/full*3/partial/inverted masks = 6 bool) and the class-wide (N,K) validators (assumed 6 f8).
#: UNMEASURED: Root peak measurement is required before any claim.
DEVICE_SCRATCH_COEFFICIENTS = {"voxel_f8_per_NV": 5, "voxel_i8_per_NV": 3, "voxel_bool_per_NV": 6, "class_f8_per_NK": 6, "ledger_leaf_temporaries": 4,
                               "flag_packet_bytes": 64, "status": "derived_from_source_reading_unmeasured"}
EXCLUSIONS = ("CuPy import", "CUDA context creation", "NVRTC/JIT of any kernel or fused reduction first touched by a validator", "device allocator pool growth/fragmentation/retention",
              "pinned-pool retention/staging (assumed one largest leaf, unmeasured)", "OS page cache", "interpreter/NumPy runtime beyond declared object/metadata bytes",
              "whole-process peak (this is an algorithmic model)", "a free reading is not a reservation")
MEASUREMENTS_NEEDED = ("process RSS peak per stage", "CuPy pool used/total peak per stage", "device validator temporaries vs DEVICE_SCRATCH_COEFFICIENTS",
                       "pinned staging of the native writer's to_host", "actual D2H counter deltas of native writer and companion")


class CudaSavePlanError(ValueError):
    """Typed refusal of the pure plan (invalid primitive, inconsistent leaf table, non-int reserve)."""


def _nonneg_int(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or int(value) < 0:
        raise CudaSavePlanError(f"{name} must be a non-Boolean integer >= 0, got {value!r}")
    return int(value)


def _pos_int(value: Any, name: str) -> int:
    v = _nonneg_int(value, name)
    if v < 1:
        raise CudaSavePlanError(f"{name} must be >= 1")
    return v


@dataclass(frozen=True)
class LeafOcc:
    """One array-leaf OCCURRENCE of the full BedState (non-memoizing: an aliased leaf appears once per reaching field)."""

    path: str
    dtype: str
    shape: tuple
    nbytes: int
    group: str  # "native_writer" | "ledger" | "committed" | "other"


def _is_leaf(obj: Any) -> bool:
    return (not isinstance(obj, np.generic) and hasattr(obj, "shape") and hasattr(obj, "dtype") and hasattr(obj, "nbytes") and hasattr(obj, "ndim"))


def leaf_occurrences(tree: Any, prefix: str = "bed") -> list[tuple[str, Any]]:
    """Every array leaf occurrence of a state tree in MAPLE `_map_tree` order (tuples, lists, dicts, `init=True` dataclass fields); NO memoization, so an alias is counted
    per occurrence exactly as `to_device_tree` would transfer/allocate it. Metadata only (no array is read)."""
    out: list[tuple[str, Any]] = []
    if _is_leaf(tree):
        out.append((prefix, tree))
    elif isinstance(tree, (tuple, list)):
        for i, item in enumerate(tree):
            out.extend(leaf_occurrences(item, f"{prefix}[{i}]"))
    elif isinstance(tree, dict):
        for k, item in tree.items():
            out.extend(leaf_occurrences(item, f"{prefix}[{k!r}]"))
    elif dataclasses.is_dataclass(tree) and not isinstance(tree, type):
        for f in dataclasses.fields(tree):
            if f.init:
                out.extend(leaf_occurrences(getattr(tree, f.name), f"{prefix}.{f.name}"))
    return out


@dataclass(frozen=True)
class CudaSaveMetadata:
    """Sealed validated primitives of one planned save. Built only by `extract_save_metadata` (or a test building the same sanitized primitives directly)."""

    role: str
    source_kind: str | None
    n_cells: int
    n_layers: int
    n_classes: int
    n_voxels: int
    n_faces: int
    n_topo_cells: int
    shape: tuple
    leaves: tuple
    context_fraction_bytes: int
    wrapper_retained_host_bytes: int
    owner_audit_device_bytes: int
    source_cold_bytes: int
    n_state_arrays_resident_bytes: int


def extract_save_metadata(*, bed: Any, context_fraction: Any, role: str, source_kind: str | None, n_cells: int, n_layers: int, n_classes: int, shape: tuple,
                          wrapper_retained_host_bytes: int, owner_audit_device_bytes: int, source_cold_bytes: int | None = None) -> CudaSaveMetadata:
    """Metadata-only extraction (call AFTER the current-device gate; reads shape/dtype/nbytes only, never a device). Validates the role/kind pairing, the six native writer leaves
    against `lc._native_snapshot_specs` (so a mismatch between the leaf occurrences and the 22-member ABI table is refused) and the ledger/committed leaves."""
    if role not in ROLES:
        raise CudaSavePlanError(f"role must be one of {ROLES} (an incomplete storm is not a public accepted boundary), got {role!r}")
    if (role == "interevent") != (source_kind is not None) or (source_kind is not None and source_kind not in SOURCE_KINDS):
        raise CudaSavePlanError("an interevent needs a closed source_kind and every other role must give none")
    N, L, K = _pos_int(n_cells, "n_cells"), _pos_int(n_layers, "n_layers"), _pos_int(n_classes, "n_classes")
    if len(shape) != 2 or any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in shape) or shape[0] * shape[1] != N:
        raise CudaSavePlanError("shape must be (ny, nx) with ny*nx == n_cells")
    ny, nx = int(shape[0]), int(shape[1])
    occ = leaf_occurrences(bed)
    paths = [p for p, _ in occ]
    shim = type("Shim", (), {})()
    vshape = None
    for p, a in occ:
        if p == "bed.voxel_column.mass_kg":
            vshape = tuple(int(v) for v in a.shape)
    if vshape is None or len(vshape) != 4 or vshape[:2] != (ny, nx) or vshape[3] != K:
        raise CudaSavePlanError("bed.voxel_column.mass_kg must be (ny, nx, V, K)")
    V = int(vshape[2])
    shim.shape, shim.n_layers, shim.n_cells, shim.n_classes, shim.bed = (ny, nx), L, N, K, bed
    specs = lc._native_snapshot_specs(shim)
    leaves: list[LeafOcc] = []
    seen_native = set()
    for path, a in occ:
        suffix = path[len("bed."):]
        group = "other"
        if suffix in NATIVE_WRITER_LEAVES:
            member = NATIVE_WRITER_LEAVES[suffix]
            shape_spec, dt = specs[member]
            if tuple(int(v) for v in a.shape) != tuple(shape_spec) or np.dtype(a.dtype).str != dt:
                raise CudaSavePlanError(f"{path}: dtype/shape differ from the native member {member} ({np.dtype(a.dtype).str} {tuple(a.shape)} vs {dt} {shape_spec})")
            group = "native_writer"
            seen_native.add(member)
        elif suffix.startswith("ledger"):
            group = "ledger"
        elif suffix.startswith("committed_topography"):
            group = "committed"
        dt = np.dtype(a.dtype)
        if dt.hasobject or dt.kind not in "fiub" or not dt.isnative:
            raise CudaSavePlanError(f"{path}: unsupported dtype {dt}")
        leaves.append(LeafOcc(path, dt.str, tuple(int(v) for v in a.shape), int(a.nbytes), group))
    if seen_native != set(NATIVE_WRITER_LEAVES.values()):
        raise CudaSavePlanError(f"the bed lacks native writer leaves {sorted(set(NATIVE_WRITER_LEAVES.values()) - seen_native)}")
    if not any(l.group == "ledger" for l in leaves) or not any(l.group == "committed" for l in leaves):
        raise CudaSavePlanError("the bed's ledger and committed-topography leaves are required (device-aware occurrence walk found none)")
    F = ny * (nx + 1) + (ny + 1) * nx
    T = (ny + 2) * (nx + 2)
    cf = _nonneg_int(context_fraction.nbytes if hasattr(context_fraction, "nbytes") else context_fraction, "context_fraction_bytes")
    cold = lk.SOURCE_AUDIT_BYTES if source_cold_bytes is None else _nonneg_int(source_cold_bytes, "source_cold_bytes")
    return CudaSaveMetadata(
        role=role, source_kind=source_kind, n_cells=N, n_layers=L, n_classes=K, n_voxels=V, n_faces=F, n_topo_cells=T, shape=(ny, nx), leaves=tuple(leaves),
        context_fraction_bytes=cf, wrapper_retained_host_bytes=_nonneg_int(wrapper_retained_host_bytes, "wrapper_retained_host_bytes"),
        owner_audit_device_bytes=_nonneg_int(owner_audit_device_bytes, "owner_audit_device_bytes"), source_cold_bytes=cold,
        n_state_arrays_resident_bytes=8 * L * N + 16 * N + 8 * F)


@dataclass(frozen=True)
class CudaSaveBudget:
    """Named, separated figures. `host_save_bytes`/`device_save_bytes`/`disk_bytes` are INCREMENTAL requirements (resident bytes reported separately, never charged)."""

    profile: str
    role: str
    source_kind: str | None
    n_cells: int
    n_layers: int
    n_classes: int
    n_voxels: int
    n_faces: int
    n_leaf_occurrences: int
    n_unique_leaf_ids_note: str
    native_members: int
    native_reader_census_keys: int
    companion_members: int
    manifest_cap_bytes: int
    manifest_max_nodes: int
    metadata_bytes: int
    serialized_bytes: int
    disk_bytes: int
    scratch_bytes: int
    host_save_bytes: int
    host_save_capture_first_bytes: int
    device_save_bytes: int
    caller_host_reserve_bytes: int
    caller_device_reserve_bytes: int
    extra_reserve_bytes: int
    host_required_with_reserves: int
    device_required_with_reserves: int
    resident: dict
    host_terms: dict
    device_terms: dict
    stage_requirements: dict
    expected_d2h: dict
    exclusions: tuple
    measurements_needed: tuple

    def as_dict(self) -> dict:
        return {k: (dict(v) if isinstance(v, dict) else v) for k, v in self.__dict__.items()}


def plan_cuda_save(meta: CudaSaveMetadata, *, caller_host_reserve_bytes: int, caller_device_reserve_bytes: int, extra_reserve_bytes: int = 0,
                   profile: str = "native9f_dense_lifecycle_cuda_v1") -> CudaSaveBudget:
    """Pure arithmetic over `meta`; all three reserves are explicit non-Boolean non-negative integers (no default reserve is invented)."""
    if type(meta) is not CudaSaveMetadata:
        raise CudaSavePlanError("meta must be a CudaSaveMetadata")
    hr = _nonneg_int(caller_host_reserve_bytes, "caller_host_reserve_bytes")
    dr = _nonneg_int(caller_device_reserve_bytes, "caller_device_reserve_bytes")
    ex = _nonneg_int(extra_reserve_bytes, "extra_reserve_bytes")
    N, L, K, V, F, T = meta.n_cells, meta.n_layers, meta.n_classes, meta.n_voxels, meta.n_faces, meta.n_topo_cells
    nat = [l for l in meta.leaves if l.group == "native_writer"]
    led = [l for l in meta.leaves if l.group == "ledger"]
    com = [l for l in meta.leaves if l.group == "committed"]
    S_nat_numeric = sum(l.nbytes for l in nat)
    led_bytes, com_bytes = sum(l.nbytes for l in led), sum(l.nbytes for l in com)
    largest_nat = max([l.nbytes for l in nat] + [0])
    largest_led = max([l.nbytes for l in led + com] + [0])
    # --- members / metadata (same declared policy as the CPU codec; the companion table is fixed arrays + ledger/committed leaf OCCURRENCES)
    companion_members = 6 + len(led) + len(com)
    n_entries = companion_members
    manifest_cap, manifest_nodes = lc._manifest_dims(n_entries)
    metadata = lc._metadata_bytes(n_entries, max(companion_members, NATIVE_MEMBERS))
    manifest_cap += lk._ENV_BYTES
    manifest_nodes += lk._ENV_NODES
    metadata += 3 * lk._ENV_BYTES + lk._ENV_NODES * lc.NODE_BYTES + len(lk.NATIVE9F_SNAPSHOT_CENSUS) * lc.NODE_BYTES
    companion_bytes = 8 * L * N + 16 * N + 8 * F + 8 * T + led_bytes + com_bytes
    native_npz = S_nat_numeric + 4 * 8 + 5 * 1  # six arrays + time/seed/step/schema_version scalars + 5 bool members (strings are in `metadata`)
    serialized = (companion_bytes + native_npz + (companion_members + NATIVE_MEMBERS) * (lc.NPY_HEADER_ALLOWANCE + lc.ZIP_MEMBER_ALLOWANCE)
                  + lc.PROVENANCE_ALLOWANCE + manifest_cap)
    io = lc.IO_CHUNK_BYTES + lc.HASH_CHUNK_BYTES
    write_chunks_nat = 2 * min(lc.NPY_WRITE_CHUNK_BYTES, largest_nat)
    write_chunks_com = 2 * min(lc.NPY_WRITE_CHUNK_BYTES, max(companion_bytes, 0), max(8 * L * N, largest_led, 8 * F, 8 * T))
    fp_stage = lc.HASH_CHUNK_BYTES + lk.OBJECT_BYTES
    ctx_stage = 3 * meta.context_fraction_bytes  # pc._canon_into: to_host + np.array copy + tobytes
    # --- host stages (sequential; metadata+io coexist with each; native-first order)
    host_terms = {
        "source_audit": meta.source_cold_bytes, "fingerprint_stage": fp_stage, "context_digest_stage": ctx_stage,
        "native_host_leaves_all_simultaneous": S_nat_numeric, "native_writer_chunks": write_chunks_nat,
        "companion_capture": companion_bytes, "companion_encode_copies": 2 * largest_led, "companion_writer_chunks": write_chunks_com,
        "metadata": metadata, "io": io}
    stages_host = {
        "source_and_validation": meta.source_cold_bytes + metadata + io,
        "fingerprint": fp_stage + ctx_stage + metadata + io,
        "native_write": S_nat_numeric + write_chunks_nat + metadata + io,
        "companion": companion_bytes + 2 * largest_led + write_chunks_com + metadata + io}
    host_peak = max(stages_host.values())
    scratch = metadata + io + max(meta.source_cold_bytes, fp_stage + ctx_stage, write_chunks_nat, 2 * largest_led + write_chunks_com)  # non-payload transients
    capture_first = max(stages_host["source_and_validation"], stages_host["fingerprint"],
                        companion_bytes + 2 * largest_led + S_nat_numeric + write_chunks_nat + metadata + io)
    # --- device stages (incremental only; resident bed/state are reported, never charged)
    c = DEVICE_SCRATCH_COEFFICIENTS
    voxel = V * N * (8 * c["voxel_f8_per_NV"] + 8 * c["voxel_i8_per_NV"] + c["voxel_bool_per_NV"])
    cls = 8 * N * K * c["class_f8_per_NK"]
    ledg = c["ledger_leaf_temporaries"] * largest_led
    native_checks = voxel + cls + ledg + c["flag_packet_bytes"]
    device_terms = {"native_validator_voxel_scratch": voxel, "native_validator_class_scratch": cls, "native_validator_ledger_scratch": ledg,
                    "flag_packets": c["flag_packet_bytes"], "owner_audit": meta.owner_audit_device_bytes,
                    "fingerprint_and_capture_device_scratch": 0}
    device_peak = max(native_checks, meta.owner_audit_device_bytes)
    stage_requirements = {s: {"host_incremental_bytes": v, "device_incremental_bytes": (device_peak if s == "source_and_validation" else 0),
                              "host_with_reserves": v + hr + ex, "device_with_reserves": (device_peak if s == "source_and_validation" else 0) + dr}
                          for s, v in stages_host.items()}
    resident = {"device_bed_leaf_occurrence_bytes": sum(l.nbytes for l in meta.leaves), "device_bed_leaf_occurrences": len(meta.leaves),
                "device_context_fraction_bytes": meta.context_fraction_bytes, "device_state_arrays_bytes": meta.n_state_arrays_resident_bytes,
                "host_lifecycle_retained_bytes": meta.wrapper_retained_host_bytes,
                "note": "REPORTED ONLY: already allocated, therefore already absent from a free-bytes reading; not charged and not required inside the caller reserve"}
    return CudaSaveBudget(
        profile=profile, role=meta.role, source_kind=meta.source_kind, n_cells=N, n_layers=L, n_classes=K, n_voxels=V, n_faces=F, n_leaf_occurrences=len(meta.leaves),
        n_unique_leaf_ids_note="occurrences are counted non-memoizing (an aliased leaf counts per occurrence)", native_members=NATIVE_MEMBERS,
        native_reader_census_keys=NATIVE_READER_CENSUS_KEYS, companion_members=companion_members, manifest_cap_bytes=manifest_cap, manifest_max_nodes=manifest_nodes,
        metadata_bytes=metadata, serialized_bytes=serialized, disk_bytes=serialized, scratch_bytes=scratch, host_save_bytes=host_peak, host_save_capture_first_bytes=capture_first,
        device_save_bytes=device_peak, caller_host_reserve_bytes=hr, caller_device_reserve_bytes=dr, extra_reserve_bytes=ex,
        host_required_with_reserves=host_peak + hr + ex, device_required_with_reserves=device_peak + dr, resident=resident, host_terms=host_terms,
        device_terms=device_terms, stage_requirements=stage_requirements,
        expected_d2h={"native_writer": {"transfers": len(nat), "bytes": S_nat_numeric},
                      "companion": {"transfers": 5 + len(led) + len(com), "bytes": 8 * L * N + 16 * N + 8 * F + led_bytes + com_bytes}},
        exclusions=EXCLUSIONS, measurements_needed=MEASUREMENTS_NEEDED)


assert len(lk.NATIVE9F_SNAPSHOT_CENSUS) == NATIVE_READER_CENSUS_KEYS == 34  # the exact ordinary 9f reader census (verified from source)
