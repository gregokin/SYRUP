"""Native9f CUDA physical lifecycle checkpoint — LOAD / restore (A6 G4; UNQUALIFIED, written without any GPU/device/CuPy execution).

Loads a directory written by `lifecycle_checkpoint_cuda.save_lifecycle_checkpoint_cuda` into a NEW device bed, a NEW genuine lifecycle operator and a NEW accepted state (closed `boundary` or accepted `interevent`, source kind
`storm_end` / `boundary_origin`), and returns them UNADOPTED. The caller's operator, bed, state and the checkpoint files are never modified; every failure leaves them unchanged (only this call's private stable copies are removed).
Not the MASY calendar restart, no completed-storm role (G5), no Windowed/native39/CPU migration. Nothing here is a restart-ready claim until Root hardware qualification.

Trust/guards: the current device gate runs before any reader/hash/transfer; the ACTUAL imported packages are freshly hashed (literal MAPLE 9f pin, helper-module pins from the streamed inventory, whole-Syrup tree digest compared strictly
with the writer's); the manifest is joined strictly BEFORE any payload byte is opened; payloads are copied through descriptors into a private directory and hash/size/zip-header audited; the genuine 9f native reader is the only reader;
the HOST bed is validated by the unchanged CPU validators and fingerprinted before ONE counted `to_device_tree`; the DEVICE bed must reproduce the same fingerprint. Owner restoration is the trusted G1 seam: no entry/step/close/reset/request
replay, no completed flag fabricated, no clock re-stamped. See docs/subsurface_hydrology/lifecycle_checkpoint_cuda_load.md and the pure plan module for the memory model.
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np

from maple_syrup import checkpoint_capability as cc
from maple_syrup import interevent_transpiration as it
from maple_syrup import interevent_water as iw
from maple_syrup import layered_event as ev
from maple_syrup import layered_checkpoint as lc
from maple_syrup import layered_storm as ls
from maple_syrup import lifecycle_checkpoint as lk
from maple_syrup import lifecycle_checkpoint_cuda as cs
from maple_syrup import lifecycle_checkpoint_cuda_load_plan as lp
from maple_syrup import lifecycle_checkpoint_cuda_policy as pol_mod
from maple_syrup import lifecycle_checkpoint_cuda_v2_plan as v2plan
from maple_syrup import lifecycle_checkpoint_cuda_plan as sp
from maple_syrup import matrix_infiltration as mi
from maple_syrup import pond_checkpoint as pc
from maple_syrup import subsurface as ss
from maple_syrup import water_lifecycle as wl
from maple_syrup import water_lifecycle_cuda as wlc
from maple_syrup.sediment_bed import BedState, is_windowed_bed

__all__ = ["CudaRestoration", "estimate_load_budget_cuda", "load_lifecycle_checkpoint_cuda"]

Err = cs.LifecycleCheckpointCudaError
QuotaErr = cs.LifecycleCheckpointCudaQuotaError
_SCALAR_PARTS = pol_mod.V1.scalar_parts  # the V1 literal policy is the single source of the V1 tables


@dataclass(frozen=True, eq=False)
class CudaRestoration:
    """The NEW lifecycle operator (prepared on the LOADED device bed), the restored accepted state bound to it and a path-free receipt. Nothing is adopted."""

    lop: Any
    state: Any
    receipt: dict


def _refuse(msg: str) -> None:
    raise Err(msg)


def _cv(exc: BaseException) -> Err:
    if isinstance(exc, Err):
        return exc
    if isinstance(exc, (lc.LayeredCheckpointQuotaError, lk.LifecycleCheckpointQuotaError)):
        return QuotaErr(str(exc))
    return Err(f"{type(exc).__name__}: {exc}")


# --- pure fences ------------------------------------------------------------------------------------------------------------------------------
def _fence_lop(lop: Any) -> None:
    if type(lop) is not wl.LifecycleOperator or getattr(lop, "_seal", None) is not wl._SEAL:
        _refuse(f"lop must be a genuine LifecycleOperator, got {type(lop).__name__}")
    if lop._cuda is None or lop.backend != "cuda" or lop.xp is np:
        _refuse("the CUDA lifecycle load needs a genuine CUDA lifecycle operator (a CPU/NumPy/Numba lifecycle uses the unchanged CPU codecs)")
    if lop.context is None:
        _refuse("a bedless lifecycle has no native9f CUDA checkpoint profile (Dense-only)")


def _fence_reference_bed(lop: Any, bed: Any, owner: Any) -> None:
    if type(bed) is not BedState:
        _refuse("reference_bed must be the genuine sediment_bed.BedState")
    try:
        windowed = is_windowed_bed(bed)
    except Exception as exc:
        raise _cv(exc) from None
    from maple.core.types.voxel import VoxelColumnState

    if windowed or type(bed.voxel_column) is not VoxelColumnState:
        _refuse("an actual native Windowed (or foreign) bed has no CUDA lifecycle checkpoint profile")
    for path, leaf in sp.leaf_occurrences(bed):
        if not cs._is_device_leaf(leaf, owner.device_id):
            _refuse(f"{path} of the reference bed is not a CuPy array on the lifecycle's bound device {owner.device_id!r}; refused")


def _hex64(v: Any, name: str) -> str:
    if type(v) is not str or len(v) != 64 or any(c not in "0123456789abcdef" for c in v):
        _refuse(f"{name} must be a lowercase 64-hex SHA-256 string")
    return v


def _gather(lop: Any, owner: Any, role: str, kind: str | None, reference_bed: Any) -> lp.CudaLoadMetadata:
    """Metadata gather AFTER the current-device gate: real pure owner functions only (no parent, no duck, no owner-admission call)."""
    save_meta = sp.extract_save_metadata(
        bed=reference_bed, context_fraction=lop.context.initial_available_fraction, role=role, source_kind=kind, n_cells=owner.n_cells, n_layers=owner.n_layers, n_classes=owner.n_classes,
        shape=owner.shape, wrapper_retained_host_bytes=owner.wrapper_retained_host_bytes, owner_audit_device_bytes=int(lop.memory.audit_scratch_bytes))
    L, N, F = lop.n_layers, lop.n_cells, lop.memory.n_faces
    cold = bool(it._windowed_source_cold())
    rec = wl._cuda_source_record_bytes(owner.profile)
    cm = wlc.estimate_cuda_lifecycle_memory(L, N, host_reserve_bytes=owner.host_reserve_bytes, source_cold=cold, source_record_bytes=rec)
    reserves = int(lop.column.reserve_bytes) + int(lop.extra_reserve_bytes) + int(lop.native_reserve_bytes)
    prep = wlc.compose_entry_plan(L, N, "prepare", device_raw_bytes=cm.cold_device_bytes + reserves, host_reserve_bytes=owner.host_reserve_bytes, source_cold=cold, source_record_bytes=rec)
    table: tuple = ()
    owner_host = 0
    if role == "interevent":
        full = wl._cuda_boundary_enter_stage_table(lop)
        keep = ("storm_operator_build", "interevent_water_prepare", "general_cold_prepare_no_supplied_operator", "audit_with_nested_packet")
        table = tuple((k, int(full[k])) for k in keep)
        iwm = wl._boundary_estimates(lop)[0]
        owner_host = int(wlc.compose_entry_plan(L, N, "enter_from_boundary", device_raw_bytes=max(b for _, b in table), host_reserve_bytes=owner.host_reserve_bytes, source_cold=cold,
                                                n_faces=F, storm_host_stage_bytes=int(lop.memory.storm.get("host_bytes", 0)), interevent_build_host_bytes=int(iwm.build_host_bytes),
                                                source_record_bytes=rec)["host_bytes"])
    reset_dev = reset_host = 0
    if role == "boundary" and owner.boundary_capable:  # conservative: the immediate-reset recheck is admitted up front (the receipt is only known after the manifest)
        reset_dev = int(lop.memory.reset_bytes + 2 * 8 * N + F)
        reset_host = int(wlc.compose_entry_plan(L, N, "reset", device_raw_bytes=reset_dev, host_reserve_bytes=owner.host_reserve_bytes, source_cold=cold, n_faces=F,
                                                source_record_bytes=rec)["host_bytes"])
    return lp.CudaLoadMetadata(save=save_meta, boundary_capable=bool(owner.boundary_capable), prepare_device_bytes=int(prep["device_bytes"]), prepare_host_bytes=int(prep["host_bytes"]),
                               newlop_persistent_device_bytes=int(cm.persistent_device_bytes), cover_copy_bytes=8 * N, owner_stage_table=table, owner_host_bytes=owner_host,
                               audit_device_bytes=int(lop.memory.audit_bytes), reset_device_bytes=reset_dev, reset_host_bytes=reset_host)


def _storm_tree(srec: Any) -> dict:
    """V3: decode the saved storm scalar record WITHOUT requiring completion (a PARTIAL storm is a genuine accepted state); only the exact types the rebuild helper needs are checked here,
    the closed key set/exact types/event identity are enforced by `lk._storm_state_from`, and the completed flag must be an exact bool."""
    try:
        tree = cc.decode_tree(srec)
    except cc.CheckpointCapabilityError as exc:
        raise Err(f"the saved storm record is malformed: {exc}") from None
    if type(tree) is not dict or type(tree.get("completed")) is not bool or type(tree.get("generation")) is not int or type(tree.get("ledger")) is not dict:
        _refuse("the saved storm record is malformed (completed must be an exact bool); refused")
    return tree


def _completed_storm_tree(srec: Any) -> dict:
    """Decode the saved storm scalar record and require EXACTLY `completed is True` (the backend-neutral rebuild helper `lk._storm_state_from` does not itself refuse a flipped flag)."""
    try:
        tree = cc.decode_tree(srec)
    except cc.CheckpointCapabilityError as exc:
        raise Err(f"the saved storm record is malformed: {exc}") from None
    if type(tree) is not dict or tree.get("completed") is not True or type(tree.get("generation")) is not int or type(tree.get("ledger")) is not dict:
        _refuse("the saved storm record is not that of a COMPLETED storm; refused")
    return tree


def _gather_v2(lop: Any, owner: Any, role: str, kind: str | None, reference_bed: Any, spec: Any) -> tuple:
    """V2 metadata: the V1 gather (a storm reuses the role-independent leaf arithmetic through the boundary role) + for the storm role the owner figures from REAL pure metadata functions (`wl.cuda_entry_plan(.., "begin_storm", schedule=..)`, `wl._event_extra_bytes`, the owner's own storm audit arithmetic without a parent)."""
    m = _gather(lop, owner, "boundary" if role == "storm" else role, None if role == "storm" else kind, reference_bed)
    if role != "storm":
        return m, None
    plan = wl.cuda_entry_plan(lop, "begin_storm", schedule=spec.schedule)
    extra = int(wl._event_extra_bytes(lop, spec.schedule))
    mem = lop.memory
    audit = int(mem.storm["resident_bytes"] + extra + mem.reserves_bytes + mem.audit_scratch_bytes + mem.owned_fixed_bytes)
    return (dataclasses.replace(m, save=dataclasses.replace(m.save, role="storm")),
            v2plan.StormLoadTerms(begin_storm_device_bytes=int(plan["device_bytes"]), begin_storm_host_bytes=int(plan["host_bytes"]), event_extra_bytes=extra, storm_audit_device_bytes=audit))


def estimate_load_budget_cuda(meta: lp.CudaLoadMetadata, *, caller_host_reserve_bytes: int, caller_device_reserve_bytes: int, extra_reserve_bytes: int = 0) -> lp.CudaLoadBudget:
    """The PURE load budget over sanitized metadata (never calls the device-gated owner accessor)."""
    return lp.plan_cuda_load(meta, caller_host_reserve_bytes=caller_host_reserve_bytes, caller_device_reserve_bytes=caller_device_reserve_bytes, extra_reserve_bytes=extra_reserve_bytes,
                             profile=cs.PROFILE)


def _admit_limits(b: lp.CudaLoadBudget, limits: Any) -> None:
    for what, need, lim in (("serialized checkpoint", b.serialized_bytes, limits.max_serialized_bytes), ("load host peak (incremental + reserves)", b.host_required_with_reserves, limits.max_host_bytes),
                            ("load device peak (stage prefixes + reserve)", b.device_required_with_reserves, limits.max_device_bytes), ("load transient scratch", b.scratch_bytes, limits.max_scratch_bytes)):
        if need > lim:
            raise QuotaErr(f"{what} needs {need} B but the caller's limit is {lim} B; nothing was read/allocated")


def _admit_stage(b: lp.CudaLoadBudget, stage: str, host_fn: Callable[[], int], device_fn: Callable[[], int] | None) -> None:
    """Fresh FREE reading before a stage: the host reading vs the stage's host figure + reserves; the device reading (already net of what the loader holds) vs the stage's own `need` + reserve."""
    r = b.stage_requirements[stage]
    avail = cs._read(host_fn, f"host stage {stage}")
    if r["host_with_reserves"] > avail:
        raise QuotaErr(f"host stage {stage} needs {r['host_with_reserves']} B but only {avail} B are reported free; nothing further was done")
    if device_fn is not None and (r["device_need_bytes"] or stage == "transfer"):
        avail_d = cs._read(device_fn, f"device stage {stage}")
        if r["device_need_with_reserve"] > avail_d:
            raise QuotaErr(f"device stage {stage} needs {r['device_need_with_reserve']} B but only {avail_d} B are reported free; nothing further was done")


# --- strict manifest / payload checks -----------------------------------------------------------------------------------------------------------------
def _check_table(table: Any, shim: Any, meta: lp.CudaLoadMetadata) -> None:
    if not isinstance(table, dict):
        _refuse("array table must be an object")
    fixed = lc._companion_specs(shim)
    for n, (shape, dtype) in fixed.items():
        rec = table.get(n)
        if not isinstance(rec, dict) or set(rec) != {"dtype", "shape"} or rec["dtype"] != dtype or tuple(rec.get("shape", ())) != shape:
            _refuse(f"array {n}: declared dtype/shape differ from the profile")
    names = [n for n in table if n not in fixed]
    for n in names:
        rec = table[n]
        if (not isinstance(n, str) or not n.startswith("native_") or not isinstance(rec, dict) or set(rec) != {"dtype", "shape"} or not pc._dtype_ok(rec["dtype"])
                or not isinstance(rec["shape"], list) or len(rec["shape"]) > 5 or any(type(v) is not int or v < 0 for v in rec["shape"])):
            _refuse(f"array {n!r}: not an allowed declared native leaf")
    ref = sorted((l.dtype, tuple(l.shape)) for l in meta.save.leaves if l.group in ("ledger", "committed"))
    got = sorted((table[n]["dtype"], tuple(table[n]["shape"])) for n in names)
    if got != ref:
        _refuse("declared native ledger/committed leaves differ from the reference bed's (class, shapes, present processes)")


def _check_snapshot(snapshot: dict, data: dict, t_s: float, revision: int, _policy: Any = None) -> None:
    pol = pol_mod.policy_of(_policy)
    from maple.io.outputs.snapshot import SNAPSHOT_SCHEMA_VERSION
    from maple.io.outputs.visualization_schema import SCHEMA_VERSION as HISTORY_SCHEMA

    got = set(snapshot)
    if got != lk.NATIVE9F_SNAPSHOT_CENSUS:
        _refuse(f"the native9f snapshot reader returned an unexpected key set (missing {sorted(lk.NATIVE9F_SNAPSHOT_CENSUS - got)}, unexpected {sorted(got - lk.NATIVE9F_SNAPSHOT_CENSUS)}); refused")
    for key in ("fixed_base", "regional_slope_wind_source"):
        if snapshot[key] is not None:
            _refuse(f"the native9f snapshot carries unsupported native state {key!r}; refused")
    if not snapshot["validated"] or not snapshot["restartable"] or snapshot["is_diagnostic"]:
        _refuse("MAPLE snapshot is not validated/restartable or is diagnostic (flags are checks, never proof)")
    if snapshot["snapshot_schema_version"] != SNAPSHOT_SCHEMA_VERSION or snapshot["history_schema_version"] != HISTORY_SCHEMA:
        _refuse("MAPLE snapshot schema/history semantic identity is not the exact supported one")
    try:
        prov = json.loads(snapshot["provenance_json"])
    except ValueError:
        prov = None
    if (not isinstance(prov, dict) or prov.get("syrup_lifecycle_checkpoint_schema") != pol.schema or prov.get("profile") != pol.profile or prov.get("source_kind") != data["source_kind"]
            or prov.get("role") != data["role"] or prov.get("binding_sha256") != data["binding_sha256"] or prov.get("identity") != data["identity"]
            or prov.get("array_backend") != data["array_backend"] or prov.get("hydraulic_backend") != data["hydraulic_backend"] or prov.get("requires_companion") is not True
            or prov.get("runtime") != data["runtime"]):
        _refuse("MAPLE snapshot provenance is not bound to this manifest")
    if (snapshot["water"] is None or snapshot["sediment_availability"] is None or snapshot["history_store_path"] or snapshot["reason"] != lc._SNAPSHOT_REASON
            or any(snapshot.get(k) is not None for k in lc._FOREIGN_SNAPSHOT_FIELDS) or snapshot.get("sediment_availability_fallback_sha256") or snapshot.get("scheduler_coupled_status")
            or snapshot.get("scheduler_wind_record_source_sha256") or snapshot["run_name"] or snapshot["error_type"] or snapshot["error_message"] or snapshot["seed"] != 0):
        _refuse("MAPLE snapshot lacks water/availability or carries foreign optional state")
    if snapshot["time_s"] != t_s or snapshot["step"] != revision:
        _refuse("MAPLE snapshot clock/step differ from the lifecycle record")


_ALLOWED_NAMES = pol_mod.V1.allowed_names


def _check_directory_names(dirfd: int, _policy: Any = None) -> None:
    """Bounded descriptor-relative directory preflight: iterate at most `len(allowed) + 1` entries of `os.scandir(dirfd)`, stop at the FIRST name outside the closed allowed set (a 4th distinct entry is necessarily foreign),
    keep constant storage and never list, sort or format the directory's names. Error categories are the old ones: no manifest -> unsealed/partial; foreign entry or missing payload -> must contain exactly the allowed set."""
    pol = pol_mod.policy_of(_policy)
    allowed = pol.allowed_names  # closed, bounded, taken from the VALIDATED private policy (never listed from the directory)
    seen: set = set()
    foreign = False
    with os.scandir(dirfd) as it:
        for n, entry in enumerate(it):
            if n >= len(allowed) or entry.name not in allowed:
                foreign = True
                break
            seen.add(entry.name)
    if foreign:
        try:
            os.stat(pol.manifest_name, dir_fd=dirfd, follow_symlinks=False)  # constant-time known-name probe: preserves the unsealed category
        except FileNotFoundError:
            _refuse("unsealed or partial checkpoint directory (no sealed manifest)")
        except OSError:
            pass
    elif pol.manifest_name not in seen:
        _refuse("unsealed or partial checkpoint directory (no sealed manifest)")
    if foreign or seen != allowed:
        _refuse(f"checkpoint directory must contain exactly {sorted(allowed)} (an entry outside this set, or a missing payload, was found)")


def _read_manifest(dirfd: int, cap_bytes: int, max_nodes: int, _policy: Any = None) -> dict:
    """The CUDA manifest (`lifecycle_checkpoint_cuda.json`) under its derived policy: bounded descriptor read, structural scan BEFORE `json.loads`, strict parse (no constants, no duplicate keys, ASCII)."""
    fd = lc._open_regular(pol_mod.policy_of(_policy).manifest_name, dirfd, cap_bytes)
    try:
        raw = b""
        while len(raw) <= cap_bytes:
            chunk = os.read(fd, lc.IO_CHUNK_BYTES)
            if not chunk:
                break
            raw += chunk
    finally:
        os.close(fd)
    if len(raw) > cap_bytes:
        _refuse("manifest exceeds its derived byte bound")
    lc._scan_json(raw, lc.MANIFEST_MAX_DEPTH, max_nodes, lc.MANIFEST_MAX_STRING)

    def no_constants(token: str) -> Any:
        raise ValueError(f"non-finite JSON constant {token}")

    def no_duplicates(pairs: list) -> dict:
        out: dict = {}
        for k, v in pairs:
            if k in out:
                raise ValueError(f"duplicate manifest key {k!r}")
            out[k] = v
        return out

    try:
        data = json.loads(raw.decode("ascii"), parse_constant=no_constants, object_pairs_hook=no_duplicates)
    except (ValueError, UnicodeDecodeError, RecursionError) as exc:
        raise Err(f"manifest is not strict JSON: {exc}") from None
    if not isinstance(data, dict):
        _refuse("manifest must be a JSON object")
    return data


def _transfer_record(delta: Any) -> dict:
    """Receipt record of a `TransferCounters` delta with EVERY named field (the G23 save helper is D2H-only and has no host_to_device_bytes; it is not reused here)."""
    return {name: int(getattr(delta, name)) for name in ("host_to_device", "host_to_device_bytes", "device_to_host", "device_to_host_bytes", "scalar_reads", "synchronizations")}


def _leaf_table(tree: Any) -> list:
    return [(p, np.dtype(a.dtype).str, tuple(int(v) for v in a.shape)) for p, a in sp.leaf_occurrences(tree)]


# --- load -----------------------------------------------------------------------------------------------------------------------------------------------
def load_lifecycle_checkpoint_cuda(path: Any, lop: Any, expected_identity: dict, *, expected_role: str, expected_source_kind: str | None = None, limits: Any, reference_bed: Any,
                                   caller_host_reserve_bytes: int, caller_device_reserve_bytes: int, host_available_bytes_fn: Any = None, device_available_bytes_fn: Any = None,
                                   disk_available_bytes_fn: Any = None, extra_reserve_bytes: int = 0, expected_physical_digest: Any = None) -> CudaRestoration:
    """Load a sealed native9f CUDA lifecycle checkpoint into a NEW bed/operator/state (UNADOPTED). `reference_bed` is the caller's genuine device `BedState` carrying the SAME fixed native material
    (borrowed, read-only; e.g. the initial bed of a freshly built lifecycle in a new process). See the module docstring for the order and guards."""
    return _load_engine(path, lop, expected_identity, expected_role=expected_role, expected_source_kind=expected_source_kind, limits=limits, reference_bed=reference_bed,
                        caller_host_reserve_bytes=caller_host_reserve_bytes, caller_device_reserve_bytes=caller_device_reserve_bytes, host_available_bytes_fn=host_available_bytes_fn,
                        device_available_bytes_fn=device_available_bytes_fn, disk_available_bytes_fn=disk_available_bytes_fn, extra_reserve_bytes=extra_reserve_bytes,
                        expected_physical_digest=expected_physical_digest, _policy=pol_mod.V1)


def _select_policy(policy: Any) -> Any:
    """Registered literal policy by identity (None -> V1); unknown/forged or unimplemented policies are refused BEFORE any side effect."""
    try:
        pol = pol_mod.policy_of(policy)
    except pol_mod.CodecPolicyError as exc:
        raise Err(str(exc)) from None
    if not pol.implemented:
        _refuse(f"the {pol.name} codec policy is declared but its loader is not implemented (no public V2 load exists yet)")
    return pol


def _load_engine(path: Any, lop: Any, expected_identity: dict, *, expected_role: str, expected_source_kind: str | None = None, limits: Any, reference_bed: Any,
                 caller_host_reserve_bytes: int, caller_device_reserve_bytes: int, host_available_bytes_fn: Any = None, device_available_bytes_fn: Any = None,
                 disk_available_bytes_fn: Any = None, extra_reserve_bytes: int = 0, expected_physical_digest: Any = None, _storm: Any = None, _policy: Any = None,
                 _expected_plan: Any = None) -> CudaRestoration:
    """The load body under a registered private policy (V1 for every public entry point); the policy is validated first, before any side effect."""
    pol = _select_policy(_policy)
    hr = cs._reserve(caller_host_reserve_bytes, "caller_host_reserve_bytes")
    dr = cs._reserve(caller_device_reserve_bytes, "caller_device_reserve_bytes")
    ex = cs._reserve(extra_reserve_bytes, "extra_reserve_bytes")
    host_fn = cs._reader(host_available_bytes_fn, "host_available_bytes_fn")
    dev_fn_user = cs._reader(device_available_bytes_fn, "device_available_bytes_fn")
    disk_fn_user = cs._reader(disk_available_bytes_fn, "disk_available_bytes_fn")
    if not isinstance(limits, lc.CheckpointLimits):
        _refuse("limits must be a CheckpointLimits record")
    try:
        limits.validated()
    except lc.LayeredCheckpointError as exc:
        raise _cv(exc) from None
    # exact closed argument TYPES first: no hashing, membership, iteration, comparison or formatting of a caller object (dict/list/ndarray/bool/str subclass with custom magic) before the typed refusal
    if type(expected_role) is not str or not (expected_source_kind is None or type(expected_source_kind) is str):
        _refuse("expected_role must be an exact str and expected_source_kind an exact str or None")
    role, kind = expected_role, expected_source_kind
    if (role, kind) not in pol.scalar_parts:
        _refuse(f"(expected_role, expected_source_kind) {(role, kind)!r} is not a covered closed pair {sorted(pol.scalar_parts, key=str)} (a storm role is not loadable here)")
    if (role == "storm") != (_storm is not None):  # exact: a StormSpec is required for (and only for) the storm role; no caller object is inspected here
        _refuse("a StormSpec is required for (and only for) the storm role")
    if _storm is not None and type(_storm) is not lk.StormSpec:
        _refuse("storm must be an exact lifecycle_checkpoint.StormSpec")
    if _expected_plan is not None and not pol.daily:
        _refuse("expected_plan is accepted only by the V3 daily codec")
    try:
        lk._validate_expected_plan(_expected_plan)  # EVERY supplied plan is re-derived canonically before any fence, budget, source audit or I/O
    except lk.LifecycleCheckpointError as exc:
        raise Err(str(exc)) from None
    if expected_physical_digest is not None:
        _hex64(expected_physical_digest, "expected_physical_digest")
    _fence_lop(lop)
    try:
        expected = lc._canonical_identity(expected_identity)
    except lc.LayeredCheckpointError as exc:
        raise _cv(exc) from None
    if not isinstance(path, (str, os.PathLike)):
        _refuse("path must be a str or PathLike")
    root = Path(path)
    if root.is_symlink() or not root.is_dir():
        _refuse(f"checkpoint path is not a plain directory: {root}")
    # ---- current-device gate: before any reader, hash, copy, transfer
    try:
        owner = wl._cuda_checkpoint_owner(lop)
    except (wl.LifecycleError, ss.SubsurfaceError) as exc:
        raise _cv(exc) from None
    if isinstance(owner.device_id, bool) or not isinstance(owner.device_id, int):
        _refuse("the lifecycle's bound device id is not an integer")
    if kind == "boundary_origin" and not owner.boundary_capable:
        _refuse("a boundary-origin source exists only under the explicit CUDA9F_BOUNDARY_PROFILE lifecycle; refused")
    _fence_reference_bed(lop, reference_bed, owner)
    try:
        if pol is pol_mod.V1:
            meta = _gather(lop, owner, role, kind, reference_bed)
            budget = lp.plan_cuda_load(meta, caller_host_reserve_bytes=hr, caller_device_reserve_bytes=dr, extra_reserve_bytes=ex, profile=pol.profile)
        else:  # V2: the V1 load plan wrapped with the V2 envelope (+ storm terms); V1 plan modules untouched
            from maple_syrup import lifecycle_checkpoint_cuda_v2 as v2

            if role == "storm":
                v2.validate_storm_spec(_storm, lop, owner.device_id)
            meta, storm_terms = _gather_v2(lop, owner, role, kind, reference_bed, _storm)
            if pol.daily:
                from maple_syrup import lifecycle_checkpoint_cuda_v3 as v3

                budget = v3.estimate_load_budget_cuda_v3(meta, caller_host_reserve_bytes=hr, caller_device_reserve_bytes=dr, extra_reserve_bytes=ex, storm=storm_terms)
            else:
                budget = v2.estimate_load_budget_cuda_v2(meta, caller_host_reserve_bytes=hr, caller_device_reserve_bytes=dr, extra_reserve_bytes=ex, storm=storm_terms)
    except (wl.LifecycleError, sp.CudaSavePlanError, lp.CudaLoadPlanError, wlc.CudaLifecycleError, it.GeneralIntereventError, iw.InterEventWaterError, ls.LayeredStormError) as exc:
        raise _cv(exc) from None
    _admit_limits(budget, limits)
    host_reader = ss.default_host_available_bytes if host_fn is None else host_fn
    dev_reader = lop.column.available_bytes_fn if dev_fn_user is None else dev_fn_user
    disk_avail = cs._read(lc._disk_reader(Path(tempfile.gettempdir()), disk_fn_user), "disk")
    if budget.disk_bytes > disk_avail:
        raise QuotaErr(f"load disk (stable copies) needs {budget.disk_bytes} B but only {disk_avail} B are reported free")
    # whole-call: host peak and the device PREFIX peak, fresh readings, BEFORE the source hash / any file body / any device copy
    avail_h = cs._read(host_reader, "host whole call")
    if budget.host_required_with_reserves > avail_h:
        raise QuotaErr(f"load needs {budget.host_required_with_reserves} B host but only {avail_h} B are reported free; nothing was read")
    avail_d = cs._read(dev_reader, "device whole call")
    if budget.device_required_with_reserves > avail_d:
        raise QuotaErr(f"load needs {budget.device_required_with_reserves} B device (stage prefixes) but only {avail_d} B are reported free; nothing was read")
    _admit_stage(budget, "pre", host_reader, None)
    source = cs._fresh_source_pair(pol)
    runtime = cs._runtime_record(lop, owner.device_id)
    material_ref = cs._material_sha(reference_bed, lop.context)  # cold source/material guard of the caller's reference bed (streamed; D2H counted)
    transfers: dict = {}
    storm_binding = None
    if pol is not pol_mod.V1 and role == "storm":  # pre-payload spec identity from the BORROWED caller spec (admitted before the field stream)
        _admit_stage(budget, "spec_digest", host_reader, None)
        c_spec = cs._counters()
        storm_binding = v2.expected_storm_binding(_storm)
        d_spec = cs._counters().delta(c_spec)
        transfers["spec_digest"] = _transfer_record(d_spec)
        v2.check_spec_digest_traffic(d_spec, budget.stage_requirements["spec_digest"]["expected_d2h"], _storm.field.multiplier)
    shim = SimpleNamespace(shape=lop.shape, n_layers=lop.n_layers, n_cells=lop.n_cells, n_classes=lop.n_classes, bed=reference_bed)
    flags_dir = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        dirfd = os.open(root, flags_dir)
    except OSError as exc:
        raise Err(f"cannot open the checkpoint directory: {exc}") from None
    private = None
    caps: list = []
    try:
        private = Path(tempfile.mkdtemp(prefix="lifecycle-checkpoint-cuda-stable-"))
        _check_directory_names(dirfd, pol)
        data = _read_manifest(dirfd, budget.manifest_cap_bytes, budget.manifest_max_nodes, pol)
        return _load_body(data, dirfd, private, lop, owner, reference_bed, material_ref, expected, role, kind, limits, budget, source, runtime, shim, meta, host_reader, dev_reader,
                          expected_physical_digest, caps, transfers, root, pol, _storm, storm_binding, _expected_plan)
    except Err:
        raise
    except Exception as exc:
        raise _cv(exc) from exc
    finally:
        for c in caps:
            cc.close(c)
        os.close(dirfd)
        if private is not None:
            shutil.rmtree(private, ignore_errors=True)


def _host_stages(data: dict, dirfd: int, private: Path, lop: Any, owner: Any, reference_bed: Any, material_ref: str, expected: dict, role: str, kind: str | None, limits: Any,
                 budget: lp.CudaLoadBudget, source: dict, runtime: dict, shim: Any, meta: lp.CudaLoadMetadata, host_reader: Callable, _policy: Any = None, _storm_binding: Any = None) -> tuple:
    """Everything up to (and including) the HOST validation and fingerprints: no device object is created or touched here (CPU-testable with real archives)."""
    from maple.io.outputs import snapshot as native_snapshot

    pol = pol_mod.policy_of(_policy)
    # ---- manifest + every binding BEFORE any payload is opened
    if set(data) != pol.manifest_keys:
        _refuse(f"manifest keys mismatch: missing {sorted(pol.manifest_keys - set(data))}, unknown {sorted(set(data) - pol.manifest_keys)}")
    if data["schema"] != pol.schema:
        _refuse(f"checkpoint schema {data['schema']!r} is not {pol.schema!r}")
    if lc._sha(data["manifest_sha256"], "manifest_sha256") != pc._seal_manifest(data)["manifest_sha256"]:
        _refuse("manifest hash mismatch (the manifest was modified)")
    if data["profile"] != pol.profile:
        _refuse(f"profile {data['profile']!r} is not {pol.profile!r}; profiles never cross-load")
    if data["role"] != role or data["source_kind"] != kind or (role, data["source_kind"]) not in pol.scalar_parts:
        _refuse(f"saved role/source kind {(data['role'], data['source_kind'])!r} differ from the expected {(role, kind)!r}")
    if data["identity"] != expected:
        _refuse("checkpoint identity differs from the expected identity")
    if data["source"] != source:
        _refuse("source pair mismatch: the imported maple_syrup / maple packages differ from the writer's (refused before any payload)")
    if data["array_backend"] != "cupy" or data["hydraulic_backend"] != "cuda":
        _refuse("backend mismatch: a CUDA checkpoint is loaded only on CuPy/CUDA (no migration)")
    if data["runtime"] != runtime:
        _refuse("runtime mismatch: the freshly inspected CuPy/CUDA/compute-capability/Python runtime differs from the writer's (no migration)")
    binding = cs._binding(lop, owner, role, kind, material_ref, runtime, meta.save.n_voxels, pol)
    if pol is not pol_mod.V1:  # V2: the closed `storm` binding rebuilt cold from the caller's borrowed spec (None for boundary/interevent)
        binding["storm"] = _storm_binding
    if data["binding"] != binding or data["binding_sha256"] != lc._sha_text(lc._json_dumps(binding)):
        diff = sorted(k for k in binding if not isinstance(data["binding"], dict) or data["binding"].get(k) != binding[k])
        _refuse(f"binding mismatch (source/runtime/context/properties/material/operator differ from the expected): {diff}")
    payload = data["payload"]
    if not isinstance(payload, dict) or set(payload) != lc.PAYLOAD_KEYS:
        _refuse("payload keys mismatch")
    scalars = payload["scalars"]
    if not isinstance(scalars, dict) or set(scalars) != pol.scalar_parts[(role, kind)]:
        _refuse("scalar record parts differ from the role's closed set")
    files, sizes = data["files"], data["sizes"]
    if not isinstance(files, dict) or set(files) != set(pol.files) or not isinstance(sizes, dict) or set(sizes) != set(pol.files):
        _refuse("payload file set mismatch")
    declared = sum(lc._int(sizes[n], f"sizes[{n}]") for n in pol.files)
    if declared > limits.max_serialized_bytes or declared > budget.serialized_bytes:
        raise QuotaErr(f"declared payload {declared} B exceeds the serialized limit/derived bound")
    table = payload["arrays"]
    _check_table(table, shim, meta)
    if not isinstance(payload["native"], dict) or set(payload["native"]) != {"ledger", "committed_topography"}:
        _refuse("native companion keys mismatch")
    try:
        life = cc.decode_tree(scalars["lifecycle"])
        t_s, revision = life["t_s"], life["revision"]
    except (cc.CheckpointCapabilityError, KeyError, TypeError) as exc:
        raise Err(f"lifecycle scalar record refused: {exc}") from None
    if type(t_s) is not float or t_s < 0.0 or type(revision) is not int or revision < 0:
        _refuse("lifecycle clock/revision must be an exact non-negative float / integer")
    # ---- stable private copies hashed through descriptors; zip header audits against the exact tables
    _admit_stage(budget, "stable_copy", host_reader, None)
    stable = {n: lc._stable_copy(n, dirfd, private, lc._sha(files[n], f"files[{n}]"), int(sizes[n]), limits.max_serialized_bytes) for n in pol.files}
    lc._audit_path(stable[cs.PAYLOAD_NATIVE], lc._native_snapshot_specs(shim), label=cs.PAYLOAD_NATIVE, byte_limit=limits.max_serialized_bytes)
    lc._audit_path(stable[cs.PAYLOAD_COMPANION], lc._companion_expected(shim, table), label=cs.PAYLOAD_COMPANION, byte_limit=limits.max_serialized_bytes)
    # ---- host decode (genuine 9f reader; companion decoder under the registered 9f policy)
    _admit_stage(budget, "native_read", host_reader, None)
    try:
        snapshot = native_snapshot.load_state_snapshot(stable[cs.PAYLOAD_NATIVE])
    except Exception as exc:
        raise Err(f"MAPLE snapshot cannot be loaded: {type(exc).__name__}: {exc}") from None
    _check_snapshot(snapshot, data, t_s, revision, pol)
    _admit_stage(budget, "companion_decode", host_reader, None)
    arrays, ledger_node, committed_node = lk._decode_companion(stable[cs.PAYLOAD_COMPANION], table, payload["native"], lk.NATIVE9F_PROFILE_V2, lk._V2_9F)
    lc._check_topology(arrays["topo_source_full"], lop.topology, data["binding"]["topology"])
    host_bed = BedState(voxel_column=snapshot["voxel_column"], active_layer=snapshot["active_layer"], water=snapshot["water"], ledger=ledger_node, committed_topography=committed_node,
                        sediment_availability=snapshot["sediment_availability"])
    del snapshot
    # ---- HOST validation before anything exists on the device
    _admit_stage(budget, "host_validate", host_reader, None)
    ref_ledger_type = type(reference_bed.ledger)
    lc._native_checks(host_bed, lop.context, t_s, ref_ledger_type)
    if getattr(ledger_node, "present_processes", None) != getattr(reference_bed.ledger, "present_processes", None):
        _refuse("native ledger present_processes differ from the reference (process selection is restart-relevant)")
    host_material = cs._material_sha(host_bed, lop.context)
    if host_material != material_ref or host_material != data["binding"]["native_material_sha256"]:
        _refuse("the loaded native material differs from the reference bed's (content SHA over voxels/active/availability/ledger/committed/mobile)")
    if _leaf_table(host_bed) != [(l.path, l.dtype, tuple(l.shape)) for l in meta.save.leaves]:
        _refuse("the loaded host bed's leaf occurrences differ from the planned candidate inventory (class/shape/dtype/occurrence order)")
    return host_bed, arrays, life, scalars, t_s, revision, host_material, ref_ledger_type


def _load_body(data: dict, dirfd: int, private: Path, lop: Any, owner: Any, reference_bed: Any, material_ref: str, expected: dict, role: str, kind: str | None, limits: Any,
               budget: lp.CudaLoadBudget, source: dict, runtime: dict, shim: Any, meta: lp.CudaLoadMetadata, host_reader: Callable, dev_reader: Callable,
               expected_physical_digest: Any, caps: list, transfers: dict, root: Path, _policy: Any = None, _storm_spec: Any = None, _storm_binding: Any = None,
               _expected_plan: Any = None) -> CudaRestoration:
    from maple.core.backend import to_device, to_device_tree

    pol = pol_mod.policy_of(_policy)

    host_bed, arrays, life, scalars, t_s, revision, host_material, ref_ledger_type = _host_stages(
        data, dirfd, private, lop, owner, reference_bed, material_ref, expected, role, kind, limits, budget, source, runtime, shim, meta, host_reader, pol, _storm_binding)
    daily = None
    if pol.daily:  # V3: the saved cursor decoded STRICTLY + role<->pending + the supplied StormSpec/expected plan bound to the saved pending occurrence, on the HOST scalars, BEFORE any device allocation
        try:
            daily = lk._daily_cursor_checked(lk._V3_9F, role, scalars, _storm_spec, _expected_plan)
        except lk.LifecycleCheckpointError as exc:
            raise Err(str(exc)) from None
    # ---- ONE cold host->device reconstruction of the FULL BedState (nonmemoizing occurrences)
    try:
        owner2 = wl._cuda_checkpoint_owner(lop)  # re-gate the CURRENT device immediately before the allocation
    except (wl.LifecycleError, ss.SubsurfaceError) as exc:
        raise _cv(exc) from None
    if owner2.device_id != owner.device_id:
        _refuse("the lifecycle's bound device changed")
    _admit_stage(budget, "transfer", host_reader, dev_reader)
    xp = lop.xp
    c0 = cs._counters()
    dev_bed = to_device_tree(host_bed, xp)
    d0 = cs._counters().delta(c0)
    transfers["bed_h2d"] = _transfer_record(d0)
    exp = budget.expected_h2d["bed"]
    if d0.host_to_device != exp["transfers"] or d0.host_to_device_bytes != exp["bytes"]:
        raise Err(f"bed H2D counters ({d0.host_to_device} transfers, {d0.host_to_device_bytes} B) differ from the planned ({exp['transfers']}, {exp['bytes']} B); a concurrent transfer occurred or the leaf inventory is wrong")
    try:
        _device_structure(dev_bed, reference_bed, owner.device_id)
        _admit_stage(budget, "device_validate", host_reader, dev_reader)
        lc._native_checks(dev_bed, lop.context, t_s, ref_ledger_type)
        c1 = cs._counters()
        dev_material = cs._material_sha(dev_bed, lop.context)
        transfers["device_fingerprint"] = _transfer_record(cs._counters().delta(c1))
        if dev_material != host_material:
            _refuse("the device bed content fingerprint differs from the validated host payload (transfer corrupted or host tree changed)")
        del host_bed
        # ---- soil / J / SEPARATE intake / faces through the public constructors (staging released after each)
        _admit_stage(budget, "arrays_soil", host_reader, dev_reader)
        c2 = cs._counters()
        staged = to_device(np.ascontiguousarray(arrays["soil_water_m"]), xp)
        soil = ss.column_state(lop.column, staged, t_s)
        del staged
        _admit_stage(budget, "arrays_history", host_reader, dev_reader)
        sw = to_device(np.ascontiguousarray(arrays["wetting_m"]), xp)
        sc = to_device(np.ascontiguousarray(arrays["cumulative_intake_m"]), xp)
        history = mi.wetting_history(lop.matrix, wetting_m=sw, cumulative_intake_m=sc)
        del sw, sc
        _admit_stage(budget, "arrays_faces", host_reader, dev_reader)
        qx = xp.ascontiguousarray(to_device(np.ascontiguousarray(arrays["qx_m2_s"]), xp))
        qy = xp.ascontiguousarray(to_device(np.ascontiguousarray(arrays["qy_m2_s"]), xp))
        d2 = cs._counters().delta(c2)
        transfers["arrays_h2d"] = _transfer_record(d2)
        expa = budget.expected_h2d["arrays"]
        if d2.host_to_device != expa["transfers"] or d2.host_to_device_bytes != expa["bytes"]:
            raise Err(f"array H2D counters ({d2.host_to_device}, {d2.host_to_device_bytes} B) differ from the planned ({expa['transfers']}, {expa['bytes']} B)")
        del arrays
        surface = dev_bed.water.depth_m
        # ---- NEW genuine operator on the LOADED bed
        _admit_stage(budget, "new_lop", host_reader, dev_reader)
        hk = {"host_available_bytes_fn": owner.host_available_bytes_fn, "host_reserve_bytes": owner.host_reserve_bytes + owner.wrapper_retained_host_bytes}
        new_lop = wl.prepare_lifecycle(lop.column, lop.preferential, lop.matrix, lop.router, lop.graph, lop.topology, lop.evaporation, lop.cover, bed=dev_bed, context=lop.context,
                                       native_reserve_bytes=lop.native_reserve_bytes, extra_reserve_bytes=lop.extra_reserve_bytes, cuda_profile=owner.profile,
                                       host_available_bytes_fn=owner.host_available_bytes_fn, host_reserve_bytes=owner.host_reserve_bytes)
        if new_lop.sha256 != lop.sha256 or new_lop.sha256 != data["binding"]["lop_sha256"]:
            _refuse("the re-prepared lifecycle operator identity differs from the expected/saved one")
        sop = life.get("storm_operator_sha256")
        base = (surface, qx, qy, soil, history, dev_bed)
        records: dict = {"lifecycle": scalars["lifecycle"]}
        objects: dict = {}
        storm_op = prep = gop = tstate = event = estate = None
        if role == "storm":  # V2 only (the V1 pair table never admits it): a genuinely COMPLETED storm restored as a storm, never closed/replayed
            _admit_stage(budget, "owners", host_reader, dev_reader)
            storm_op = ls.prepare_layered_storm(lop.column, lop.preferential, lop.matrix, lop.router, lop.graph, lop.topology, bed=dev_bed, context=lop.context,
                                                extra_reserve_bytes=lop.extra_reserve_bytes, native_reserve_bytes=lop.native_reserve_bytes, **hk)
            if sop is None or storm_op.sha256 != sop:
                _refuse("the re-prepared storm operator identity differs from the saved one (native material or bindings differ)")
            srec = scalars["storm"]
            stree = _storm_tree(srec) if pol.partial_storm else _completed_storm_tree(srec)
            event = ev.prepare_layered_event(storm_op, _storm_spec.schedule, _storm_spec.field, event_origin_s=_storm_spec.event_origin_s, control=_storm_spec.control,
                                             policy=_storm_spec.policy, **hk)
            water = ls.LayeredWaterState(t_s, surface, qx, qy, soil, history, dev_bed, it._typed_ledger(ls.LayeredWaterLedger, stree["ledger"], "ledger"), stree["generation"], storm_op.sha256)
            estate = lk._storm_state_from(srec, event, water)  # event sha / origin / rain end / deadline / generation / ledger equality with the re-prepared event
        elif role == "interevent":
            _admit_stage(budget, "owners", host_reader, dev_reader)
            storm_op = ls.prepare_layered_storm(lop.column, lop.preferential, lop.matrix, lop.router, lop.graph, lop.topology, bed=dev_bed, context=lop.context,
                                                extra_reserve_bytes=lop.extra_reserve_bytes, native_reserve_bytes=lop.native_reserve_bytes, **hk)
            if storm_op.sha256 != sop:
                _refuse("the re-prepared storm operator identity differs from the saved one (native material or bindings differ)")
            prep = iw.prepare_interevent_water(storm_op, lop.evaporation, new_lop.cover, **hk)
            gop = it.prepare_general_interevent(prep, cuda_profile=owner.general_profile, **hk)
            env = scalars["source"]
            if type(env) is not dict or set(env) != {"source_kind", "body"} or env["source_kind"] != kind:
                _refuse("the saved source envelope is malformed or names a different kind than the manifest")
            records.update(t_state=scalars["t_state"], source_envelope=env)
            objects.update(t_state=base, source_envelope=())
            if kind == iw.STORM_END:
                records["receipt"] = env["body"]
                objects["receipt"] = ()
        # A closed `boundary` is restored with `binding=None` and keeps its HISTORICAL `storm_operator_sha256` (the last storm/interevent's operator id) exactly as saved: no operator is prepared or compared for it
        # (as in the CPU codec) and the field is never cleared or rewritten; its integrity is the G1 restorer's hex check plus the recomputed scalar digest. Only an interevent re-prepares and checks the operator.

        def mint(recs: dict, objs: dict) -> Any:
            cap = cc._issue(role=role, lop_sha256=new_lop.sha256, manifest_sha256=data["manifest_sha256"], records=recs, objects=objs)
            caps.append(cap)
            return cap

        _admit_stage(budget, "restore_audit", host_reader, dev_reader)
        if role == "interevent":
            t_parts = {k: records[k] for k in ("t_state", "source_envelope", "receipt") if k in records}
            cap_t = mint(t_parts, {k: objects[k] for k in t_parts})
            tstate = it._restore_checkpoint_general_state(gop, cap_t, records["t_state"], records["source_envelope"], arrays=base, _source_restorer=iw._restore_checkpoint_source_v2)
        cap = mint({"lifecycle": records["lifecycle"]}, {"lifecycle": (*base, storm_op, event, estate, prep, gop, tstate)})
        if pol is pol_mod.V1:
            state = wl._restore_checkpoint_lifecycle_cuda(new_lop, cap, records["lifecycle"], role=role, arrays=base, storm_op=storm_op, prep=prep, gop=gop, tstate=tstate,
                                                           _phase_kinds=wl._V2_PHASE_KINDS)
        else:  # V2/V3: boundary/interevent delegate to the G1 restorer inside the sibling; a storm is restored through O5 (V3: partial allowed, the cursor supplied at CONSTRUCTION)
            state = wl._restore_checkpoint_lifecycle_cuda_v2(new_lop, cap, records["lifecycle"], role=role, arrays=base, storm_op=storm_op, event=event, estate=estate, prep=prep, gop=gop,
                                                              tstate=tstate, _phase_kinds=wl._V2_PHASE_KINDS, daily=daily, _partial_ok=pol.partial_storm)
        lk._v2_cross_checks(state, role, kind)
        digest = cs.physical_digest_cuda(new_lop, state)
        if expected_physical_digest is not None and digest != expected_physical_digest:
            _refuse("the restored physical digest differs from the expected physical digest")
    except Err:
        raise
    except (wl.LifecycleError, ls.LayeredStormError, ev.LayeredEventError, iw.InterEventWaterError, it.GeneralIntereventError, cc.CheckpointCapabilityError, ss.SubsurfaceError, mi.MatrixError,
            lk.LifecycleCheckpointError, lc.LayeredCheckpointError) as exc:
        raise _cv(exc) from None
    receipt = {"profile": pol.profile, "role": role, "source_kind": kind, "t_s": float(t_s), "revision": int(state.revision), "manifest_sha256": data["manifest_sha256"], "files": dict(data["files"]),
               "source": dict(data["source"]), "runtime": dict(data["runtime"]), "state_digest": state.digest, "physical_digest": digest, "material_sha256": material_ref,
               "lop_sha256": new_lop.sha256, "budget": budget.as_dict(), "transfers": transfers, "device_id": owner.device_id, "adopted": False}
    if pol.daily:
        receipt["daily_cursor_digest"] = state.daily.digest()
    return CudaRestoration(new_lop, state, receipt)


def _device_structure(dev_bed: Any, reference_bed: Any, device_id: int) -> None:
    from maple.core.types.voxel import VoxelColumnState

    if type(dev_bed) is not BedState or type(dev_bed.voxel_column) is not VoxelColumnState:
        _refuse("the loaded bed is not a genuine Dense BedState")
    for name in ("voxel_column", "active_layer", "water", "sediment_availability", "ledger", "committed_topography"):
        if type(getattr(dev_bed, name)) is not type(getattr(reference_bed, name)):
            _refuse(f"loaded {name} class differs from the reference bed's (the native class family is fixed)")
    for path, leaf in sp.leaf_occurrences(dev_bed):
        if not cs._is_device_leaf(leaf, device_id):
            _refuse(f"{path} did not land on the lifecycle's bound device {device_id!r} as a CuPy array")
        if not leaf.flags.c_contiguous:
            _refuse(f"{path} is not C-contiguous on the device")
    if [(np.dtype(a.dtype).str, tuple(a.shape)) for _, a in sp.leaf_occurrences(dev_bed)] != [(np.dtype(a.dtype).str, tuple(a.shape)) for _, a in sp.leaf_occurrences(reference_bed)]:
        _refuse("loaded device leaf dtypes/shapes differ from the reference bed's")
