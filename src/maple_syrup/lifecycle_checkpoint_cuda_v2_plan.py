"""PURE V2 plan wrappers of the native9f CUDA lifecycle physical checkpoint (A6 G5c-d; UNQUALIFIED, written without any device access).

Wraps the accepted V1 plans (`lifecycle_checkpoint_cuda_plan`, `lifecycle_checkpoint_cuda_load_plan`, byte-exact, never edited) with the V2 additions via `dataclasses.replace`: the manifest/metadata/serialized envelope of the V2 `binding.storm` field and
the completed-storm scalar record (DERIVED from a worst-case record, never from an observation and never by raising a V1 cap), the spec-digest host stage, the event audit device scratch (save) and the begin-storm owner stage / event-owned maps (load).
Imports no CuPy and no native runtime beyond the pure plan modules; never calls a device-gated accessor, builds no parent/duck/capability. Borrowed `StormSpec` arrays and the caller's reference bed are REPORTED, never charged; the event's own device copies ARE charged. Exclusions as the V1 plans (CuPy import/context/JIT, pools, pinned, page cache, process).
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from maple_syrup import checkpoint_capability as cc
from maple_syrup import layered_checkpoint as lc
from maple_syrup import layered_storm as ls
from maple_syrup import lifecycle_checkpoint as lk
from maple_syrup import lifecycle_checkpoint_cuda_load_plan as lp
from maple_syrup import lifecycle_checkpoint_cuda_plan as sp

__all__ = ["ENVELOPE_NONSTORM", "ENVELOPE_STORM", "StormLoadTerms", "StormSaveTerms", "extract_save_metadata_v2", "plan_load_v2", "plan_save_v2", "storm_record_worst_case"]

PROFILE_V2 = "native9f_dense_lifecycle_cuda_v2"
EVENT_AUDIT_N_DOUBLES = 12  # `prepare_layered_event`: "twelve N-double temporaries for field validation, rainfall, quiet metrics and anchor reductions"


def storm_record_worst_case() -> dict:
    """Closed worst-case ENCODED storm scalar record (`lifecycle_checkpoint._storm_record` key set): every float at its widest hex form, every integer 2**63-1, 64-hex strings."""
    wide = float.fromhex("-0x1.fffffffffffffp-1022").hex()
    f, big, sha = {"f": wide}, 2 ** 63 - 1, "f" * 64
    ledger = {x.name: (big if x.name == "n_steps" else f) for x in dataclasses.fields(ls.LayeredWaterLedger)}
    return {"generation": big, "event_sha256": sha, "ledger": ledger, "anchor_ledger": dict(ledger), "anchor_surface_m3": f, "anchor_soil_m3": f, "start_generation": big, "next_dt_cap_s": f,
            "forcing_index": big, "accepted_steps": big, "rejected_attempts": big, "quiet": {"since_s": f, "window_depth_change_m": f}, "completed": True, "event_origin_s": f, "rainfall_end_s": f,
            "deadline_s": f}


def _text_cost(obj: Any, prefix: str) -> tuple[int, int]:
    text = json.dumps(obj, indent=1, sort_keys=True)
    return len(text) + 3 * text.count("\n") + len(prefix) + 2, lk._count_json_nodes(text) + 2


def _derive() -> tuple[tuple[int, int], tuple[int, int]]:
    wide = float.fromhex("-0x1.fffffffffffffp-1022").hex()
    storm_scalar = _text_cost(cc.encode_tree(storm_record_worst_case()), '"storm": ')
    binding_storm = _text_cost({"spec_sha256": "f" * 64, "event_origin_s": wide, "rainfall_end_s": wide, "deadline_s": wide}, '"storm": ')
    null_binding = _text_cost(None, '"storm": ')
    storm = (storm_scalar[0] + binding_storm[0], storm_scalar[1] + binding_storm[1])
    return storm, null_binding


ENVELOPE_STORM, ENVELOPE_NONSTORM = _derive()


def _strict_nonneg_int(value: Any, name: str, error: type) -> None:
    """Exact primitive validation BEFORE any conversion or arithmetic: a non-Boolean integer (`int` or NumPy integer, the V1 plans' own convention) >= 0. No truncation, coercion, clamp or default."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
        raise error(f"{name} must be a non-Boolean integer >= 0 (no coercion), got {type(value).__name__}")


@dataclass(frozen=True)
class StormSaveTerms:
    """Primitives for the storm role (save): the event audit's N-double temporaries are derived from N; the schedule host arrays are hashed in place (no copy). Every field is validated at construction and again at plan entry."""

    spec_hash_host_bytes: int = lc.HASH_CHUNK_BYTES

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        _strict_nonneg_int(self.spec_hash_host_bytes, "spec_hash_host_bytes", sp.CudaSavePlanError)


@dataclass(frozen=True)
class StormLoadTerms:
    """Primitives for the storm role (load), all from real pure owner metadata functions evaluated by the codec after its device gate."""

    begin_storm_device_bytes: int
    begin_storm_host_bytes: int
    event_extra_bytes: int
    storm_audit_device_bytes: int
    spec_hash_host_bytes: int = lc.HASH_CHUNK_BYTES

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        for name in ("begin_storm_device_bytes", "begin_storm_host_bytes", "event_extra_bytes", "storm_audit_device_bytes", "spec_hash_host_bytes"):
            _strict_nonneg_int(getattr(self, name), name, lp.CudaLoadPlanError)


def spec_digest_expected_d2h(n_cells: int) -> dict:
    """Named, separate expected device-to-host traffic of the spec-digest stage: the f64 rainfall field streamed in <= HASH_CHUNK_BYTES chunks (bytes exact; transfer count assumes a C-contiguous field); the host schedule is hashed in place (no D2H).
    Never merged with the native-writer / companion / bed-transfer expectations."""
    nbytes = 8 * int(n_cells)
    return {"bytes": nbytes, "transfers": -(-nbytes // lc.HASH_CHUNK_BYTES), "assumes_c_contiguous_f64_field": True, "host_schedule_d2h_bytes": 0}


def extract_save_metadata_v2(*, role: str, source_kind: str | None, **kw: Any) -> sp.CudaSaveMetadata:
    """The V1 metadata extraction plus the `storm` role (the leaf/occurrence arithmetic is role-independent; the V1 extractor and its closed ROLES are untouched)."""
    if role == "storm":
        if source_kind is not None:
            raise sp.CudaSavePlanError("a storm role has no source kind")
        return dataclasses.replace(sp.extract_save_metadata(role="boundary", source_kind=None, **kw), role="storm")
    return sp.extract_save_metadata(role=role, source_kind=source_kind, **kw)


def _envelope(role: str) -> tuple[int, int]:
    return ENVELOPE_STORM if role == "storm" else ENVELOPE_NONSTORM


def plan_save_v2(meta: sp.CudaSaveMetadata, *, caller_host_reserve_bytes: int, caller_device_reserve_bytes: int, extra_reserve_bytes: int = 0,
                 storm: StormSaveTerms | None = None) -> sp.CudaSaveBudget:
    """V2 save budget = the V1 plan over the same leaf occurrences + the V2 envelope + (storm) the spec-digest stage and the event audit scratch."""
    if (meta.role == "storm") != (storm is not None):
        raise sp.CudaSavePlanError("storm terms are required for (and only for) the storm role")
    if storm is not None:
        if type(storm) is not StormSaveTerms:
            raise sp.CudaSavePlanError("storm must be exactly a StormSaveTerms record")
        storm.validate()  # again at entry, before any arithmetic or conversion
    base = sp.plan_cuda_save(dataclasses.replace(meta, role="boundary") if meta.role == "storm" else meta, caller_host_reserve_bytes=caller_host_reserve_bytes,
                             caller_device_reserve_bytes=caller_device_reserve_bytes, extra_reserve_bytes=extra_reserve_bytes, profile=PROFILE_V2)
    env_b, env_n = _envelope(meta.role)
    delta_md = 3 * env_b + env_n * lc.NODE_BYTES
    hr, ex = base.caller_host_reserve_bytes, base.extra_reserve_bytes
    req = {s: {**v, "host_incremental_bytes": v["host_incremental_bytes"] + delta_md, "host_with_reserves": v["host_with_reserves"] + delta_md} for s, v in base.stage_requirements.items()}
    host_terms = dict(base.host_terms)
    device_terms = dict(base.device_terms)
    device_save = base.device_save_bytes
    if storm is not None:
        io = lc.IO_CHUNK_BYTES + lc.HASH_CHUNK_BYTES
        spec_host = storm.spec_hash_host_bytes + lc.HASH_CHUNK_BYTES + base.metadata_bytes + delta_md + io
        req["spec_digest"] = {"host_incremental_bytes": spec_host, "device_incremental_bytes": 0, "host_with_reserves": spec_host + hr + ex, "device_with_reserves": base.caller_device_reserve_bytes,
                              "expected_d2h": spec_digest_expected_d2h(meta.n_cells)}
        event_audit = EVENT_AUDIT_N_DOUBLES * 8 * meta.n_cells
        device_terms["event_audit_scratch"] = event_audit
        host_terms["spec_digest_stage"] = storm.spec_hash_host_bytes
        device_save = max(device_save, event_audit)
        v = req["source_and_validation"]
        req["source_and_validation"] = {**v, "device_incremental_bytes": max(v["device_incremental_bytes"], device_save), "device_with_reserves": max(v["device_with_reserves"], device_save + base.caller_device_reserve_bytes)}
    host_peak = max(v["host_incremental_bytes"] for v in req.values())
    return dataclasses.replace(
        base, role=meta.role, manifest_cap_bytes=base.manifest_cap_bytes + env_b, manifest_max_nodes=base.manifest_max_nodes + env_n, metadata_bytes=base.metadata_bytes + delta_md,
        serialized_bytes=base.serialized_bytes + env_b, disk_bytes=base.disk_bytes + env_b, scratch_bytes=base.scratch_bytes + delta_md, host_save_bytes=host_peak,
        host_save_capture_first_bytes=max(base.host_save_capture_first_bytes + delta_md, host_peak), device_save_bytes=device_save, host_required_with_reserves=host_peak + hr + ex,
        device_required_with_reserves=device_save + base.caller_device_reserve_bytes, stage_requirements=req, host_terms=host_terms, device_terms=device_terms)


def plan_load_v2(meta: lp.CudaLoadMetadata, *, caller_host_reserve_bytes: int, caller_device_reserve_bytes: int, extra_reserve_bytes: int = 0,
                 storm: StormLoadTerms | None = None) -> lp.CudaLoadBudget:
    """V2 load budget = the V1 load plan (storm: with the begin-storm owner stage and the storm audit figure) + the V2 envelope + the spec-digest stage; the event-owned device maps persist through `restore_audit`."""
    role = meta.save.role
    if (role == "storm") != (storm is not None):
        raise lp.CudaLoadPlanError("storm terms are required for (and only for) the storm role")
    if storm is not None:
        if type(storm) is not StormLoadTerms:
            raise lp.CudaLoadPlanError("storm must be exactly a StormLoadTerms record")
        storm.validate()  # again at entry, before any arithmetic or conversion
    m = meta
    if storm is not None:
        m = dataclasses.replace(meta, owner_stage_table=(("begin_storm", int(storm.begin_storm_device_bytes)),), owner_host_bytes=int(storm.begin_storm_host_bytes),
                                audit_device_bytes=int(storm.storm_audit_device_bytes), reset_device_bytes=0, reset_host_bytes=0)
    base = lp.plan_cuda_load(m, caller_host_reserve_bytes=caller_host_reserve_bytes, caller_device_reserve_bytes=caller_device_reserve_bytes, extra_reserve_bytes=extra_reserve_bytes,
                             profile=PROFILE_V2)
    env_b, env_n = _envelope(role)
    delta_md = 3 * env_b + env_n * lc.NODE_BYTES
    hr, ex, dr = base.caller_host_reserve_bytes, base.extra_reserve_bytes, base.caller_device_reserve_bytes
    req = {s: {**v, "host_incremental_bytes": v["host_incremental_bytes"] + delta_md, "host_with_reserves": v["host_with_reserves"] + delta_md} for s, v in base.stage_requirements.items()}
    host_terms, device_terms = dict(base.host_terms), dict(base.device_terms)
    if storm is not None:
        io = lc.IO_CHUNK_BYTES + lc.HASH_CHUNK_BYTES
        spec_host = storm.spec_hash_host_bytes + base.metadata_bytes + delta_md + io
        req["spec_digest"] = {"host_incremental_bytes": spec_host, "device_live_bytes": 0, "device_need_bytes": 0, "device_peak_prefix_bytes": 0, "host_with_reserves": spec_host + hr + ex,
                              "device_need_with_reserve": dr, "expected_d2h": spec_digest_expected_d2h(meta.save.n_cells)}
        a = req["restore_audit"]  # the event-owned device maps stay alive through the restore audit
        live = a["device_live_bytes"] + int(storm.event_extra_bytes)
        req["restore_audit"] = {**a, "device_live_bytes": live, "device_peak_prefix_bytes": live + a["device_need_bytes"]}
        host_terms["spec_digest_stage"] = storm.spec_hash_host_bytes
        device_terms["event_owned_maps"] = int(storm.event_extra_bytes)
        device_terms["begin_storm_owner_stage"] = int(storm.begin_storm_device_bytes)
    host_peak = max(v["host_incremental_bytes"] for v in req.values())
    dev_peak = max(v["device_peak_prefix_bytes"] for v in req.values())
    return dataclasses.replace(
        base, manifest_cap_bytes=base.manifest_cap_bytes + env_b, manifest_max_nodes=base.manifest_max_nodes + env_n, metadata_bytes=base.metadata_bytes + delta_md,
        serialized_bytes=base.serialized_bytes + env_b, disk_bytes=base.disk_bytes + env_b, scratch_bytes=base.scratch_bytes + delta_md, stage_requirements=req, host_peak_bytes=host_peak,
        host_required_with_reserves=host_peak + hr + ex, device_peak_whole_call_bytes=dev_peak, device_required_with_reserves=dev_peak + dr, host_terms=host_terms, device_terms=device_terms)
