"""PURE V3 plan wrappers of the native9f CUDA lifecycle physical checkpoint for Phase 1.5 DAILY lineages (written without any device access).

Wraps the V2 plans (`lifecycle_checkpoint_cuda_v2_plan`, which themselves wrap the accepted V1 plans, never edited) with the ONE extra V3 metadata term: the closed `daily` scalar part of the
manifest. Its worst-case size is DERIVED from the declared record bounds (`lifecycle_checkpoint._derive_daily_allowance`: every digest 64 hex, every counter at the declared
`recharge.MAX_RECORD_COUNT`, the clock-contract literal) and enforced by the cursor codec itself on encode AND decode, so no accepted record can exceed it; no V1/V2 cap or slack is raised or
borrowed. The storm record of a PARTIAL storm has the same closed key set as the completed one but is NOT byte-identical in the worst case: the V2 storm term is derived with `completed: true` (4 characters) while a partial storm stores `false` (5 characters), i.e. exactly ONE byte more in the storm scalar record text. V3 therefore adds an explicit V3-only storm delta (`PARTIAL_FLAG_BYTES = 1` on the manifest/serialized/disk figures and `3 * 1` on the metadata/scratch/host-stage figures for the three concurrent text lifetimes, no extra nodes); V2 meaning and caps are unchanged.
Imports no CuPy; never calls a device-gated accessor.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from maple_syrup import layered_checkpoint as lc
from maple_syrup import lifecycle_checkpoint as lk
from maple_syrup import lifecycle_checkpoint_cuda_load_plan as lp
from maple_syrup import lifecycle_checkpoint_cuda_plan as sp
from maple_syrup import lifecycle_checkpoint_cuda_v2_plan as v2plan

__all__ = ["DAILY_NODES", "DAILY_RECORD_BYTES", "PARTIAL_FLAG_BYTES", "PROFILE_V3", "plan_load_v3", "plan_save_v3"]

PROFILE_V3 = "native9f_dense_lifecycle_cuda_v3"
DAILY_RECORD_BYTES, DAILY_NODES = lk._DAILY_BYTES, lk._DAILY_NODES
_DELTA_MD = 3 * DAILY_RECORD_BYTES + DAILY_NODES * lc.NODE_BYTES  # three concurrent text lifetimes + the decoded nodes (the V2 rule)


#: A PARTIAL storm stores `"completed": false` where the V2 worst case was derived with `true`: `len("false") - len("true")` = 1 byte of the storm scalar record text (V3-only; storm role only).
PARTIAL_FLAG_BYTES = len("false") - len("true")


def _storm_flag_bytes(role: str) -> int:
    return PARTIAL_FLAG_BYTES if role == "storm" else 0


def _bump(req: dict, extra: int) -> dict:
    d = _DELTA_MD + 3 * extra
    return {s: {**v, "host_incremental_bytes": v["host_incremental_bytes"] + d, "host_with_reserves": v["host_with_reserves"] + d} for s, v in req.items()}


def plan_save_v3(meta: sp.CudaSaveMetadata, *, caller_host_reserve_bytes: int, caller_device_reserve_bytes: int, extra_reserve_bytes: int = 0,
                 storm: v2plan.StormSaveTerms | None = None) -> sp.CudaSaveBudget:
    """V3 save budget = the V2 save budget + the derived `daily` allowance on the manifest/metadata/serialized/disk/scratch figures and on every host stage."""
    b = v2plan.plan_save_v2(meta, caller_host_reserve_bytes=caller_host_reserve_bytes, caller_device_reserve_bytes=caller_device_reserve_bytes, extra_reserve_bytes=extra_reserve_bytes,
                            storm=storm)
    x = _storm_flag_bytes(meta.role)
    d = _DELTA_MD + 3 * x
    req = _bump(b.stage_requirements, x)
    host_peak = max(v["host_incremental_bytes"] for v in req.values())
    hr, ex = b.caller_host_reserve_bytes, b.extra_reserve_bytes
    host_terms = {**b.host_terms, "daily_record_allowance": _DELTA_MD, "partial_flag_allowance": 3 * x}
    return dataclasses.replace(
        b, profile=PROFILE_V3, manifest_cap_bytes=b.manifest_cap_bytes + DAILY_RECORD_BYTES + x, manifest_max_nodes=b.manifest_max_nodes + DAILY_NODES, metadata_bytes=b.metadata_bytes + d,
        serialized_bytes=b.serialized_bytes + DAILY_RECORD_BYTES + x, disk_bytes=b.disk_bytes + DAILY_RECORD_BYTES + x, scratch_bytes=b.scratch_bytes + d, stage_requirements=req,
        host_save_bytes=host_peak, host_save_capture_first_bytes=max(b.host_save_capture_first_bytes + d, host_peak), host_required_with_reserves=host_peak + hr + ex,
        host_terms=host_terms)


def plan_load_v3(meta: lp.CudaLoadMetadata, *, caller_host_reserve_bytes: int, caller_device_reserve_bytes: int, extra_reserve_bytes: int = 0,
                 storm: v2plan.StormLoadTerms | None = None) -> lp.CudaLoadBudget:
    """V3 load budget = the V2 load budget + the derived `daily` allowance (same terms as the save)."""
    b = v2plan.plan_load_v2(meta, caller_host_reserve_bytes=caller_host_reserve_bytes, caller_device_reserve_bytes=caller_device_reserve_bytes, extra_reserve_bytes=extra_reserve_bytes,
                            storm=storm)
    x = _storm_flag_bytes(meta.save.role)
    d = _DELTA_MD + 3 * x
    req = _bump(b.stage_requirements, x)
    host_peak = max(v["host_incremental_bytes"] for v in req.values())
    hr, ex = b.caller_host_reserve_bytes, b.extra_reserve_bytes
    host_terms = {**b.host_terms, "daily_record_allowance": _DELTA_MD, "partial_flag_allowance": 3 * x}
    return dataclasses.replace(
        b, profile=PROFILE_V3, manifest_cap_bytes=b.manifest_cap_bytes + DAILY_RECORD_BYTES + x, manifest_max_nodes=b.manifest_max_nodes + DAILY_NODES, metadata_bytes=b.metadata_bytes + d,
        serialized_bytes=b.serialized_bytes + DAILY_RECORD_BYTES + x, disk_bytes=b.disk_bytes + DAILY_RECORD_BYTES + x, scratch_bytes=b.scratch_bytes + d, stage_requirements=req,
        host_peak_bytes=host_peak, host_required_with_reserves=host_peak + hr + ex, host_terms=host_terms)
