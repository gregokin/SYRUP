"""V3 native9f CUDA lifecycle physical checkpoint for Phase 1.5 DAILY lineages — PUBLIC entry points (written without any GPU/device/CuPy execution; UNQUALIFIED until Root's guarded run).

V3 = schema `/3`, profile `native9f_dense_lifecycle_cuda_v3`, manifest `lifecycle_checkpoint_cuda_v3.json`: the closed `boundary`, accepted `interevent` (`storm_end` / `boundary_origin`) and a
`storm` that may be PARTIAL (accepted/rejected steps, quiet history, next dt cap, forcing index, anchors, cumulative ledgers and window marks are the existing storm scalar record; completion is
never required, fabricated or defaulted), each with ONE extra closed scalar part, `daily`: the bounded `DailyCursor` (versioned clock contract, run, next ordinal, the pending occurrence with its
forcing/event identity and capacity-fallback failing-column count, the last consumed digests). The T request sequence, J, modeled matrix/preferential intake, the single device water inventory and the
actual native Dense bed travel through the SAME audited engines as V1/V2 (`lifecycle_checkpoint_cuda`, `lifecycle_checkpoint_cuda_load`) under the registered literal V3 policy; V1/V2 keep their
exact meaning (V2 stays completed-storm-only) and refuse a cursor state before any I/O, V3 refuses a cursor-free one, and the three codec versions never load each other's directories. V3 checks
its helper modules against ITS OWN reviewed identity set (`lifecycle_checkpoint_cuda_policy.V3_HELPER_PINS`), never by overwriting the C1 pins.

HOW A CALLER RESUMES (same contract as the CPU V3 codec, `lifecycle_checkpoint_v3`): rebuild the FIXED components from the genuine CUDA file factories; supply the expected identity, role, (interevent)
closed source kind, the reference bed, and for a storm the genuine `StormSpec` reconstructed from the ORIGINAL canonical plan (`forcing = plan.bind_forcing(origin)`, LOCAL schedule `[0, D]`, physical
`event_origin_s`, device rainfall field from the device-resident active mask, same control/quiet policy); the loader requires that spec to reproduce the saved forcing digest and event identity and,
when `expected_plan` is given (re-validated with `recharge.verify_plan` before any I/O), that plan to equal the saved occurrence. The loader never completes a storm and never replays rain.
"""

from __future__ import annotations

from typing import Any

from maple_syrup import lifecycle_checkpoint as lk
from maple_syrup import lifecycle_checkpoint_cuda as cs
from maple_syrup import lifecycle_checkpoint_cuda_load as ld
from maple_syrup import lifecycle_checkpoint_cuda_policy as pol_mod
from maple_syrup import lifecycle_checkpoint_cuda_v2_plan as v2plan
from maple_syrup import lifecycle_checkpoint_cuda_v3_plan as v3plan

__all__ = ["PROFILE_V3", "SCHEMA_V3", "estimate_load_budget_cuda_v3", "estimate_save_budget_cuda_v3", "load_lifecycle_checkpoint_cuda_v3", "save_lifecycle_checkpoint_cuda_v3"]

SCHEMA_V3 = pol_mod.V3.schema
PROFILE_V3 = pol_mod.V3.profile


def estimate_save_budget_cuda_v3(meta: Any, *, caller_host_reserve_bytes: int, caller_device_reserve_bytes: int, extra_reserve_bytes: int = 0) -> Any:
    """PURE V3 save budget over sanitized metadata (V2 terms + the derived `daily` allowance)."""
    storm = v2plan.StormSaveTerms() if meta.role == "storm" else None
    return v3plan.plan_save_v3(meta, caller_host_reserve_bytes=caller_host_reserve_bytes, caller_device_reserve_bytes=caller_device_reserve_bytes, extra_reserve_bytes=extra_reserve_bytes,
                               storm=storm)


def estimate_load_budget_cuda_v3(meta: Any, *, caller_host_reserve_bytes: int, caller_device_reserve_bytes: int, extra_reserve_bytes: int = 0, storm: Any = None) -> Any:
    """PURE V3 load budget over sanitized metadata (`storm` = `StormLoadTerms` for the storm role, else None)."""
    return v3plan.plan_load_v3(meta, caller_host_reserve_bytes=caller_host_reserve_bytes, caller_device_reserve_bytes=caller_device_reserve_bytes, extra_reserve_bytes=extra_reserve_bytes,
                               storm=storm)


def save_lifecycle_checkpoint_cuda_v3(path: Any, lop: Any, state: Any, identity: dict, *, limits: Any, caller_host_reserve_bytes: int, caller_device_reserve_bytes: int,
                                      host_available_bytes_fn: Any = None, device_available_bytes_fn: Any = None, disk_available_bytes_fn: Any = None,
                                      extra_reserve_bytes: int = 0) -> cs.CudaSaveReceipt:
    """Write a NEW sealed V3 checkpoint directory for the accepted DAILY `state` (closed boundary, accepted interevent, or a storm — partial or completed). Same ceremony/guards as V1/V2: typed
    refusal and nothing sealed on any failure; a state without a cursor is refused before any I/O."""
    return cs._save_engine(path, lop, state, identity, limits=limits, caller_host_reserve_bytes=caller_host_reserve_bytes, caller_device_reserve_bytes=caller_device_reserve_bytes,
                           host_available_bytes_fn=host_available_bytes_fn, device_available_bytes_fn=device_available_bytes_fn, disk_available_bytes_fn=disk_available_bytes_fn,
                           extra_reserve_bytes=extra_reserve_bytes, _policy=pol_mod.V3)


def load_lifecycle_checkpoint_cuda_v3(path: Any, lop: Any, expected_identity: dict, *, expected_role: str, expected_source_kind: str | None = None, storm: Any = None, expected_plan: Any = None,
                                      limits: Any, reference_bed: Any, caller_host_reserve_bytes: int, caller_device_reserve_bytes: int, host_available_bytes_fn: Any = None,
                                      device_available_bytes_fn: Any = None, disk_available_bytes_fn: Any = None, extra_reserve_bytes: int = 0,
                                      expected_physical_digest: Any = None) -> ld.CudaRestoration:
    """Load a sealed V3 checkpoint into a NEW bed/operator/daily state (UNADOPTED). `storm` is required iff `expected_role == "storm"`; `expected_plan` (optional) is validated first."""
    try:
        lk._validate_expected_plan(expected_plan)  # before any fence, budget, source audit or I/O
    except lk.LifecycleCheckpointError as exc:
        raise cs.LifecycleCheckpointCudaError(str(exc)) from None
    return ld._load_engine(path, lop, expected_identity, expected_role=expected_role, expected_source_kind=expected_source_kind, limits=limits, reference_bed=reference_bed,
                           caller_host_reserve_bytes=caller_host_reserve_bytes, caller_device_reserve_bytes=caller_device_reserve_bytes, host_available_bytes_fn=host_available_bytes_fn,
                           device_available_bytes_fn=device_available_bytes_fn, disk_available_bytes_fn=disk_available_bytes_fn, extra_reserve_bytes=extra_reserve_bytes,
                           expected_physical_digest=expected_physical_digest, _storm=storm, _policy=pol_mod.V3, _expected_plan=expected_plan)
