"""Public state-to-budget facade of the V2 native9f CUDA lifecycle checkpoint codec (candidate; UNQUALIFIED, written without any GPU/CuPy/device execution).

`estimate_save_cuda_v2(lop, state, ...)` and `estimate_load_cuda_v2(lop, ...)` return the ACTUAL `CudaSaveBudget` / `CudaLoadBudget` that `save_lifecycle_checkpoint_cuda_v2` /
`load_lifecycle_checkpoint_cuda_v2` compute for the same genuine lifecycle, state and reserves, without any checkpoint I/O. They run the engines' own fences in the engines' own
order under the registered V2 policy, then the existing V2 planners: no formula, serialization or physics lives here.

DEVICE-GATED metadata planning (not device-free arithmetic): the owner gate (`water_lifecycle._cuda_checkpoint_owner`) queries the CURRENT device through the column preflight and
checks the cached native9f source association; it reads no payload, hashes nothing and copies nothing. Excluded (they belong to the engines): availability readings, disk/path/limits,
source-package rehash, runtime record, material/physical/spec digests, any file, D2H/H2D, array creation, or state mutation. The sanitized-metadata planners
`estimate_save_budget_cuda_v2` / `estimate_load_budget_cuda_v2` remain the pure device-free alternative. Reserves are validated (like the engines) before the lifecycle is touched;
for load the role/kind/pair/storm-presence/storm-type checks also precede any device access.

Same-package private helpers are reused deliberately (the codec's own trusted seams): `lifecycle_checkpoint_cuda` `_reserve/_fence_types/_fence_bed/_convert/_refuse`,
`lifecycle_checkpoint_cuda_load` `_fence_lop/_fence_reference_bed/_gather_v2`, `water_lifecycle` `_cuda_checkpoint_owner/_check_state/_checkpoint_state_record_cuda_v2`.
A native39/CPU/unknown lifecycle is refused by those fences (never a positive). Errors are `LifecycleCheckpointCudaError`.
"""

from __future__ import annotations

from typing import Any

from maple_syrup import interevent_transpiration as it
from maple_syrup import interevent_water as iw
from maple_syrup import layered_storm as ls
from maple_syrup import lifecycle_checkpoint as lk
from maple_syrup import lifecycle_checkpoint_cuda as cs
from maple_syrup import lifecycle_checkpoint_cuda_load as ld
from maple_syrup import lifecycle_checkpoint_cuda_load_plan as lp
from maple_syrup import lifecycle_checkpoint_cuda_plan as sp
from maple_syrup import lifecycle_checkpoint_cuda_policy as pol_mod
from maple_syrup import lifecycle_checkpoint_cuda_v2 as v2
from maple_syrup import lifecycle_checkpoint_cuda_v2_plan as v2plan
from maple_syrup import subsurface as ss
from maple_syrup import water_lifecycle as wl
from maple_syrup import water_lifecycle_cuda as wlc

__all__ = ["estimate_load_cuda_v2", "estimate_save_cuda_v2"]

_POLICY = pol_mod.V2
_PLAN_ERRORS = (wl.LifecycleError, sp.CudaSavePlanError, lp.CudaLoadPlanError, wlc.CudaLifecycleError, it.GeneralIntereventError, iw.InterEventWaterError, ls.LayeredStormError)


def estimate_save_cuda_v2(lop: Any, state: Any, *, caller_host_reserve_bytes: int, caller_device_reserve_bytes: int, extra_reserve_bytes: int = 0) -> sp.CudaSaveBudget:
    """The V2 save budget of the accepted `state` (closed boundary, accepted interevent, or genuinely COMPLETED storm): exactly `CudaSaveReceipt.budget` of the real save."""
    hr = cs._reserve(caller_host_reserve_bytes, "caller_host_reserve_bytes")
    dr = cs._reserve(caller_device_reserve_bytes, "caller_device_reserve_bytes")
    ex = cs._reserve(extra_reserve_bytes, "extra_reserve_bytes")
    role, kind = cs._fence_types(lop, state, _POLICY)
    try:  # current-device gate: the owner accessor queries the CURRENT device before anything else is read
        owner = wl._cuda_checkpoint_owner(lop)
        wl._check_state(lop, state)
        wl._checkpoint_state_record_cuda_v2(lop, state)
    except (wl.LifecycleError, ss.SubsurfaceError) as exc:
        raise cs._convert(exc) from None
    if isinstance(owner.device_id, bool) or not isinstance(owner.device_id, int):
        cs._refuse("the lifecycle's bound device id is not an integer")
    bed = cs._fence_bed(lop, state, owner)
    if (role, kind) not in _POLICY.pairs or (kind == "boundary_origin" and not owner.boundary_capable):
        cs._refuse(f"role/source kind {(role, kind)!r} is not covered by this lifecycle's profile")
    try:
        meta = v2plan.extract_save_metadata_v2(
            bed=bed, context_fraction=lop.context.initial_available_fraction, role=role, source_kind=kind, n_cells=owner.n_cells, n_layers=owner.n_layers, n_classes=owner.n_classes,
            shape=owner.shape, wrapper_retained_host_bytes=owner.wrapper_retained_host_bytes, owner_audit_device_bytes=int(lop.memory.audit_scratch_bytes))
        return v2.estimate_save_budget_cuda_v2(meta, caller_host_reserve_bytes=hr, caller_device_reserve_bytes=dr, extra_reserve_bytes=ex)
    except _PLAN_ERRORS as exc:
        raise cs._convert(exc) from None


def estimate_load_cuda_v2(lop: Any, *, expected_role: str, expected_source_kind: str | None = None, reference_bed: Any, storm: Any = None, caller_host_reserve_bytes: int,
                          caller_device_reserve_bytes: int, extra_reserve_bytes: int = 0) -> lp.CudaLoadBudget:
    """The V2 load budget for a NEW lifecycle `lop` and its `reference_bed` (storm role: the caller's expected `StormSpec`, validated for shape/device but never digested): exactly the
    budget of `load_lifecycle_checkpoint_cuda_v2` (`restoration.receipt["budget"]`) for the same arguments."""
    hr = cs._reserve(caller_host_reserve_bytes, "caller_host_reserve_bytes")
    dr = cs._reserve(caller_device_reserve_bytes, "caller_device_reserve_bytes")
    ex = cs._reserve(extra_reserve_bytes, "extra_reserve_bytes")
    if type(expected_role) is not str or not (expected_source_kind is None or type(expected_source_kind) is str):
        cs._refuse("expected_role must be an exact str and expected_source_kind an exact str or None")
    role, kind = expected_role, expected_source_kind
    if (role, kind) not in _POLICY.scalar_parts:
        cs._refuse(f"(expected_role, expected_source_kind) {(role, kind)!r} is not a covered closed pair {sorted(_POLICY.scalar_parts, key=str)}")
    if (role == "storm") != (storm is not None):
        cs._refuse("a StormSpec is required for (and only for) the storm role")
    if storm is not None and type(storm) is not lk.StormSpec:
        cs._refuse("storm must be an exact lifecycle_checkpoint.StormSpec")
    ld._fence_lop(lop)
    try:  # current-device gate
        owner = wl._cuda_checkpoint_owner(lop)
    except (wl.LifecycleError, ss.SubsurfaceError) as exc:
        raise cs._convert(exc) from None
    if isinstance(owner.device_id, bool) or not isinstance(owner.device_id, int):
        cs._refuse("the lifecycle's bound device id is not an integer")
    if kind == "boundary_origin" and not owner.boundary_capable:
        cs._refuse("a boundary-origin source exists only under the explicit CUDA9F_BOUNDARY_PROFILE lifecycle; refused")
    ld._fence_reference_bed(lop, reference_bed, owner)
    try:
        if role == "storm":
            v2.validate_storm_spec(storm, lop, owner.device_id)
        meta, storm_terms = ld._gather_v2(lop, owner, role, kind, reference_bed, storm)
        return v2.estimate_load_budget_cuda_v2(meta, caller_host_reserve_bytes=hr, caller_device_reserve_bytes=dr, extra_reserve_bytes=ex, storm=storm_terms)
    except _PLAN_ERRORS as exc:
        raise cs._convert(exc) from None
