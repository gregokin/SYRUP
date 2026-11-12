"""V2 native9f CUDA lifecycle physical checkpoint — PUBLIC entry points and storm-spec helpers (A6 G5c-d; UNQUALIFIED, written without any GPU/device/CuPy execution).

V2 = schema `/2`, profile `native9f_dense_lifecycle_cuda_v2`, manifest `lifecycle_checkpoint_cuda_v2.json`: closed `boundary`, accepted `interevent` (`storm_end` / `boundary_origin`) AND a genuinely COMPLETED `storm`. It runs on the SAME audited engines as V1 under the registered literal
V2 policy (V1 public entry points, literals, role refusal and behaviour are unchanged, and V1/V2 refuse each other's directories). A completed storm is saved/restored as a storm (never closed, restamped, dried, re-flagged or replayed): the next `enter_interevent_phase` creates the genuine STORM_END.
Forcing identity: `binding.storm.spec_sha256` is a domain-tagged streamed digest of the caller's `StormSpec` (schedule edges/intensities, native origin, control, quiet policy, the OWNED effective rainfall field streamed from the device in <= 1 MiB chunks). The expected spec is BORROWED from the caller (the loader never reads forcing from the
file); the re-prepared `event.sha256` stays the final authority. Not the MASY calendar restart; no incomplete-storm checkpoint; no physics.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import numpy as np

from maple_syrup import layered_event as ev
from maple_syrup import layered_storm as ls
from maple_syrup import lifecycle_checkpoint as lk
from maple_syrup import lifecycle_checkpoint_cuda as cs
from maple_syrup import lifecycle_checkpoint_cuda_load as ld
from maple_syrup import lifecycle_checkpoint_cuda_policy as pol_mod
from maple_syrup import lifecycle_checkpoint_cuda_v2_plan as v2plan
from maple_syrup.rainfall import RainfallField, RainfallSchedule

__all__ = ["estimate_load_budget_cuda_v2", "estimate_save_budget_cuda_v2", "load_lifecycle_checkpoint_cuda_v2", "save_lifecycle_checkpoint_cuda_v2", "spec_sha256_of_event", "spec_sha256_of_spec",
           "storm_binding", "validate_storm_spec"]

SPEC_DOMAIN = b"maple_syrup.lifecycle_checkpoint_cuda_v2.storm_spec/1"
Err = cs.LifecycleCheckpointCudaError


def _refuse(msg: str) -> None:
    raise Err(msg)


def _control_fields(c: Any) -> list:
    return [float(c.max_dt_s).hex(), float(c.min_dt_s).hex(), int(c.max_retries), int(c.max_steps), float(c.max_recession_s).hex()]


def _policy_fields(p: Any) -> list:
    return [[name, float(getattr(p, name)).hex()] for name in ("hold_s", "max_depth_change_m", "max_unit_discharge_m2_s", "outlet_discharge_m3_s", "max_excess_depth_m", "excess_volume_m3")]


def _spec_digest(edges: Any, intensity: Any, origin: Any, control: Any, policy: Any, multiplier: Any) -> str:
    """Domain-tagged streamed digest: host schedule buffers hashed in place, canonical control/policy/origin, then the effective field multiplier streamed from its namespace (device: <= `HASH_CHUNK_BYTES` chunks, counted D2H)."""
    h = hashlib.sha256(SPEC_DOMAIN)
    ls._stream_hash(h, edges)       # host buffer for a C-contiguous array, bounded `nditer` staging for a strided one: never a whole-array copy
    ls._stream_hash(h, intensity)
    h.update(json.dumps({"origin": float(origin).hex(), "control": _control_fields(control), "policy": _policy_fields(policy)}, sort_keys=True, allow_nan=False).encode("ascii"))
    ls._stream_hash(h, multiplier)
    return h.hexdigest()


def spec_sha256_of_spec(spec: Any) -> str:
    """Digest of the caller's (already validated) `StormSpec`."""
    return _spec_digest(spec.schedule.edges_s, spec.schedule.intensity_mm_per_h, spec.event_origin_s, spec.control, spec.policy, spec.field.multiplier)


def spec_sha256_of_event(event: Any) -> str:
    """Digest of a sealed prepared event's OWNED forcing copies (trusted same-package read of the event's private slots; no accessor exists or is added)."""
    if type(event) is not ev.LayeredEvent:
        _refuse("event must be a genuine prepared LayeredEvent")
    return _spec_digest(event._schedule.edges_s, event._schedule.intensity_mm_per_h, event.event_origin_s, event.control, event.policy, event._field.multiplier)


def _event_times(origin: Any, edges_s: Any, max_recession_s: Any) -> tuple[float, float]:
    end = float(origin) + float(edges_s[-1])  # the event's own rule `absolute_edges_s[-1]` (the same IEEE float64 addition of the LAST edge only: no schedule-sized vector)
    return end, end + float(max_recession_s)


def storm_binding(spec_sha256: str, origin: Any, rainfall_end_s: float, deadline_s: float) -> dict:
    """The closed manifest `binding.storm` record: spec-derivable fields only (so a loader can rebuild it cold from the caller's spec)."""
    return {"spec_sha256": spec_sha256, "event_origin_s": float(origin).hex(), "rainfall_end_s": float(rainfall_end_s).hex(), "deadline_s": float(deadline_s).hex()}


def storm_binding_of_event(event: Any) -> dict:
    return storm_binding(spec_sha256_of_event(event), event.event_origin_s, event.rainfall_end_s, event.deadline_s)


def validate_storm_spec(spec: Any, lop: Any, device_id: Any) -> None:
    """Exact types only (no iteration/formatting of caller objects), namespace/shape of the borrowed rainfall field and its device residency on the lifecycle's bound device."""
    from maple_syrup.layered_event import LayeredEventControl, LayeredQuietPolicy

    if type(spec) is not lk.StormSpec:
        _refuse("storm must be an exact lifecycle_checkpoint.StormSpec")
    if (type(spec.schedule) is not RainfallSchedule or type(spec.field) is not RainfallField or type(spec.control) is not LayeredEventControl or type(spec.policy) is not LayeredQuietPolicy):
        _refuse("the StormSpec needs genuine RainfallSchedule/RainfallField/LayeredEventControl/LayeredQuietPolicy records")
    o = spec.event_origin_s
    if isinstance(o, (bool, np.bool_)) or not isinstance(o, (int, float, np.integer, np.floating)) or not math.isfinite(float(o)):
        _refuse("the StormSpec event_origin_s must be a finite real number")
    if spec.field.xp is not lop.xp or tuple(spec.field.shape) != tuple(lop.shape):
        _refuse("the StormSpec rainfall field namespace/shape differs from the lifecycle's")
    if not cs._is_device_leaf(spec.field.multiplier, device_id):
        _refuse("the StormSpec rainfall field is not a CuPy array on the lifecycle's bound device")
    for name in ("edges_s", "intensity_mm_per_h"):
        a = getattr(spec.schedule, name)
        if type(a) is not np.ndarray or a.dtype != np.float64 or a.ndim != 1 or a.shape[0] < 1:
            _refuse(f"the StormSpec schedule {name} must be a 1-D float64 host array")
    try:
        spec.control.validated()
        spec.policy.validated()
    except (ev.LayeredEventError, ls.LayeredStormError) as exc:
        raise Err(f"invalid StormSpec control/policy: {exc}") from None


def check_spec_digest_traffic(delta: Any, expected: dict, multiplier: Any) -> None:
    """The spec-digest stage's measured D2H against its OWN named expectation (bytes exact; transfer count exact for a C-contiguous field); separate from every native/bed/companion counter expectation."""
    if int(delta.device_to_host_bytes) != int(expected["bytes"]):
        _refuse(f"spec-digest D2H bytes {int(delta.device_to_host_bytes)} differ from the planned {int(expected['bytes'])} (the f64 field only; no host schedule traffic); nothing proceeds")
    if multiplier.flags.c_contiguous and int(delta.device_to_host) != int(expected["transfers"]):
        _refuse(f"spec-digest D2H transfers {int(delta.device_to_host)} differ from the planned {int(expected['transfers'])} chunks of a C-contiguous field; nothing proceeds")


def expected_storm_binding(spec: Any) -> dict:
    end, dead = _event_times(spec.event_origin_s, spec.schedule.edges_s, spec.control.max_recession_s)
    return storm_binding(spec_sha256_of_spec(spec), spec.event_origin_s, end, dead)


# --- public pure planners -------------------------------------------------------------------------------------------------------------------------
def estimate_save_budget_cuda_v2(meta: Any, *, caller_host_reserve_bytes: int, caller_device_reserve_bytes: int, extra_reserve_bytes: int = 0) -> Any:
    """PURE V2 save budget over sanitized metadata (storm role adds the spec-digest stage and the event audit scratch)."""
    storm = v2plan.StormSaveTerms() if meta.role == "storm" else None
    return v2plan.plan_save_v2(meta, caller_host_reserve_bytes=caller_host_reserve_bytes, caller_device_reserve_bytes=caller_device_reserve_bytes, extra_reserve_bytes=extra_reserve_bytes, storm=storm)


def estimate_load_budget_cuda_v2(meta: Any, *, caller_host_reserve_bytes: int, caller_device_reserve_bytes: int, extra_reserve_bytes: int = 0, storm: Any = None) -> Any:
    """PURE V2 load budget over sanitized metadata (`storm` = `StormLoadTerms` for the storm role, else None)."""
    return v2plan.plan_load_v2(meta, caller_host_reserve_bytes=caller_host_reserve_bytes, caller_device_reserve_bytes=caller_device_reserve_bytes, extra_reserve_bytes=extra_reserve_bytes, storm=storm)


# --- public V2 entry points (thin wrappers over the registered-policy engines) -----------------------------------------------------------------------------
def save_lifecycle_checkpoint_cuda_v2(path: Any, lop: Any, state: Any, identity: dict, *, limits: Any, caller_host_reserve_bytes: int, caller_device_reserve_bytes: int,
                                      host_available_bytes_fn: Any = None, device_available_bytes_fn: Any = None, disk_available_bytes_fn: Any = None,
                                      extra_reserve_bytes: int = 0) -> cs.CudaSaveReceipt:
    """Write a NEW sealed V2 checkpoint directory for the accepted `state` (closed boundary, accepted interevent, or a genuinely COMPLETED storm). Same ceremony/guards as V1; see the module docstring."""
    return cs._save_engine(path, lop, state, identity, limits=limits, caller_host_reserve_bytes=caller_host_reserve_bytes, caller_device_reserve_bytes=caller_device_reserve_bytes,
                           host_available_bytes_fn=host_available_bytes_fn, device_available_bytes_fn=device_available_bytes_fn, disk_available_bytes_fn=disk_available_bytes_fn,
                           extra_reserve_bytes=extra_reserve_bytes, _policy=pol_mod.V2)


def load_lifecycle_checkpoint_cuda_v2(path: Any, lop: Any, expected_identity: dict, *, expected_role: str, expected_source_kind: str | None = None, storm: Any = None, limits: Any,
                                      reference_bed: Any, caller_host_reserve_bytes: int, caller_device_reserve_bytes: int, host_available_bytes_fn: Any = None,
                                      device_available_bytes_fn: Any = None, disk_available_bytes_fn: Any = None, extra_reserve_bytes: int = 0,
                                      expected_physical_digest: Any = None) -> ld.CudaRestoration:
    """Load a sealed V2 checkpoint into a NEW bed/operator/state (UNADOPTED). `storm` (the caller's expected `StormSpec`) is required iff `expected_role == "storm"` and refused otherwise."""
    return ld._load_engine(path, lop, expected_identity, expected_role=expected_role, expected_source_kind=expected_source_kind, limits=limits, reference_bed=reference_bed,
                           caller_host_reserve_bytes=caller_host_reserve_bytes, caller_device_reserve_bytes=caller_device_reserve_bytes, host_available_bytes_fn=host_available_bytes_fn,
                           device_available_bytes_fn=device_available_bytes_fn, disk_available_bytes_fn=disk_available_bytes_fn, extra_reserve_bytes=extra_reserve_bytes,
                           expected_physical_digest=expected_physical_digest, _storm=storm, _policy=pol_mod.V2)
