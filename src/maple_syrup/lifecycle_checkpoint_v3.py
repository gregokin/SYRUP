"""V3 CPU lifecycle checkpoint for PHASE 1.5 DAILY lineages (native9f Dense; NumPy/Numba).

A NEW schema (`SCHEMA_V3`) and ONE explicit profile (`NATIVE9F_PROFILE_V3`) save/load/continue the ONE current role of a daily lineage (`boundary`, `interevent`
(`storm_end` / `boundary_origin`) and `storm` — a genuinely PARTIAL or completed one: accepted/rejected steps, quiet history, next dt cap, forcing index, anchors, cumulative
ledgers and window marks are the existing V2 storm record) and add exactly one closed scalar part, `daily`: the bounded `DailyCursor` (versioned clock contract, run, next
ordinal, the one pending occurrence with its forcing/event identity and the capacity-fallback failing-column count, the last consumed occurrence digests). The T request
sequence, J, modeled matrix/preferential intake, the one authoritative water inventory, the actual native Dense bed and every ledger travel unchanged in the shared engine's
payload; recharge is a distinct external-rain ledger column, never matrix intake.

It is a thin layer over the SAME shared engine as V1/V2 (`lifecycle_checkpoint`: fences, admission before any I/O/cold source/hash/array, fresh source audit, descriptor-bound
staging, no-clobber publication, native snapshot census, owner restoration) under a registered literal private policy (`_V3_9F`); nothing public accepts a policy, schema,
digest or profile table. V1/V2 writers refuse a state WITH a cursor before any I/O; this codec refuses a state WITHOUT one; the three schemas/profiles never load each other's
directories; there is no migration, cursor stripping or cursor re-attachment to a synthetic twin (the cursor is supplied at state CONSTRUCTION and the recomputed digest, which
includes it, must equal the saved one).

HOW A CALLER RESUMES. The caller rebuilds the FIXED components exactly as the original run (the genuine file/NCP factories) and supplies: the expected identity dict, the
expected role (and for an interevent its closed `expected_source_kind`), the reference bed, and — for a storm — the `StormSpec` of the saved storm (reconstruct it from the
ORIGINAL canonical plan: `forcing = plan.bind_forcing(origin_s)`, schedule `forcing.schedule` (LOCAL `[0, D]`), `event_origin_s = origin_s` = the parent clock at the storm start,
a rainfall field from the active mask, and the same control/quiet policy). The loader REQUIRES that spec to reproduce the saved forcing digest and event identity, and, when
`expected_plan` is given, that plan to equal the saved plan identity/ordinal/run (never inferring a threshold from soil or file content). After a boundary: next plan =
`daily_water.next_plan(state, ...)`. After a completed storm: `finish_daily_storm` (ONE day). After an accepted day before T: `request_withdrawal` continues the saved request
sequence; after T: `close_interevent`. A partial storm continues through the existing bounded `evolve_storm` and is never completed by the loader.
"""

from __future__ import annotations

import dataclasses
import hashlib
from dataclasses import dataclass
from typing import Any, Callable

from maple_syrup import layered_checkpoint as lc
from maple_syrup import lifecycle_checkpoint as lk
from maple_syrup.lifecycle_checkpoint import (LifecycleBudget, LifecycleCheckpointError, LifecycleCheckpointQuotaError, LifecycleRestoration,
                                              LifecycleSaveReceipt)

__all__ = ["LifecycleRestorationV3", "LifecycleSaveReceiptV3", "NATIVE9F_PROFILE_V3", "ROLE_SOURCE_PAIRS", "SCHEMA_V3", "SOURCE_KINDS", "estimate_lifecycle_budget_v3",
           "load_lifecycle_checkpoint_v3", "physical_digest_v3", "save_lifecycle_checkpoint_v3"]

SCHEMA_V3 = lk.SCHEMA_V3
NATIVE9F_PROFILE_V3 = lk.NATIVE9F_PROFILE_V3
SOURCE_KINDS = lk.SOURCE_KINDS_V2
ROLE_SOURCE_PAIRS = (("boundary", None), ("storm", None), ("interevent", "storm_end"), ("interevent", "boundary_origin"))
_PROFILES = (NATIVE9F_PROFILE_V3,)
_POLICY = lk._V3_9F


@dataclass(frozen=True)
class LifecycleSaveReceiptV3(LifecycleSaveReceipt):
    source_kind: str | None = None
    schema: str = SCHEMA_V3
    daily_cursor_digest: str | None = None


@dataclass(frozen=True, eq=False)
class LifecycleRestorationV3(LifecycleRestoration):
    source_kind: str | None = None


def _profile(profile: Any) -> str:
    if type(profile) is not str or profile not in _PROFILES:
        raise LifecycleCheckpointError(f"profile {profile!r} is not the qualified v3 profile {_PROFILES}")
    return profile


def estimate_lifecycle_budget_v3(lop: Any, role: str, *, profile: str = NATIVE9F_PROFILE_V3, source_kind: str | None = None, bed: Any = None,
                                 event_extra_bytes: int = 0, extra_reserve_bytes: int = 0) -> LifecycleBudget:
    """The pure metadata-only whole-call model (V2 terms + the derived `daily` manifest allowance); no array, file or source read."""
    prof = _profile(profile)
    got = lk._fence(lop, bed, _POLICY)
    if got != prof:
        raise LifecycleCheckpointError(f"profile {prof!r} does not match this lifecycle ({got!r})")
    return lk._estimate_budget(lop, role, bed=bed, event_extra_bytes=event_extra_bytes, extra_reserve_bytes=extra_reserve_bytes, _policy=_POLICY, source_kind=source_kind)


def save_lifecycle_checkpoint_v3(path: Any, lop: Any, state: Any, identity: dict, *, limits: Any, host_available_bytes_fn: Callable[[], int],
                                 device_available_bytes_fn: Any = None, disk_available_bytes_fn: Callable[[], int] | None = None,
                                 extra_reserve_bytes: int = 0, profile: str = NATIVE9F_PROFILE_V3) -> LifecycleSaveReceiptV3:
    """Write a NEW sealed V3 directory for the accepted DAILY `state` (one current role). Typed refusal and nothing sealed on any failure; a state without a cursor
    is refused before any I/O; inputs/accepted state/foreign files are never touched."""
    prof = _profile(profile)
    try:
        rec = lk._save(path, lop, state, identity, limits=limits, host_available_bytes_fn=host_available_bytes_fn, device_available_bytes_fn=device_available_bytes_fn,
                       disk_available_bytes_fn=disk_available_bytes_fn, extra_reserve_bytes=extra_reserve_bytes, _policy=_POLICY, _profile=prof)
    except LifecycleCheckpointError:
        raise
    except lc.LayeredCheckpointQuotaError as exc:
        raise LifecycleCheckpointQuotaError(str(exc)) from None
    except lc.LayeredCheckpointError as exc:
        raise LifecycleCheckpointError(str(exc)) from None
    fields = {f.name: getattr(rec, f.name) for f in dataclasses.fields(rec)}
    return LifecycleSaveReceiptV3(**fields, source_kind=lk._state_source_kind(_POLICY, state), daily_cursor_digest=state.daily.digest())


def load_lifecycle_checkpoint_v3(path: Any, lop: Any, expected_identity: dict, *, expected_role: str, limits: Any, host_available_bytes_fn: Callable[[], int],
                                 reference_bed: Any, storm: Any = None, expected_source_kind: str | None = None, expected_plan: Any = None,
                                 device_available_bytes_fn: Any = None, disk_available_bytes_fn: Callable[[], int] | None = None, extra_reserve_bytes: int = 0,
                                 profile: str = NATIVE9F_PROFILE_V3) -> LifecycleRestorationV3:
    """Restore a sealed V3 directory against the caller's EXPECTED fixed components, role, identity, closed source kind (interevent), reference bed, `StormSpec` (storm) and
    optionally the ORIGINAL canonical plan. Returns a NEW lifecycle operator prepared on the LOADED bed and a state of the daily lineage bound to it; nothing is adopted. Every
    refusal leaves all inputs, the caller's operator and the files unchanged."""
    prof = _profile(profile)
    lk._validate_expected_plan(expected_plan)  # first thing: before any fence, budget, source audit or I/O
    out = lk._load_path(path, lop, expected_identity, expected_role=expected_role, limits=limits, host_available_bytes_fn=host_available_bytes_fn, reference_bed=reference_bed,
                        storm=storm, device_available_bytes_fn=device_available_bytes_fn, disk_available_bytes_fn=disk_available_bytes_fn,
                        extra_reserve_bytes=extra_reserve_bytes, _policy=_POLICY, _profile=prof, expected_source_kind=expected_source_kind, expected_plan=expected_plan)
    receipt = dict(out.receipt, schema=SCHEMA_V3, source_kind=expected_source_kind)
    return LifecycleRestorationV3(out.lop, out.state, receipt, expected_source_kind)


def physical_digest_v3(lop: Any, state: Any) -> str:
    """V1's streamed physical digest (which already covers the scalar state digest, i.e. the cursor) bound to the V3 domain, the current source kind and the cursor digest."""
    h = hashlib.sha256(b"maple_syrup.lifecycle_checkpoint_v3.physical/1")
    h.update(lk.physical_digest(lop, state).encode())
    kind = lk._state_source_kind(_POLICY, state)
    h.update(repr((kind, None if kind is None else state.payload.source.digest, state.daily.digest())).encode())
    return h.hexdigest()
