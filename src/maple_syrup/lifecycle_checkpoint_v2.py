"""Versioned (v2) CPU lifecycle checkpoint: adds the genuine BOUNDARY-ORIGIN interevent source (task
phase1_boundary_origin_checkpoint_impl_20261110, Stage A).

A NEW schema (`SCHEMA_V2`) and three explicit profiles (`NATIVE39_PROFILE_V2`, `BEDLESS_PROFILE_V2`, `NATIVE9F_PROFILE_V2`) save/load/continue ONE current role of
`water_lifecycle` exactly like v1 (`boundary`, `storm`, `interevent`) and record, for an interevent, the closed `source_kind` of its CURRENT
source: `storm_end` (a genuine `StormEndReceipt`) or `boundary_origin` (a genuine `BoundaryOriginReceipt`: an interevent that began at an
initialized or previously closed boundary, no past storm). It adds no physical law.

This module is a thin public layer over the shared engine in `lifecycle_checkpoint` (one implementation of the fences, admission, fresh source
audit, descriptor-bound reservation/staging/no-clobber publication, stable-copy load, native validators and owner restoration). The engine is
driven by a sealed, immutable, registered private policy; nothing public accepts a policy, digest, schema or profile table. v1 is unchanged:
its public API, schema, profiles, literal MAPLE39 pin, vocabulary and its typed refusal of a boundary-origin window stay as they were, and the
two versions never load each other's archives. The Native9f profile (Stage B) is a third registered literal policy with its own MAPLE 9f package pin, Dense-only (no bedless alias) and an exact
ordinary snapshot return-key census; an unknown profile string is refused before any I/O.

Trust boundary (as v1): digests and recomputed owner digests demonstrate source consistency/integrity of THIS payload; they are not signatures and
cannot re-prove that a past entry/step physically ran. See docs/subsurface_hydrology/lifecycle_checkpoint_v2.md.
"""

from __future__ import annotations

import dataclasses
import hashlib
from dataclasses import dataclass
from typing import Any, Callable

from maple_syrup import layered_checkpoint as lc
from maple_syrup import lifecycle_checkpoint as lk
from maple_syrup.lifecycle_checkpoint import (LifecycleBudget, LifecycleCheckpointError, LifecycleCheckpointIndeterminate,
                                              LifecycleCheckpointQuotaError, LifecycleRestoration, LifecycleSaveReceipt, StormSpec)

__all__ = [
    "BEDLESS_PROFILE_V2", "LifecycleRestorationV2", "LifecycleSaveReceiptV2", "NATIVE39_PROFILE_V2", "NATIVE9F_PROFILE_V2", "ROLE_SOURCE_PAIRS",
    "SCHEMA_V2",
    "SOURCE_KINDS", "estimate_lifecycle_budget_v2", "load_lifecycle_checkpoint_v2", "physical_digest_v2", "save_lifecycle_checkpoint_v2",
]

SCHEMA_V2 = lk.SCHEMA_V2
NATIVE39_PROFILE_V2 = lk.NATIVE_PROFILE_V2
BEDLESS_PROFILE_V2 = lk.BEDLESS_PROFILE_V2
NATIVE9F_PROFILE_V2 = lk.NATIVE9F_PROFILE_V2
SOURCE_KINDS = lk.SOURCE_KINDS_V2
#: The closed (current role, current interevent source kind) pairs; `None` for a role that has no current interevent source.
ROLE_SOURCE_PAIRS = (("boundary", None), ("storm", None), ("interevent", "storm_end"), ("interevent", "boundary_origin"))
_PROFILES = (NATIVE39_PROFILE_V2, BEDLESS_PROFILE_V2, NATIVE9F_PROFILE_V2)
#: Closed explicit profile string -> registered literal policy (no caller-supplied policy, schema, pin or validator).
_POLICY_FOR = {NATIVE39_PROFILE_V2: lk._V2, BEDLESS_PROFILE_V2: lk._V2, NATIVE9F_PROFILE_V2: lk._V2_9F}


@dataclass(frozen=True)
class LifecycleSaveReceiptV2(LifecycleSaveReceipt):
    source_kind: str | None = None
    schema: str = SCHEMA_V2


@dataclass(frozen=True, eq=False)
class LifecycleRestorationV2(LifecycleRestoration):
    source_kind: str | None = None


def _profile(profile: Any) -> tuple[str, Any]:
    """Closed explicit profile string -> (string, its registered literal policy), validated before any I/O. Unknown strings are refused."""
    if type(profile) is not str or profile not in _PROFILES:
        raise LifecycleCheckpointError(f"profile {profile!r} is not a qualified v2 profile {_PROFILES}")
    return profile, _POLICY_FOR[profile]


def estimate_lifecycle_budget_v2(lop: Any, role: str, *, profile: str, source_kind: str | None = None, bed: Any = None,
                                 event_extra_bytes: int = 0, extra_reserve_bytes: int = 0) -> LifecycleBudget:
    """v1's pure metadata-only whole-call model plus the named v2 terms (`source_envelope`, `quiet_scratch`, derived manifest allowance).
    An interevent needs its closed `source_kind`; every other role must give None."""
    prof, pol = _profile(profile)
    got = lk._fence(lop, bed, pol)
    if got != prof:
        raise LifecycleCheckpointError(f"profile {prof!r} does not match this lifecycle ({got!r})")
    return lk._estimate_budget(lop, role, bed=bed, event_extra_bytes=event_extra_bytes, extra_reserve_bytes=extra_reserve_bytes,
                               _policy=pol, source_kind=source_kind)


def save_lifecycle_checkpoint_v2(path: Any, lop: Any, state: Any, identity: dict, *, profile: str, limits: Any,
                                 host_available_bytes_fn: Callable[[], int], device_available_bytes_fn: Any = None,
                                 disk_available_bytes_fn: Callable[[], int] | None = None,
                                 extra_reserve_bytes: int = 0) -> LifecycleSaveReceiptV2:
    """Write a NEW sealed v2 checkpoint directory for the accepted `state` (one current role, its current source kind recorded). Same
    error contract as v1: typed refusal, nothing sealed, inputs/accepted state/foreign files unchanged."""
    prof, pol = _profile(profile)
    try:
        rec = lk._save(path, lop, state, identity, limits=limits, host_available_bytes_fn=host_available_bytes_fn,
                       device_available_bytes_fn=device_available_bytes_fn, disk_available_bytes_fn=disk_available_bytes_fn,
                       extra_reserve_bytes=extra_reserve_bytes, _policy=pol, _profile=prof)
    except LifecycleCheckpointError:
        raise
    except lc.LayeredCheckpointQuotaError as exc:
        raise LifecycleCheckpointQuotaError(str(exc)) from None
    except lc.LayeredCheckpointError as exc:
        raise LifecycleCheckpointError(str(exc)) from None
    kind = lk._state_source_kind(pol, state)
    fields = {f.name: getattr(rec, f.name) for f in dataclasses.fields(rec)}
    return LifecycleSaveReceiptV2(**fields, source_kind=kind)


def load_lifecycle_checkpoint_v2(path: Any, lop: Any, expected_identity: dict, *, profile: str, expected_role: str,
                                 expected_source_kind: str | None = None, limits: Any,
                                 host_available_bytes_fn: Callable[[], int], reference_bed: Any = None, storm: Any = None,
                                 device_available_bytes_fn: Any = None, disk_available_bytes_fn: Callable[[], int] | None = None,
                                 extra_reserve_bytes: int = 0) -> LifecycleRestorationV2:
    """Restore a sealed v2 checkpoint against the caller's EXPECTED fixed components, role, identity and (interevent only) closed source
    kind. Returns a NEW lifecycle operator prepared on the LOADED bed and a state bound to it; nothing is adopted. Every refusal leaves all
    inputs, the caller's operator and the files unchanged. Goes through the engine's full admitted/stable-descriptor load path."""
    prof, pol = _profile(profile)
    out = lk._load_path(path, lop, expected_identity, expected_role=expected_role, limits=limits,
                        host_available_bytes_fn=host_available_bytes_fn, reference_bed=reference_bed, storm=storm,
                        device_available_bytes_fn=device_available_bytes_fn, disk_available_bytes_fn=disk_available_bytes_fn,
                        extra_reserve_bytes=extra_reserve_bytes, _policy=pol, _profile=prof,
                        expected_source_kind=expected_source_kind)
    receipt = dict(out.receipt, schema=SCHEMA_V2, source_kind=expected_source_kind)
    return LifecycleRestorationV2(out.lop, out.state, receipt, expected_source_kind)


def physical_digest_v2(lop: Any, state: Any) -> str:
    """v1's streamed physical digest bound to the current source kind and source digest (interevent), for receipts/tests."""
    h = hashlib.sha256(b"maple_syrup.lifecycle_checkpoint_v2.physical/1")
    h.update(lk.physical_digest(lop, state).encode())
    kind = lk._state_source_kind(lk._V2, state)
    h.update(repr((kind, None if kind is None else state.payload.source.digest)).encode())
    return h.hexdigest()
