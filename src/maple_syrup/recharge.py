"""Phase 1.5 daily small-rain selection, full-storm forcing and bounded occurrence cursor (S1: pure host records).

SYRUP owns water; a future scheduler owns the nominal calendar. This module holds no water, no arrays and no clock: it only

  * validates a daily rainfall descriptor `P` (mm), `I` (mm/h) and the user-selected amount threshold `T` (mm) BEFORE anything else
    (finite real numbers, no Boolean/string; `P >= 0`, `T >= 0`; `I > 0` whenever `P > 0`; no default threshold and no extra
    intensity cutoff exists),
  * selects the branch exactly as specified: `P == 0` -> `no_rain`; `0 < P <= T` -> `recharge` (equality attempts recharge: user decision
    2026-11-12 UTC); `P > T` -> `storm`,
  * derives the FULL native storm forcing for EVERY positive `P` (also for the `recharge` selection, because a capacity failure falls
    back to the full storm from the unchanged parent with the FULL `P`): duration `3600 P / I` seconds is validated to be finite,
    strictly positive and to reproduce `P` (an underflowing duration or depth would silently lose positive rain and is refused);
    `bind_forcing(t0)` additionally checks that `t0 + duration` is representable and builds the schedule with the actual
    `rainfall.constant_rainfall` API,
  * defines the bounded scalar `DailyCursor` (run digest, ordinal, at most one pending storm occurrence, last digests) that a later stage
    attaches to the lifecycle; there is no growing set or history.

Nothing here selects a threshold, changes a clock or applies water. See `subsurface.step_with_recharge` for the one pulse-aware daily
column operation and the task design for the integration (later stages).
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, fields
from typing import Any

import numpy as np

from maple_syrup.conservation import volume_roundoff_bound_m3
from maple_syrup.rainfall import RainfallError, RainfallSchedule, constant_rainfall

__all__ = [
    "BRANCHES",
    "CLOCK_CONTRACT",
    "MAX_RECORD_COUNT",
    "DAY_S",
    "SELECTIONS",
    "BoundForcing",
    "DailyCursor",
    "DailyRainPlan",
    "PendingStorm",
    "RechargeError",
    "cursor_from_record",
    "cursor_record",
    "plan_daily_rain",
    "verify_forcing",
    "verify_plan",
]

SELECTIONS = ("no_rain", "recharge", "storm")
BRANCHES = ("no_rain", "recharge", "storm", "storm_fallback")
SECONDS_PER_HOUR = 3600.0
#: The fixed native day of a Phase 1.5 daily lineage (physical seconds) and the explicit, versioned clock contract that every cursor digest
#: and daily receipt carries. The NOMINAL occurrence ordinal is separate from physical seconds: a no-rain/recharge occurrence elapses exactly
#: one day; a storm occurrence elapses the genuine storm (rain + recession) duration PLUS one day. Historical profiles never carry this literal.
DAY_S = 86400.0
CLOCK_CONTRACT = "maple_syrup.daily_clock/1: fixed 86400 s native day; nominal ordinal separate from physical seconds; storm occurrence = actual storm + one day"
# Representability bounds are NOT new epsilons: they are MAPLE's own FP64 summation/operation coefficient
# (`conservation.volume_roundoff_bound_m3(n, scale)` = 4 n eps64 scale, the shared `summation_error_bound_kg` method) with the ACTUAL number of
# FP64 operations of each derivation, applied at the rain amount P as the scale. Operation counts:
_OPS_DEPTH = 2  # depth = P / 1000 ; depth * 1000
_OPS_DURATION = 4  # 3600 * P ; / I ; then the check re-derives * I ; / 3600
_OPS_NATIVE = 6  # duration (2) + rate = I * (1 / 3.6e6) (2: the constant's own rounding and the product) + rate * realized * 1000 (2)
_OPS_CLOCK = 2  # end = t0 + duration ; realized = end - t0: two operations whose operand scale is the clock value `end`, in seconds
_SHA = frozenset("0123456789abcdef")


class RechargeError(ValueError):
    """Invalid daily rain descriptor, forcing, cursor or occurrence binding. Raised before any state exists."""


def _real(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise RechargeError(f"{name} must be a real host number, got {type(value).__name__}")
    out = float(value)
    if not math.isfinite(out):
        raise RechargeError(f"{name} must be finite, got {value!r}")
    return out


def _hex64(value: Any, name: str) -> str:
    if type(value) is not str or len(value) != 64 or not set(value) <= _SHA:
        raise RechargeError(f"{name} must be a lowercase 64-hex SHA-256 string")
    return value


#: The declared interface bound of EVERY counter of the daily records (cursor ordinals, pending ordinal, capacity failing-column count): the signed 64-bit maximum. It is
#: exactly the value the V3 manifest allowance (`lifecycle_checkpoint._derive_daily_allowance`) is derived for, and it is enforced on construction (hence on encode AND on
#: decode, which constructs): a larger value is a typed `RechargeError` before any record exists.
MAX_RECORD_COUNT = 2 ** 63 - 1


def _count(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or int(value) < 0:
        raise RechargeError(f"{name} must be a non-Boolean integer >= 0, got {value!r}")
    if int(value) > MAX_RECORD_COUNT:
        raise RechargeError(f"{name} exceeds the declared record bound 2**63 - 1 (the V3 daily record allowance is derived for it), got {int(value)}")
    return int(value)


def _digest(tag: str, body: Any) -> str:
    return hashlib.sha256((tag + "\n" + json.dumps(body, sort_keys=True, separators=(",", ":"))).encode("ascii")).hexdigest()


def _amount_ok(value: float, p: float, n_ops: int) -> bool:
    """`value` reproduces the rain amount `p` within MAPLE's FP64 operation bound for `n_ops` operations at scale `p`. Every operand is
    validated finite FIRST (a non-finite pair is never 'close')."""
    if not (math.isfinite(value) and math.isfinite(p) and p > 0.0 and value > 0.0):
        return False
    return abs(value - p) <= volume_roundoff_bound_m3(n_ops, p)


@dataclass(frozen=True)
class BoundForcing:
    """The FULL native storm forcing of one positive-rain plan at an actual physical start. TIME CONTRACT (native, unchanged): the genuine
    `RainfallSchedule` (`rainfall.constant_rainfall`) is LOCAL to the event, edges `[0, D]` with `D = 3600 P / I`, because the native
    `LayeredEvent` adds `event_origin_s` to every schedule edge (`absolute_edges_s = event_origin_s + edges`). The PHYSICAL (absolute) start and end are
    explicit scalars: `t0_s` (= the event origin = the accepted parent's clock) and `end_s = t0_s + D`; the native event's absolute edges must equal
    `[t0_s, end_s]`. `digest` binds the physical start/end AND the local schedule. Used by the `storm` branch AND the capacity fallback of the
    `recharge` branch (always the FULL `P`)."""

    plan_digest: str
    t0_s: float  # physical start = native event origin
    end_s: float  # physical end = t0_s + duration_s (the absolute rainfall end of the native event)
    duration_s: float
    intensity_mm_h: float
    p_mm: float
    schedule: RainfallSchedule  # LOCAL edges [0, duration_s]
    digest: str  # physical t0/end + the local schedule digest
    amount_mm: float  # what the native event integrates over [t0, end]: rate * ((t0 + D) - t0) * 1000
    amount_bound_mm: float  # the declared MAPLE FP64 operation bound |amount_mm - p_mm| had to satisfy (recorded; never silently exceeded)


@dataclass(frozen=True)
class DailyRainPlan:
    """Validated, frozen, hashed daily rainfall descriptor and branch selection. `label` (e.g. a date) is informational only;
    occurrence uniqueness is `(run_digest, ordinal)` when supplied."""

    p_mm: float
    i_mm_h: float
    t_mm: float
    selection: str
    reason: str
    depth_m: float
    duration_s: float | None
    run_digest: str | None
    ordinal: int | None
    label: str | None
    digest: str

    def bind_forcing(self, t0_s: Any) -> BoundForcing | None:
        """Full storm forcing at the actual start `t0_s` for EVERY positive `P` (None for `no_rain`). The plan is re-derived and compared
        field by field first (a replaced/forged record is refused). The end `t0 + duration` must be representable and the NATIVE storm
        (the schedule's own endpoints and rate, exactly what the storm integrates) must reproduce `P` within MAPLE's FP64 operation bound
        for the counted derivation; otherwise the clock cannot hold this rain at this time and the plan is refused. Rain is never
        shortened, lengthened, re-stamped or lost by clock rounding."""
        verify_plan(self)
        if self.selection == "no_rain":
            return None
        t0 = _real(t0_s, "t0_s")
        if t0 < 0.0:
            raise RechargeError(f"t0_s must be >= 0 (seconds from the event origin), got {t0!r}")
        end = t0 + self.duration_s
        if not math.isfinite(end) or not end > t0:
            raise RechargeError(f"the storm end {t0!r} + {self.duration_s!r} s is not representable on the clock (refused; no rain is dropped)")
        try:
            schedule = constant_rainfall(0.0, self.duration_s, self.i_mm_h)  # LOCAL [0, D]: the native event adds its origin t0 (never twice)
        except RainfallError as exc:
            raise RechargeError(f"the actual rainfall API refused the full-storm forcing: {exc}") from None
        realized = float(end) - float(t0)  # what the native absolute edges [t0 + 0, t0 + D] span
        rate = float(schedule.rate_m_per_s[0])
        native_mm = rate * realized * 1000.0
        # declared bound = MAPLE's FP64 operation bound (volume_roundoff_bound_m3 = 4 n eps scale): the arithmetic of the derivation at the rain
        # amount's scale (n = _OPS_NATIVE) PLUS the two clock operations at the CLOCK's scale `end` (n = _OPS_CLOCK; seconds converted to mm
        # of rain by I / 3600). The clock term is absolute: a long clock cannot resolve a short storm, so a storm whose own duration does
        # not exceed the clock bound is refused, and anything else carries its realized amount and bound in the record.
        clock_s = volume_roundoff_bound_m3(_OPS_CLOCK, end)
        bound_mm = volume_roundoff_bound_m3(_OPS_NATIVE, self.p_mm) + self.i_mm_h / SECONDS_PER_HOUR * clock_s
        if not (math.isfinite(realized) and math.isfinite(rate) and math.isfinite(native_mm) and math.isfinite(bound_mm)
                and realized > 0.0 and rate > 0.0 and clock_s < self.duration_s and abs(native_mm - self.p_mm) <= bound_mm):
            raise RechargeError(
                f"the native storm [{t0!r}, {end!r}] s at {self.i_mm_h!r} mm/h carries {native_mm!r} mm, not P = {self.p_mm!r} mm within the "
                f"declared MAPLE FP64 operation bound {bound_mm!r} mm (clock resolution {clock_s!r} s, underflow or overflow; refused, never re-stamped)")
        return BoundForcing(self.digest, t0, float(end), float(self.duration_s), self.i_mm_h, self.p_mm, schedule,
                            _forcing_digest(schedule, t0, float(end)), native_mm, float(bound_mm))


def _forcing_digest(schedule: RainfallSchedule, t0: float, end: float) -> str:
    return _digest("maple_syrup.recharge.forcing/2", {"t0": float(t0).hex(), "end": float(end).hex(), "schedule": _schedule_digest(schedule)})


def _schedule_digest(schedule: RainfallSchedule) -> str:
    return _digest("maple_syrup.recharge.schedule/1", {"edges": [float(e).hex() for e in schedule.edges_s],
                                                       "intensity": [float(v).hex() for v in schedule.intensity_mm_per_h]})


def _same(a: Any, b: Any) -> bool:
    return type(a) is type(b) and a == b


def verify_plan(plan: Any) -> None:
    """Re-derive EVERY canonical field (selection, units, depth, duration, digest) from the finite descriptor and run/ordinal/label and
    require exact equality: frozen records built directly or by `dataclasses.replace` with valid-looking but altered fields are refused.
    No digest is trusted as an assertion."""
    if type(plan) is not DailyRainPlan:
        raise RechargeError("plan must be a DailyRainPlan")
    canonical = plan_daily_rain(plan.p_mm, plan.i_mm_h, plan.t_mm, run_digest=plan.run_digest, ordinal=plan.ordinal, label=plan.label)
    for f in fields(DailyRainPlan):
        if not _same(getattr(plan, f.name), getattr(canonical, f.name)):
            raise RechargeError(f"the plan field {f.name!r} is not the canonical value derived from its descriptor (altered or forged plan)")


def verify_forcing(plan: Any, forcing: Any) -> None:
    """The forcing is exactly `plan.bind_forcing(forcing.t0_s)`: scalar fields, the actual native schedule arrays/provenance and the
    schedule digest are recomputed and compared; a replaced schedule (other intensity/edges) is refused even if the digests are kept."""
    verify_plan(plan)
    if type(forcing) is not BoundForcing:
        raise RechargeError("forcing must be a BoundForcing")
    if type(forcing.schedule) is not RainfallSchedule:
        raise RechargeError("forcing.schedule must be a RainfallSchedule")
    expected = plan.bind_forcing(forcing.t0_s)
    if expected is None:
        raise RechargeError("a no_rain plan has no forcing")
    for f in fields(BoundForcing):
        if f.name == "schedule":
            continue
        if not _same(getattr(forcing, f.name), getattr(expected, f.name)):
            raise RechargeError(f"the forcing field {f.name!r} is not derived from this plan at its start (altered or substituted forcing)")
    a, b = forcing.schedule, expected.schedule
    if not (np.array_equal(a.edges_s, b.edges_s) and np.array_equal(a.intensity_mm_per_h, b.intensity_mm_per_h)
            and np.array_equal(a.rate_m_per_s, b.rate_m_per_s) and a.provenance == b.provenance):
        raise RechargeError("the native rainfall schedule is not the one derived from this plan (altered schedule)")
    if _forcing_digest(a, forcing.t0_s, forcing.end_s) != forcing.digest:
        raise RechargeError("the forcing digest does not describe its physical start/end and local schedule")


def plan_daily_rain(p_mm: Any, i_mm_h: Any, t_mm: Any, *, run_digest: Any = None, ordinal: Any = None, label: Any = None) -> DailyRainPlan:
    """Validate everything first, then select the branch. `T` is the caller's threshold (no default); equality `P == T > 0` selects recharge.
    Positive `P` requires finite `I > 0`, a finite strictly positive `3600 P / I` and a positive depth `P / 1000` that reproduces `P`."""
    p, i, t = _real(p_mm, "p_mm"), _real(i_mm_h, "i_mm_h"), _real(t_mm, "t_mm")
    if p < 0.0:
        raise RechargeError(f"p_mm must be >= 0, got {p!r}")
    if t < 0.0:
        raise RechargeError(f"t_mm (the user-selected amount threshold) must be >= 0, got {t!r}")
    if i < 0.0:
        raise RechargeError(f"i_mm_h must be >= 0, got {i!r}")
    run = None if run_digest is None else _hex64(run_digest, "run_digest")
    ordn = None if ordinal is None else _count(ordinal, "ordinal")
    if (run is None) != (ordn is None):
        raise RechargeError("run_digest and ordinal are supplied together or not at all")
    if label is not None and (type(label) is not str or not label or len(label) > 128 or any(ord(c) < 0x20 for c in label)):
        raise RechargeError("label must be a short printable string")
    duration = None
    depth = p / 1000.0
    if p > 0.0:
        if not i > 0.0:
            raise RechargeError("i_mm_h must be > 0 whenever p_mm > 0 (no storm duration exists otherwise)")
        duration = SECONDS_PER_HOUR * p / i
        if not math.isfinite(duration) or not duration > 0.0:
            raise RechargeError(f"the full-storm duration 3600 P / I = {duration!r} s is not finite and positive (underflow/overflow refused)")
        if not _amount_ok(depth * 1000.0, p, _OPS_DEPTH):
            raise RechargeError("the rain depth P / 1000 is not representable (underflow refused; positive rain is never dropped)")
        if not _amount_ok(duration * i / SECONDS_PER_HOUR, p, _OPS_DURATION):
            raise RechargeError("3600 P / I does not reproduce P within the MAPLE FP64 operation bound (underflow refused; positive rain is never dropped)")
        if p <= t:
            selection, reason = "recharge", "0 < P <= T: simplified recharge attempted (equality attempts recharge by user decision)"
        else:
            selection, reason = "storm", "P > T: full native storm"
    else:
        selection, reason = "no_rain", "P = 0: no precipitation operation"
    body = {"p": p.hex(), "i": i.hex(), "t": t.hex(), "selection": selection, "run": run, "ordinal": ordn, "label": label}
    return DailyRainPlan(p, i, t, selection, reason, depth, duration, run, ordn, label, _digest("maple_syrup.recharge.plan/1", body))


@dataclass(frozen=True)
class PendingStorm:
    """The one storm occurrence in progress (storm branch or capacity fallback): bound to its plan and forcing digests."""

    ordinal: int
    plan_digest: str
    branch: str
    forcing_digest: str
    event_sha256: str | None = None  # the genuine prepared storm event identity (set by the lifecycle's daily storm start)
    capacity_failing_columns: int | None = None  # fallback only: how many ACTIVE columns the recharge capacity test rejected (bounded scalar reason)

    def __post_init__(self) -> None:
        _count(self.ordinal, "pending.ordinal")
        _hex64(self.plan_digest, "pending.plan_digest")
        _hex64(self.forcing_digest, "pending.forcing_digest")
        if self.event_sha256 is not None:
            _hex64(self.event_sha256, "pending.event_sha256")
        if self.capacity_failing_columns is not None:
            if self.branch != "storm_fallback" or _count(self.capacity_failing_columns, "pending.capacity_failing_columns") < 1:
                raise RechargeError("a capacity reason (>= 1 failing active column) exists only for the storm_fallback branch")
        if self.branch not in ("storm", "storm_fallback"):
            raise RechargeError("pending.branch must be 'storm' or 'storm_fallback'")


@dataclass(frozen=True)
class DailyCursor:
    """Bounded occurrence cursor (scalars only, never a set or a history). `ordinal` is the NEXT occurrence to consume. Every branch
    (no_rain, recharge, storm, storm_fallback) consumes exactly one ordinal; `pending` exists only between a storm start and its
    genuine completion. All transitions are pure and return a new cursor or raise."""

    run_digest: str
    ordinal: int = 0
    pending: PendingStorm | None = None
    last_plan_digest: str | None = None
    last_branch: str | None = None
    last_parent_digest: str | None = None
    clock_contract: str = CLOCK_CONTRACT

    def __post_init__(self) -> None:
        if self.clock_contract != CLOCK_CONTRACT or type(self.clock_contract) is not str:
            raise RechargeError("the cursor's clock contract is not this module's versioned daily clock contract")
        _hex64(self.run_digest, "run_digest")
        _count(self.ordinal, "ordinal")
        if self.pending is not None and type(self.pending) is not PendingStorm:
            raise RechargeError("pending must be a PendingStorm")
        for name in ("last_plan_digest", "last_parent_digest"):
            if getattr(self, name) is not None:
                _hex64(getattr(self, name), name)
        if self.last_branch is not None and self.last_branch not in BRANCHES:
            raise RechargeError("last_branch is not a known branch")
        last = (self.last_plan_digest, self.last_branch, self.last_parent_digest)
        if any(v is not None for v in last) != all(v is not None for v in last):
            raise RechargeError("last_plan_digest, last_branch and last_parent_digest are all present or all absent")
        if (self.ordinal > 0) != (self.last_branch is not None):
            raise RechargeError("a cursor with consumed occurrences (ordinal > 0) records its last occurrence, and ordinal 0 records none")
        if self.pending is not None and self.pending.ordinal != self.ordinal:
            raise RechargeError("the pending storm ordinal must be the next cursor ordinal")

    def check(self, plan: DailyRainPlan) -> None:
        """The (re-derived, genuine) plan belongs to THIS run and is exactly the next occurrence (no gap, no repeat) and nothing is pending."""
        verify_plan(plan)
        self._check_occurrence(plan)
        if self.pending is not None:
            raise RechargeError("a storm occurrence is pending; it must complete before another occurrence")

    def _check_occurrence(self, plan: DailyRainPlan) -> None:
        if plan.run_digest != self.run_digest:
            raise RechargeError("the plan belongs to another run")
        if plan.ordinal is None or plan.ordinal != self.ordinal:
            raise RechargeError(f"occurrence ordinal {plan.ordinal!r} is not the next one ({self.ordinal}): gap, repeat or stale plan refused")

    def check_for_consume(self, plan: DailyRainPlan, branch: str) -> None:
        """`no_rain`/`recharge` consume only their own plan with nothing pending; `storm`/`storm_fallback` only the matching pending
        occurrence (ordinal, plan digest, branch) of a storm-selected / recharge-selected plan respectively."""
        verify_plan(plan)
        if branch not in BRANCHES:
            raise RechargeError("unknown branch")
        self._check_occurrence(plan)
        if self.pending is None:
            if branch not in ("no_rain", "recharge") or branch != plan.selection:
                raise RechargeError(f"branch {branch!r} cannot be consumed for a {plan.selection!r} plan without a pending storm occurrence")
            return
        want = "storm_fallback" if plan.selection == "recharge" else plan.selection
        if (plan.digest != self.pending.plan_digest or plan.ordinal != self.pending.ordinal or branch != self.pending.branch
                or branch != want):
            raise RechargeError("the plan/branch is not the pending storm occurrence")

    def consume(self, plan: DailyRainPlan, branch: str, parent_digest: str) -> DailyCursor:
        """Accept a complete occurrence (no_rain, recharge, or the daily step after a completed storm). This scalar cursor records the
        binding only; that the physical work ran is proved by the lifecycle that calls it."""
        self.check_for_consume(plan, branch)
        return DailyCursor(self.run_digest, self.ordinal + 1, None, plan.digest, branch, _hex64(parent_digest, "parent_digest"), self.clock_contract)

    def begin_storm(self, plan: DailyRainPlan, forcing: BoundForcing, *, fallback: bool, t0_s: Any, event_sha256: Any = None,
                    capacity_failing_columns: Any = None) -> DailyCursor:
        """Mark one storm occurrence pending (ordinal NOT advanced until the genuine completion is consumed). The forcing is re-derived
        from the plan at the actual physical start `t0_s` and must equal the supplied record, native schedule included."""
        if type(fallback) is not bool:
            raise RechargeError("fallback must be a real bool")
        self.check(plan)
        if plan.selection == "no_rain" or (plan.selection == "recharge" and not fallback) or (plan.selection == "storm" and fallback):
            raise RechargeError(f"branch/selection mismatch: selection {plan.selection!r}, fallback={fallback}")
        verify_forcing(plan, forcing)
        if not _same(_real(t0_s, "t0_s"), forcing.t0_s):
            raise RechargeError("the forcing does not start at the actual physical start time")
        event = None if event_sha256 is None else _hex64(event_sha256, "event_sha256")
        return DailyCursor(self.run_digest, self.ordinal, PendingStorm(self.ordinal, plan.digest, "storm_fallback" if fallback else "storm",
                                                                       forcing.digest, event, None if capacity_failing_columns is None
                                                                       else _count(capacity_failing_columns, "capacity_failing_columns")),
                           self.last_plan_digest, self.last_branch,
                           self.last_parent_digest, self.clock_contract)

    def digest(self) -> str:
        p = self.pending
        return _digest("maple_syrup.recharge.cursor/1", {
            "run": self.run_digest, "ordinal": self.ordinal, "last_plan": self.last_plan_digest, "last_branch": self.last_branch,
            "last_parent": self.last_parent_digest, "clock": self.clock_contract,
            "pending": None if p is None else [p.ordinal, p.plan_digest, p.branch, p.forcing_digest, p.event_sha256, p.capacity_failing_columns]})


# --- strict scalar record of the cursor (the V3 daily checkpoint part) ------------------------------------------------------------------------------------------
_CURSOR_KEYS = frozenset({"run_digest", "ordinal", "pending", "last_plan_digest", "last_branch", "last_parent_digest", "clock_contract"})
_PENDING_KEYS = frozenset({"ordinal", "plan_digest", "branch", "forcing_digest", "event_sha256", "capacity_failing_columns"})


def cursor_record(cursor: Any) -> dict:
    """The bounded closed scalar record of a genuine `DailyCursor` (plain JSON-compatible values, no arrays, no history): the versioned clock contract, run, the NEXT
    ordinal, the one pending occurrence (with its fallback reason count) and the last consumed occurrence digests."""
    if type(cursor) is not DailyCursor:
        raise RechargeError("cursor must be a DailyCursor")
    p = cursor.pending
    return {"run_digest": cursor.run_digest, "ordinal": int(cursor.ordinal), "last_plan_digest": cursor.last_plan_digest, "last_branch": cursor.last_branch,
            "last_parent_digest": cursor.last_parent_digest, "clock_contract": cursor.clock_contract,
            "pending": None if p is None else {"ordinal": int(p.ordinal), "plan_digest": p.plan_digest, "branch": p.branch, "forcing_digest": p.forcing_digest,
                                               "event_sha256": p.event_sha256, "capacity_failing_columns": p.capacity_failing_columns}}


def cursor_from_record(record: Any) -> DailyCursor:
    """STRICT inverse of `cursor_record`: closed key sets, exact types (no bool-as-int, no floats for counts), then the cursor's own validators (hex digests, the
    versioned clock contract literal, the ordinal/last-occurrence/pending coherence). A malformed or tampered record raises `RechargeError`."""
    if type(record) is not dict or set(record) != _CURSOR_KEYS:
        raise RechargeError("cursor record keys mismatch (closed key set)")
    pend = record["pending"]
    pending = None
    if pend is not None:
        if type(pend) is not dict or set(pend) != _PENDING_KEYS:
            raise RechargeError("pending record keys mismatch (closed key set)")
        for key in ("ordinal",):
            if type(pend[key]) is not int:
                raise RechargeError(f"pending {key} must be an exact integer")
        if pend["capacity_failing_columns"] is not None and type(pend["capacity_failing_columns"]) is not int:
            raise RechargeError("pending capacity_failing_columns must be an exact integer or null")
        pending = PendingStorm(pend["ordinal"], pend["plan_digest"], pend["branch"], pend["forcing_digest"], pend["event_sha256"], pend["capacity_failing_columns"])
    if type(record["ordinal"]) is not int:
        raise RechargeError("cursor ordinal must be an exact integer")
    for key in ("run_digest", "last_plan_digest", "last_branch", "last_parent_digest", "clock_contract"):
        if record[key] is not None and type(record[key]) is not str:
            raise RechargeError(f"cursor {key} must be a string")
    return DailyCursor(record["run_digest"], record["ordinal"], pending, record["last_plan_digest"], record["last_branch"], record["last_parent_digest"],
                       record["clock_contract"])
