"""Phase 2 G2a: the SMALL native hook that couples the final daily CLOSE of the Phase 1.5 daily lifecycle to the G0 every-N-days moisture writer.

What it is
----------
One operation, `close_final_day`, takes the genuine `interevent` state of a daily lineage (the state `daily_water.daily_water_action` /
`finish_daily_storm` returns, after any actual LEAF `request_withdrawal` of the caller) and:

1. checks, BEFORE any model call or I/O, that it is a genuine daily-lineage interevent (not a boundary, not a storm, no pending storm, and
   a lineage with a real `DailyCursor` -- nothing here constructs a cursor or a nominal clock);
2. calls the EXISTING `water_lifecycle.close_interevent` with the CALLER's `expected_parent` capability for that state (full window/lifetime audit, the
   closed `boundary` that carries the SAME cursor -- the day is final only here, after any accepted T);
3. if an output is supplied, notifies the existing G0 `MoistureOutput.notify_day_closed` for exactly that accepted closed boundary
   (the child capability is minted from the closed state this operation itself returns; G0 owns cadence N/2N/..., the sequential-day rule, quotas, atomic no-clobber publication, replay/resume); with no output nothing is
   read, transferred or written (the option is OFF by absence; there is no default N).

It returns `DailyCloseResult(state=<closed boundary>, notice=<MoistureNotice | None>)`. If the output step refuses or fails, the G0
exception propagates and NO closed state is returned: the caller still holds the accepted interevent parent, which was never mutated
(the lifecycle API is pure), and an I/O failure is never treated as model completion. A retry calls this operation again with the same
parent: G0 verifies an already sealed day against the live bytes (a due day) instead of overwriting; a duplicate call for an already
notified NON-due day is refused by G0 (different state instance) -- keep and reuse the returned closed state instead.

What it is not
--------------
No E/T/growth, no extra physical time, no drainage, no second debit or invented demand: plants (if any) already withdrew water through
the actual `request_withdrawal`; without plants the close is still the legitimate final close. Nominal days are the lineage's daily
ordinals (storm/recession physical duration is additional and never divides into days). No per-day physical checkpoint, no CLI key,
no scheduler, no weather. LIMITS: the G0 snapshot schema is the FIXED-full-porous common-column profile (`water_m` + common `dz`); the
floating porous-depth hook (W + canonical D + generations, `floating_moisture_frame`) is NOT wired here and remains pending the
floating daily owner (E3/E4) and the explicit new floating profile.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from maple_syrup import soil_moisture_output as mo
from maple_syrup import water_lifecycle as wl

__all__ = [
    "DailyCloseResult",
    "close_final_day",
    "create_daily_moisture_output",
    "resume_daily_moisture_output",
]


@dataclass(frozen=True)
class DailyCloseResult:
    """`state`: the genuine closed daily `boundary`; `notice`: the G0 notice for that day (None when no output is configured)."""

    state: Any
    notice: mo.MoistureNotice | None


def _need_expected_parent(expected_parent: Any) -> None:
    if type(expected_parent) is not wl.ExpectedParent:
        raise mo.MoistureOutputError("expected_parent is REQUIRED and must be an ExpectedParent minted by water_lifecycle.expect_parent() for the "
                                     "accepted state the caller means; this facade never mints it for the state argument")


def create_daily_moisture_output(directory: str | Path, lop: Any, origin_state: Any, *, expected_parent: Any, every_n_days: int,
                                 limits: mo.MoistureOutputLimits, origin_date: str | None = None, caller_identity: dict | None = None,
                                 host_available_bytes_fn: Callable[[], int] | None = None,
                                 disk_available_bytes_fn: Callable[[], int] | None = None) -> mo.MoistureOutput:
    """Thin delegate of `MoistureOutput.create` for a closed daily boundary (`every_n_days` is required: no default; limits are explicit).
    The caller's `expected_parent` is forwarded unchanged to the native/G0 checks."""
    _need_expected_parent(expected_parent)
    return mo.MoistureOutput.create(directory, lop, origin_state, expected_parent=expected_parent, every_n_days=every_n_days,
                                    limits=limits, origin_date=origin_date, caller_identity=caller_identity,
                                    host_available_bytes_fn=host_available_bytes_fn, disk_available_bytes_fn=disk_available_bytes_fn)


def resume_daily_moisture_output(directory: str | Path, lop: Any, state: Any, *, expected_parent: Any, limits: mo.MoistureOutputLimits,
                                 caller_identity: dict | None = None,
                                 host_available_bytes_fn: Callable[[], int] | None = None) -> mo.MoistureOutput:
    """Thin delegate of `MoistureOutput.resume` for the (possibly lagging) physical restart boundary; `expected_parent` is the caller's."""
    _need_expected_parent(expected_parent)
    return mo.MoistureOutput.resume(directory, lop, state, expected_parent=expected_parent, limits=limits,
                                    caller_identity=caller_identity, host_available_bytes_fn=host_available_bytes_fn)


def _check_final_close_input(state: Any) -> None:
    if getattr(state, "role", None) != "interevent":
        raise mo.MoistureOutputError(f"only the interevent of a daily occurrence can be closed here, got role {getattr(state, 'role', None)!r} "
                                     "(a boundary is already closed; a storm is not an interevent)")
    cur = getattr(state, "daily", None)
    if cur is None:
        raise mo.MoistureOutputError("the state does not belong to a daily lineage (no DailyCursor): the nominal day is not defined")
    if cur.pending is not None:
        raise mo.MoistureOutputError("a storm occurrence is pending in the cursor: the day is not complete")


def close_final_day(lop: Any, interevent_state: Any, *, expected_parent: Any, output: mo.MoistureOutput | None = None,
                    host_available_bytes_fn: Callable[[], int] | None = None,
                    disk_available_bytes_fn: Callable[[], int] | None = None) -> DailyCloseResult:
    """Close the daily interevent with the actual lifecycle and, if configured, notify the G0 writer at that accepted final close (see
    the module docstring). `interevent_state` is never mutated. `expected_parent` is REQUIRED caller authority (the capability naming the exact
    accepted parent instance) and is forwarded to the actual `close_interevent` check; a wrong/stale/alternate-branch capability is refused
    there before any audit, model work or I/O. `output` must be None or an EXACT `MoistureOutput` (checked before closing)."""
    _need_expected_parent(expected_parent)
    if output is not None and type(output) is not mo.MoistureOutput:
        raise mo.MoistureOutputError(f"output must be None or a MoistureOutput, got {type(output).__name__} (no duck-typed substitutes)")
    _check_final_close_input(interevent_state)
    closed = wl.close_interevent(lop, interevent_state, expected_parent=expected_parent)
    if output is None:
        return DailyCloseResult(closed, None)
    notice = output.notify_day_closed(lop, closed, expected_parent=wl.expect_parent(closed), host_available_bytes_fn=host_available_bytes_fn,
                                      disk_available_bytes_fn=disk_available_bytes_fn)
    return DailyCloseResult(closed, notice)
