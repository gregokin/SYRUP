"""Phase 1.5 native daily water transactions on a DAILY lineage (S2: CPU, Dense/bedless; no scheduler, no calendar, no CSV, no LEAF physics).

SYRUP owns the water; a later scheduler owns the nominal calendar. This module exposes a narrow native API that advances a lifecycle by
exactly ONE occurrence per accepted call, with a fixed native day (`recharge.DAY_S` = 86400 s, not caller adjustable) and the versioned
`recharge.CLOCK_CONTRACT`:

  * `start_daily_lineage` : a genuine closed `boundary` -> the same state carrying a bounded `DailyCursor` (explicit opt-in; a state without
    a cursor is a Phase 1 lineage and behaves exactly as before). The old free-standing transitions (`begin_storm`, `enter_interevent_*`,
    `evolve_interevent`) are refused on a daily lineage (`water_lifecycle._refuse_daily`) so no day, storm or entry can bypass the cursor.
  * `daily_water_action`   : `no_rain` (P = 0) and `recharge` (0 < P <= T): validation, canonical full forcing, expected parent and whole-call
    memory BEFORE any trial; the genuine interevent entry; then ONE general interevent trial of 86400 s (retained pond preferential/matrix
    -> the ONE pulse-aware column movement + E + diffusion -> the actual router). No `evolve_interevent`, no halving, no sub-day acceptance.
    A router rejection (`HydraulicStepRejected`), a quiet/state/memory/forcing error is a typed REFUSAL (`DailyRefusal`): the parent, clock,
    E/T, bed, ledgers and cursor stay exactly as they were. Only a genuine active-column capacity failure of the pulse
    (`SubsurfaceFitFailure`) returns the distinct `capacity_failure` outcome carrying a non-owning `CapacityFailure` receipt.
  * `begin_daily_storm`    : `P > T`, or the capacity fallback of a `recharge` plan (requires its `CapacityFailure`): the genuine `begin_storm` with
    the FULL native constant forcing of the plan and the uniform active field, bound to the plan, forcing, ordinal and the real event identity
    in the cursor's pending occurrence. The caller drives the existing `evolve_storm`; an unfinished storm stays unfinished.
  * `finish_daily_storm`   : a genuinely COMPLETED storm -> the genuine `enter_interevent` -> ONE 86400 s no-recharge general trial -> consumes
    the occurrence.

Every accepted daily/finish call consumes exactly one ordinal of the cursor; a P = 0 day is a real day. The partial-rain candidate of a failed
recharge is never adopted and nothing of the original boundary is retained (the receipt holds a weak reference only). Zero-time T requests and
`close_interevent` remain the existing functions and are usable on the resulting interevent state.
"""

from __future__ import annotations

import hashlib
import json
import weakref
from dataclasses import dataclass
from typing import Any

import numpy as np

from maple_syrup import interevent_transpiration as it
from maple_syrup import interevent_water as iw
from maple_syrup import layered_event as ev
from maple_syrup import layered_storm as ls
from maple_syrup import recharge as rc
from maple_syrup import subsurface as ss
from maple_syrup import water_lifecycle as wl
from maple_syrup import water_lifecycle_cuda as wlc
from maple_syrup.rainfall import rainfall_field
from maple_syrup.surface_routing import HydraulicStepRejected

__all__ = [
    "CapacityFailure",
    "DailyError",
    "DailyReceipt",
    "DailyRefusal",
    "DailyResult",
    "begin_daily_storm",
    "cuda_daily_plan",
    "daily_water_action",
    "finish_daily_storm",
    "next_plan",
    "required_begin_daily_storm_bytes",
    "required_daily_bytes",
    "required_finish_daily_bytes",
    "start_daily_lineage",
]

DAY_S = rc.DAY_S
_SEAL = object()
_MINTED_FALLBACK = weakref.WeakSet()


class DailyError(wl.LifecycleError):
    """A daily-lineage contract violation (not a physical outcome)."""


class DailyRefusal(DailyError):
    """A typed refusal of one daily transaction: the parent, its clock, E/T, bed, every ledger and the cursor are unchanged. It is never a
    capacity fallback. `code` names the class: occurrence, forcing, state, entry, memory, router_step_rejected, step_refused, storm_incomplete."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"[{code}] {message}")
        self.code = code


def _digest(tag: str, body: Any) -> str:
    return hashlib.sha256((tag + "\n" + json.dumps(body, sort_keys=True, separators=(",", ":"))).encode("ascii")).hexdigest()


# --- records -------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class DailyReceipt:
    """Bounded scalars of ONE accepted occurrence (never arrays or a chain). Modeled preferential/matrix intake (from the retained pond) and
    the recharge pulse (external rain) are reported separately; the nominal ordinal is separate from the physical seconds."""

    clock_contract: str
    run_digest: str
    ordinal: int
    branch: str
    plan_digest: str
    forcing_digest: str | None
    parent_digest: str
    parent_revision: int
    selection: str  # the plan's selection (no_rain / recharge / storm); `branch` is what actually ran (storm_fallback after a capacity failure)
    p_mm: float
    i_mm_h: float
    t_mm: float
    capacity_failing_columns: int | None  # storm_fallback only: how many ACTIVE columns rejected the recharge pulse
    occurrence_start_s: float  # the physical start of the occurrence (the storm origin for a storm branch, else the day start)
    t_start_s: float  # the physical start of the fixed DAY step (the storm end for a storm branch)
    t_end_s: float
    recharge_depth_m: float
    recharge_m3: float
    preferential_intake_m3: float
    matrix_intake_m3: float
    drainage_m3: float
    surface_evaporation_m3: float
    soil_evaporation_m3: float
    export_m3: float

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    def digest(self) -> str:
        return _digest("maple_syrup.daily_water.receipt/1", self.as_dict())


class CapacityFailure:
    """Non-owning, factory-only evidence that the ACTIVE-column capacity test rejected the recharge pulse of ONE plan at ONE physical parent
    instance. It holds only scalars and a weak reference (never the parent, bed or any array), so it cannot keep the original boundary alive
    and an equal-digest alternate branch is a different instance. `begin_daily_storm` consumes it as the only authority for the whole-occurrence
    fallback of a `recharge` plan."""

    __slots__ = ("__weakref__", "_ref", "digest", "failing_columns", "forcing_digest", "i_mm_h", "ordinal", "p_mm", "parent_digest",
                 "parent_revision", "plan_digest", "reason_bit", "run_digest", "selection", "t_mm", "t_s")

    def __init__(self, seal: object, parent: wl.LifecycleState, plan: rc.DailyRainPlan, forcing: rc.BoundForcing,
                 failure: ss.SubsurfaceFitFailure) -> None:
        if seal is not _SEAL:
            raise DailyError("CapacityFailure is minted only by daily_water_action()")
        set_ = object.__setattr__
        set_(self, "_ref", weakref.ref(parent))
        set_(self, "run_digest", plan.run_digest)
        set_(self, "ordinal", int(plan.ordinal))
        set_(self, "plan_digest", plan.digest)
        set_(self, "forcing_digest", forcing.digest)
        set_(self, "parent_digest", parent.digest)
        set_(self, "parent_revision", int(parent.revision))
        set_(self, "t_s", float(parent.t_s))
        # the bounded scalar REASON (the owner packet already provides it; no failed-cell mask, no history): the lowest flag bit (17 = recharge
        # capacity) and how many ACTIVE columns failed, plus the plan inputs the choice was made on
        set_(self, "reason_bit", int(failure.bit))
        set_(self, "failing_columns", int(failure.n_flagged))
        set_(self, "selection", plan.selection)
        set_(self, "p_mm", float(plan.p_mm))
        set_(self, "i_mm_h", float(plan.i_mm_h))
        set_(self, "t_mm", float(plan.t_mm))
        set_(self, "digest", _digest("maple_syrup.daily_water.capacity_failure/1", {
            "run": self.run_digest, "ordinal": self.ordinal, "plan": self.plan_digest, "forcing": self.forcing_digest,
            "parent": self.parent_digest, "revision": self.parent_revision, "t": self.t_s, "bit": self.reason_bit,
            "columns": self.failing_columns, "p": self.p_mm, "i": self.i_mm_h, "tt": self.t_mm}))
        _MINTED_FALLBACK.add(self)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("CapacityFailure is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("CapacityFailure is immutable")


@dataclass(frozen=True, eq=False)
class DailyResult:
    """`complete`: `state` is the accepted interevent state and `receipt` the occurrence record. `capacity_failure`: no state (the parent is
    unchanged and the cursor was not advanced); `fallback` is the non-owning evidence for `begin_daily_storm`."""

    status: str
    state: wl.LifecycleState | None
    receipt: DailyReceipt | None
    fallback: CapacityFailure | None


# --- shared checks -------------------------------------------------------------------------------------------------------------
def _backend_gate(lop: wl.LifecycleOperator) -> None:
    """Backend dispatch of every daily transaction. NumPy/Numba lifecycles: the unchanged CPU path. CUDA: only the explicit native9f BOUNDARY-capable
    Dense lifecycle (genuine capability record, current-device gate, cached native9f source association and the bound native Dense class are all
    revalidated by `_check_lop` BEFORE any allocation or query); the storm-first profile (no boundary entry), the native39 pair, a bedless or a
    Windowed lifecycle are refused here, never routed to a CPU fallback."""
    wl._check_lop(lop)
    cap = lop._cuda
    if cap is None:
        return
    if cap.profile != wlc.CUDA9F_BOUNDARY_PROFILE:
        raise DailyRefusal("state", f"the daily CUDA transactions need the explicit native9f CUDA9F_BOUNDARY_PROFILE lifecycle, got {cap.profile!r}; refused before any work")
    if lop.context is None:
        raise DailyRefusal("state", "the daily CUDA transactions need a native Dense bed lifecycle (bedless is not covered); refused before any work")
    if getattr(lop, "windowed", None) is not None or getattr(lop.column, "windowed", None) is not None:
        raise DailyRefusal("state", "Windowed storage is not covered by the daily transactions; refused before any work")


def _check_pet(lop: wl.LifecycleOperator, pet: Any) -> None:
    """Strict resident float64 (N,) potential-evaporation input of the lifecycle's own namespace/device: metadata only; a NumPy array on a CUDA lifecycle
    (or a foreign-device array) is refused here, never uploaded or copied implicitly."""
    try:
        ss._check_dynamic(lop.column, "pet_terrain_m_s", pet, (lop.n_cells,))
    except ss.SubsurfaceError as exc:
        raise _wrap("step_refused", exc) from None


def _wrap(code: str, exc: BaseException) -> DailyRefusal:
    err = DailyRefusal(code, f"{type(exc).__name__}: {exc}")
    err.__cause__ = exc
    return err


def _daily_parent(lop: wl.LifecycleOperator, parent: Any, expected_parent: Any, role: str, what: str) -> rc.DailyCursor:
    _backend_gate(lop)
    try:
        wl._require_parent(parent, expected_parent)
    except wl.LifecycleError as exc:
        raise _wrap("state", exc) from None
    if parent.role != role:
        raise DailyRefusal("state", f"{what} needs a {role!r} parent, got {parent.role!r}")
    if parent.daily is None:
        raise DailyRefusal("state", f"{what} needs a daily lineage (start_daily_lineage); this state carries no daily cursor")
    return parent.daily


def _check_plan(cursor: rc.DailyCursor, plan: Any, branch: str) -> None:
    try:
        cursor.check_for_consume(plan, branch)
    except rc.RechargeError as exc:
        raise _wrap("occurrence", exc) from None


def _cpu_figure_only(lop: wl.LifecycleOperator) -> None:
    if lop._cuda is not None:
        raise DailyRefusal("memory", "the required_*_bytes figures are single CPU integers; a CUDA lifecycle has separate host and device figures: use "
                                     "cuda_daily_plan(...) (never one count interpreted as both)")


def cuda_daily_plan(lop: wl.LifecycleOperator, kind: str, parent: wl.LifecycleState, *, plan: rc.DailyRainPlan | None = None,
                    retained_feedback_bytes: Any = 0) -> dict[str, Any]:
    """The METADATA-ONLY whole-call plan of one daily CUDA transaction with SEPARATE `host_bytes` and `device_bytes` (and named terms/stages): `kind` is
    `start`, `daily`, `begin_storm` (needs the storm/recharge `plan` for its actual forcing schedule) or `finish`. It runs the sealed type/component/
    source/cover-metadata checks without the current-device query, reads no availability, no array, no hash, copies nothing. Execution paths admit exactly
    this plan through the registered CUDA admission (host first, then device) AFTER the current-device gate and BEFORE any new N-sized allocation."""
    if lop._cuda is None:
        raise DailyRefusal("state", "cuda_daily_plan exists only for a CUDA lifecycle (CPU figures are the required_*_bytes functions)")
    fb = wl._count(retained_feedback_bytes, "retained_feedback_bytes")
    entry = {"start": "daily_start", "daily": "daily", "begin_storm": "daily_begin_storm", "finish": "daily_finish"}.get(kind)
    if entry is None:
        raise DailyRefusal("state", f"unknown daily plan kind {kind!r}")
    schedule = None
    if entry == "daily_begin_storm":
        if type(plan) is not rc.DailyRainPlan:
            raise DailyRefusal("occurrence", "the begin_storm plan needs the DailyRainPlan (its actual forcing schedule is part of the figure)")
        try:
            forcing = plan.bind_forcing(parent.t_s)
        except rc.RechargeError as exc:
            raise _wrap("forcing", exc) from None
        if forcing is None:
            raise DailyRefusal("occurrence", "a no_rain plan has no storm")
        schedule = forcing.schedule
    try:
        return wl.cuda_entry_plan(lop, entry, parent=parent, schedule=schedule, retained_feedback_bytes=fb)
    except wl.LifecycleError as exc:
        raise _wrap("state", exc) from None


def required_daily_bytes(lop: wl.LifecycleOperator, parent: wl.LifecycleState, retained_feedback_bytes: Any = 0) -> int:
    """METADATA-ONLY whole-call requirement of one `daily_water_action`, admitted BEFORE any entry, hash, canonical array or trial: the
    largest simultaneous stage of (a) the genuine boundary entry (operator build, owners, quiet scratch: `required_enter_from_boundary_bytes`)
    and (b) the ONE day step with the accepted parent still live beside the entry's owners, candidate and scratch: the general owner's step/
    audit/post figures (which count its RESIDENT operands, including the zero-added tensor that stays resident though the pulse step does not
    read it) plus the wrapper-owned call-parent arrays, the lifecycle's own cover/T bounds and the caller's retained feedback bytes. The caller's
    reserves are inside the owner figures. No deep-bed copy, no retained parent chain, no dense history, no per-layer host read."""
    wl._check_lop(lop)
    _cpu_figure_only(lop)
    if type(parent) is not wl.LifecycleState or parent.role != "boundary":
        raise DailyRefusal("state", "a 'boundary' lifecycle state is required")
    fb = wl._count(retained_feedback_bytes, "retained_feedback_bytes")
    m = lop.memory
    _, _, gm = wl._boundary_estimates(lop)
    enter = wl.required_enter_from_boundary_bytes(lop, parent)
    after = gm.resident_bytes + gm.reserves_bytes + gm.column_reserve_bytes + m.audit_scratch_bytes
    step = max(int(gm.step_required_bytes), int(gm.audit_required_bytes), int(after)) + m.owned_fixed_bytes + m.call_parent_bytes + fb
    return int(max(int(enter), int(step)))


#: Uniform active rainfall field built by `rainfall.rainfall_field`: `ones` (8 N), the `where` result (8 N), at most one frozen copy (8 N) and the
#: reshaped bool mask view/copy (N): declared as 25 N bytes, live beside the event build.
_FIELD_BYTES_PER_CELL = 25


def required_begin_daily_storm_bytes(lop: wl.LifecycleOperator, parent: wl.LifecycleState, plan: rc.DailyRainPlan) -> int:
    """METADATA-ONLY whole-call requirement of `begin_daily_storm`, admitted BEFORE the N-sized uniform rainfall field is built: the genuine
    `begin_storm` requirement for the plan's ACTUAL forcing schedule (storm operator + event build + begin copies + owned fixed bytes + reserves,
    with the accepted boundary parent live) plus the field temporaries the wrapper itself allocates. The forcing is O(1) scalars."""
    wl._check_lop(lop)
    _cpu_figure_only(lop)
    if type(parent) is not wl.LifecycleState or parent.role != "boundary":
        raise DailyRefusal("state", "a 'boundary' lifecycle state is required")
    try:
        forcing = plan.bind_forcing(parent.t_s)
    except rc.RechargeError as exc:
        raise _wrap("forcing", exc) from None
    if forcing is None:
        raise DailyRefusal("occurrence", "a no_rain plan has no storm")
    return int(wl.required_begin_storm_bytes(lop, forcing.schedule) + _FIELD_BYTES_PER_CELL * lop.n_cells)


def required_finish_daily_bytes(lop: wl.LifecycleOperator, parent: wl.LifecycleState, retained_feedback_bytes: Any = 0) -> int:
    """METADATA-ONLY whole-call requirement of `finish_daily_storm`, admitted BEFORE the entry: the largest simultaneous stage of (a) the genuine
    `enter_interevent` (its own figure already counts the live completed-storm event beside the owners) and (b) the ONE day step while the
    ORIGINAL completed-storm parent (its binding, event, field and quiet state) stays live beside the new owners, candidate and scratch: the
    general owner's step/audit/post figures + the live event bytes + the wrapper-owned call-parent arrays + the lifecycle's owned fixed bytes +
    the caller's retained feedback. Scope exit or a copy-free alias is never assumed to release the parent."""
    wl._check_lop(lop)
    _cpu_figure_only(lop)
    if type(parent) is not wl.LifecycleState or parent.role != "storm":
        raise DailyRefusal("state", "a 'storm' lifecycle state is required")
    fb = wl._count(retained_feedback_bytes, "retained_feedback_bytes")
    m = lop.memory
    op, event = parent.binding.op, parent.binding.event
    _, gm = wl._interevent_estimates(lop, op)
    live_event = max(int(event.required_bytes) - int(op.memory.required_bytes), 0)
    enter = wl.required_enter_interevent_bytes(lop, parent)
    after = gm.resident_bytes + gm.reserves_bytes + gm.column_reserve_bytes + m.audit_scratch_bytes
    step = (max(int(gm.step_required_bytes), int(gm.audit_required_bytes), int(after)) + m.owned_fixed_bytes + m.call_parent_bytes + live_event + fb)
    return int(max(int(enter), int(step)))


def next_plan(state: wl.LifecycleState, p_mm: Any, i_mm_h: Any, t_mm: Any, *, label: Any = None) -> rc.DailyRainPlan:
    """The plan of the NEXT occurrence of a daily lineage (its run digest and ordinal are taken from the cursor)."""
    if type(state) is not wl.LifecycleState or state.daily is None:
        raise DailyRefusal("state", "a daily lineage state is required")
    try:
        return rc.plan_daily_rain(p_mm, i_mm_h, t_mm, run_digest=state.daily.run_digest, ordinal=state.daily.ordinal, label=label)
    except rc.RechargeError as exc:
        raise _wrap("occurrence", exc) from None


# --- start ---------------------------------------------------------------------------------------------------------------------
def start_daily_lineage(lop: wl.LifecycleOperator, parent: wl.LifecycleState, *, expected_parent: wl.ExpectedParent,
                        run_digest: Any) -> wl.LifecycleState:
    """Begin a daily lineage at an ACTUAL validated closed boundary: the same physical arrays (shared by reference, bitwise), a fresh
    cursor at ordinal 0 with nothing pending, and revision + 1. The explicit opt-in: nothing is inferred from a filename, a schedule or a
    restart. A state that already carries a cursor is refused."""
    _backend_gate(lop)
    try:
        wl._require_parent(parent, expected_parent)
        if parent.role != "boundary":
            raise DailyRefusal("state", f"start_daily_lineage needs a closed 'boundary' parent, got {parent.role!r}")
        if parent.daily is not None:
            raise DailyRefusal("state", "the state already belongs to a daily lineage")
        cursor = rc.DailyCursor(rc._hex64(run_digest, "run_digest"))
        # the inventory audit's finite/capacity masks, BEFORE they exist (CUDA: the registered `daily_start` plan, separate host then device)
        wl._admit(lop, 0 if lop._cuda is not None else wl.required_audit_bytes(lop, parent), "daily_start", parent=parent)
        wl._check_state(lop, parent)
        wl._validate_inventories(lop, wl._arrays_of(parent))
    except DailyRefusal:
        raise
    except (wl.LifecycleError, rc.RechargeError, ss.SubsurfaceError, ls.LayeredStormError, MemoryError) as exc:
        raise _wrap("memory" if isinstance(exc, MemoryError) else "state", exc) from None
    return wl._mint(parent, revision=parent.revision + 1, daily=cursor)


# --- no_rain / recharge ----------------------------------------------------------------------------------------------------------
def daily_water_action(lop: wl.LifecycleOperator, parent: wl.LifecycleState, *, expected_parent: wl.ExpectedParent, plan: rc.DailyRainPlan,
                       pet_terrain_m_s: Any, policy: ev.LayeredQuietPolicy = ev.LayeredQuietPolicy(),
                       retained_feedback_bytes: Any = 0) -> DailyResult:
    """One occurrence of a `no_rain` (P = 0: no precipitation operation, but the day's E/soil update still runs) or `recharge` plan
    (0 < P <= T: the uniform top-layer pulse through the ONE column movement). Order: genuine parent -> cursor/plan occurrence -> canonical
    forcing (recharge) -> whole-call memory -> genuine boundary entry (existing quiet gates) -> ONE general interevent trial of the fixed day
    -> window audit -> fold + consume. Pure: any refusal or the capacity outcome leaves `parent` untouched (nothing was adopted)."""
    cursor = _daily_parent(lop, parent, expected_parent, "boundary", "daily_water_action")
    if type(plan) is not rc.DailyRainPlan:
        raise DailyRefusal("occurrence", "plan must be a DailyRainPlan")
    if plan.selection not in ("no_rain", "recharge"):
        raise DailyRefusal("occurrence", f"a {plan.selection!r} plan is a storm occurrence: use begin_daily_storm")
    _check_plan(cursor, plan, plan.selection)
    _check_pet(lop, pet_terrain_m_s)
    forcing = None
    if plan.selection == "recharge":  # the FULL native forcing of every positive P is derived and validated BEFORE the trial (fallback needs it)
        try:
            forcing = plan.bind_forcing(parent.t_s)
        except rc.RechargeError as exc:
            raise _wrap("forcing", exc) from None
    fb = wl._count(retained_feedback_bytes, "retained_feedback_bytes")
    try:
        need = 0 if lop._cuda is not None else required_daily_bytes(lop, parent, fb)  # CUDA: the registered `daily` plan replaces the figure
        wl._admit(lop, need, "daily", parent=parent, feedback=fb)  # BEFORE the entry, any hash, any canonical array or trial
        wl._check_state(lop, parent)
        consumed = cursor.consume(plan, plan.selection, parent.digest)  # PURE staging: the parent's cursor object is never touched
    except DailyRefusal:
        raise
    except (wl.LifecycleError, ls.LayeredStormError, ss.SubsurfaceError, rc.RechargeError, MemoryError) as exc:
        raise _wrap("memory" if isinstance(exc, MemoryError) else "state", exc) from None
    try:  # the genuine entry on the TRUE parent and the caller's ExpectedParent; the staged interevent carries the prescribed consumed cursor
        with wl._daily_engine(parent, lambda payload: consumed):
            entry = wl.enter_interevent_from_boundary(lop, parent, expected_parent=expected_parent, policy=policy)
    except (wl.LifecycleError, iw.InterEventWaterError, it.GeneralIntereventError, ls.LayeredStormError, ss.SubsurfaceError, MemoryError) as exc:
        raise _wrap("entry", exc) from None
    gop, state0 = entry.binding.gop, entry.payload
    t_start = float(state0.t_s)
    try:
        if plan.selection == "recharge":
            cand = it.general_interevent_recharge_step(gop, state0, pet_terrain_m_s, DAY_S, recharge_depth_m=plan.depth_m, phase="interevent",
                                                       retained_feedback_bytes=fb)
        else:
            cand = it.general_interevent_water_step(gop, state0, pet_terrain_m_s, DAY_S, phase="interevent", retained_feedback_bytes=fb)
    except ss.SubsurfaceFitFailure as failure:
        if plan.selection != "recharge":
            raise
        return DailyResult("capacity_failure", None, None, CapacityFailure(_SEAL, parent, plan, forcing, failure))
    except HydraulicStepRejected as exc:  # the late CFL of the ONE day step: a refusal, never a sub-day retry and never a storm fallback
        raise _wrap("router_step_rejected", exc) from None
    except (it.GeneralIntereventError, iw.InterEventWaterError, ls.LayeredStormError, ss.SubsurfaceError, MemoryError) as exc:
        raise _wrap("memory" if isinstance(exc, MemoryError) else "step_refused", exc) from None
    try:
        it.audit_general_interevent(gop, cand.state)
        new = wl._fold_interevent(lop, entry, cand.state, daily=consumed)
        wl._audit_core(lop, new, wl._arrays_of(new), new.t_s)
    except (wl.LifecycleError, iw.InterEventWaterError, it.GeneralIntereventError, rc.RechargeError) as exc:
        raise _wrap("step_refused", exc) from None
    return DailyResult("complete", new, _receipt(plan, plan.selection, forcing, parent, t_start, t_start, cand), None)


def _receipt(plan: rc.DailyRainPlan, branch: str, forcing: Any, parent: wl.LifecycleState, origin: float, t_start: float, cand: Any) -> DailyReceipt:
    r = cand.receipt
    rech = float(r.recharge_m3) if type(r) is iw.RechargeStepReceipt else 0.0
    failing = None if parent.daily is None or parent.daily.pending is None else parent.daily.pending.capacity_failing_columns
    depth = float(r.recharge_depth_m) if type(r) is iw.RechargeStepReceipt else 0.0
    return DailyReceipt(
        rc.CLOCK_CONTRACT, plan.run_digest, int(plan.ordinal), branch, plan.digest, None if forcing is None else forcing.digest, parent.digest,
        int(parent.revision), plan.selection, float(plan.p_mm), float(plan.i_mm_h), float(plan.t_mm), failing, float(origin), float(t_start), float(cand.state.t_s), depth, rech, float(r.preferential.intake_m3), float(r.matrix.intake_m3),
        float(r.column.drainage_m3), float(r.surface_evaporation_m3), float(r.soil_evaporation_m3), float(r.export_m3))


# --- storm branches -----------------------------------------------------------------------------------------------------------
def begin_daily_storm(lop: wl.LifecycleOperator, parent: wl.LifecycleState, *, expected_parent: wl.ExpectedParent, plan: rc.DailyRainPlan,
                      fallback_of: Any = None, control: ev.LayeredEventControl = ev.LayeredEventControl(),
                      policy: ev.LayeredQuietPolicy = ev.LayeredQuietPolicy()) -> wl.LifecycleState:
    """Begin the FULL native storm of a `storm` plan (P > T), or the whole-occurrence fallback of a `recharge` plan whose pulse the active
    column genuinely could not hold (`fallback_of` = that `CapacityFailure`, bound to THIS physical parent instance, run, ordinal, plan and
    forcing). The genuine `begin_storm` runs from the ORIGINAL accepted boundary with the plan's full constant forcing (full P) and the
    uniform active field; the cursor then holds the one pending occurrence bound to the plan, forcing and the real event identity (the
    ordinal is NOT advanced). The caller drives the existing `evolve_storm`."""
    cursor = _daily_parent(lop, parent, expected_parent, "boundary", "begin_daily_storm")
    if type(plan) is not rc.DailyRainPlan:
        raise DailyRefusal("occurrence", "plan must be a DailyRainPlan")
    if plan.selection == "no_rain":
        raise DailyRefusal("occurrence", "a no_rain plan has no storm")
    fallback = fallback_of is not None
    if (plan.selection == "recharge") != fallback:
        raise DailyRefusal("occurrence", "a recharge plan starts a storm only as the fallback of its own CapacityFailure; a storm plan has no fallback receipt")
    try:
        cursor.check(plan)
        forcing = plan.bind_forcing(parent.t_s)
    except rc.RechargeError as exc:
        raise _wrap("occurrence" if "forcing" not in str(exc) else "forcing", exc) from None
    if fallback:
        f = fallback_of
        if type(f) is not CapacityFailure or f not in _MINTED_FALLBACK:
            raise DailyRefusal("occurrence", "fallback_of must be the CapacityFailure minted by daily_water_action")
        if (f._ref() is not parent or f.parent_digest != parent.digest or f.parent_revision != parent.revision or f.t_s != float(parent.t_s)
                or f.plan_digest != plan.digest or f.run_digest != cursor.run_digest or f.ordinal != plan.ordinal
                or f.forcing_digest != forcing.digest):
            raise DailyRefusal("occurrence", "the CapacityFailure belongs to another physical parent instance, plan, ordinal, run or forcing (stale or foreign); refused")
    ny, nx = (int(v) for v in lop.shape)
    failing = fallback_of.failing_columns if fallback else None

    def pending_for(payload: Any) -> rc.DailyCursor:  # PURE: builds the pending cursor from the genuine event identity; the parent's cursor is untouched
        return cursor.begin_storm(plan, forcing, fallback=fallback, t0_s=parent.t_s, event_sha256=payload.event_sha256, capacity_failing_columns=failing)

    try:
        need = 0 if lop._cuda is not None else required_begin_daily_storm_bytes(lop, parent, plan)  # CUDA: the registered plan (field is a device term)
        wl._admit(lop, need, "daily_begin_storm", parent=parent, schedule=forcing.schedule)  # BEFORE the field
        if lop._cuda is None:
            field = rainfall_field(ny, nx, active_mask=np.asarray(lop.topology.active).reshape(ny, nx), xp=np)
        else:  # the device-resident active mask (a VIEW, no copy, no host round trip) and the lifecycle's own namespace
            field = rainfall_field(ny, nx, active_mask=lop.column.active.reshape(ny, nx), xp=lop.xp)
        with wl._daily_engine(parent, pending_for):  # the genuine begin_storm on the TRUE parent; its result carries the pending cursor
            st = wl.begin_storm(lop, parent, expected_parent=expected_parent, schedule=forcing.schedule, field=field, control=control, policy=policy)
    except DailyRefusal:
        raise
    except (wl.LifecycleError, ls.LayeredStormError, ev.LayeredEventError, rc.RechargeError, ss.SubsurfaceError, MemoryError) as exc:
        raise _wrap("memory" if isinstance(exc, MemoryError) else "entry", exc) from None
    return st


def finish_daily_storm(lop: wl.LifecycleOperator, parent: wl.LifecycleState, *, expected_parent: wl.ExpectedParent, plan: rc.DailyRainPlan,
                       pet_terrain_m_s: Any, retained_feedback_bytes: Any = 0) -> DailyResult:
    """The day after a GENUINELY COMPLETED daily storm (rain + recession; a bound/target stop is not completion and enters nothing): the
    genuine `enter_interevent` then ONE 86400 s general trial with NO recharge, consuming the pending occurrence. The storm's schedule is
    re-derived from the plan at the storm's physical start and must equal the actual prepared event schedule and the pending forcing/event
    identities (a cursor alone is never proof of execution). A router rejection is a typed refusal that leaves the completed storm and
    the cursor unchanged (retry from the same parent)."""
    cursor = _daily_parent(lop, parent, expected_parent, "storm", "finish_daily_storm")
    pend = cursor.pending
    if pend is None or type(plan) is not rc.DailyRainPlan:
        raise DailyRefusal("occurrence", "no pending storm occurrence / plan must be a DailyRainPlan")
    _check_plan(cursor, plan, pend.branch)
    _check_pet(lop, pet_terrain_m_s)
    if parent.payload.completed is not True:
        raise DailyRefusal("storm_incomplete", "the storm is not genuinely completed (a bound-stopped or flowing storm is still rain/recession and "
                                               "receives no daily step); refused before any owner is prepared")
    try:
        event = parent.binding.event
        forcing = plan.bind_forcing(event.event_origin_s)
        edges = event.absolute_edges_s  # the ACTUAL native absolute edges (origin + local schedule): must be exactly [t0, t0 + D]
        if (forcing.digest != pend.forcing_digest or rc._schedule_digest(event.schedule) != rc._schedule_digest(forcing.schedule)
                or len(edges) != 2 or float(edges[0]) != forcing.t0_s or float(edges[-1]) != forcing.end_s or event.rainfall_end_s != forcing.end_s
                or pend.event_sha256 is None or parent.payload.event_sha256 != pend.event_sha256):
            raise rc.RechargeError("the completed storm is not the occurrence's forced event (forcing/event identity mismatch)")
    except rc.RechargeError as exc:
        raise _wrap("occurrence", exc) from None
    fb = wl._count(retained_feedback_bytes, "retained_feedback_bytes")
    try:
        need = 0 if lop._cuda is not None else required_finish_daily_bytes(lop, parent, fb)
        wl._admit(lop, need, "daily_finish", parent=parent, feedback=fb)  # whole call, BEFORE the entry
        consumed = cursor.consume(plan, pend.branch, parent.digest)  # PURE staging
        with wl._daily_engine(parent, lambda payload: consumed):  # the genuine entry on the TRUE completed-storm parent
            ie = wl.enter_interevent_phase(lop, parent, expected_parent=expected_parent)
    except DailyRefusal:
        raise
    except (wl.LifecycleError, iw.InterEventWaterError, it.GeneralIntereventError, ls.LayeredStormError, ss.SubsurfaceError, rc.RechargeError, MemoryError) as exc:
        raise _wrap("memory" if isinstance(exc, MemoryError) else "entry", exc) from None
    gop, state0 = ie.binding.gop, ie.payload
    t_start = float(state0.t_s)
    try:
        cand = it.general_interevent_water_step(gop, state0, pet_terrain_m_s, DAY_S, phase="interevent", retained_feedback_bytes=fb)
    except HydraulicStepRejected as exc:
        raise _wrap("router_step_rejected", exc) from None
    except (it.GeneralIntereventError, iw.InterEventWaterError, ls.LayeredStormError, ss.SubsurfaceError, MemoryError) as exc:
        raise _wrap("memory" if isinstance(exc, MemoryError) else "step_refused", exc) from None
    try:
        it.audit_general_interevent(gop, cand.state)
        new = wl._fold_interevent(lop, ie, cand.state, daily=consumed)
        wl._audit_core(lop, new, wl._arrays_of(new), new.t_s)
    except (wl.LifecycleError, iw.InterEventWaterError, it.GeneralIntereventError, rc.RechargeError) as exc:
        raise _wrap("step_refused", exc) from None
    return DailyResult("complete", new, _receipt(plan, pend.branch, forcing, parent, float(event.event_origin_s), t_start, cand), None)
