"""Pure physical INTEREVENT INTERVAL evolver over a constant, caller-declared terrain-applied PET rate (Phase 1, task
phase1_interevent_interval_20261110).

SYRUP owns the water physics; this module only iterates the EXISTING pure `interevent_water.interevent_water_step` over ONE
caller-declared native-time interval `[start_s, end_s]` during which the terrain-applied PET rate is constant, retries
only the genuine late-CFL rejection from the SAME accepted water, preserves the numerical step cap, and returns bounded,
honest, whole accepted progress:

    op      = interevent_water.prepare_interevent_water(...)                  # the actual final operator
    st0     = interevent_water.enter_interevent(op, event, completed)         # genuine completed storm, entered ONCE
    b1      = prepare_interval_forcing(op, pet_1, start_s=t0, end_s=e1, identity="piece-1", max_dt_s=..., min_dt_s=...)
    prog    = start_interval(b1, st0)                                         # audited cold start
    out     = evolve_interval(b1, prog, max_accepted_steps=..., max_rejected_attempts=...)
    prog    = out.progress                                                    # the caller adopts the whole progress
    b2      = prepare_interval_forcing(op, pet_2, start_s=e1, end_s=e2, ...)  # next forcing edge (any calendar is the caller's)
    prog    = carry_interval(b2, prog)                                        # typed transition: same water, carried cap

There is NO calendar, date, weather, MASY, LEAF or scheduler concept here; the interval edges are numbers on the existing
native clock, and `identity` is an opaque bounded caller binding (NOT accepted weather/pipeline provenance). The water
state, the source receipt, the layers, H/faces, the persistent J/lifetime intake and the window ledger continue across
pieces; `enter_interevent` is never called again and no history is reset.

An interval is COMPLETE only at its exact `end_s`, never at hydraulic quiet: soil/pond evaporation, intake, drainage and
redistribution continue on a quiet surface. The full surface-aware step runs every time (no dry shortcut). Nothing here was
run on a GPU by its author; Codex records executions.
"""

from __future__ import annotations

import hashlib
import math
import weakref
from dataclasses import dataclass
from typing import Any

import numpy as np

from maple_syrup import interevent_water as iw
from maple_syrup import layered_event as ev
from maple_syrup import layered_storm as ls
from maple_syrup import subsurface as ss
from maple_syrup.surface_routing import HydraulicStepRejected

__all__ = [
    "MEMORY_SCOPE",
    "REASONS",
    "HydraulicStepRejected",
    "IntervalBinding",
    "IntervalError",
    "IntervalMemory",
    "IntervalProgress",
    "IntervalResult",
    "IntervalTimeError",
    "carry_interval",
    "describe",
    "estimate_interval_memory",
    "evolve_interval",
    "prepare_interval_forcing",
    "start_interval",
]

REASONS = ("complete", "target", "work_limit", "retry_limit", "min_dt")
MAX_IDENTITY_CHARS = 256
DEFAULT_MAX_RETRIES_PER_STEP = 12
DEFAULT_MAX_STEPS_CEILING = 100_000
_SEAL = object()
_MINTED = weakref.WeakSet()  # progress records minted by this module (in-memory authenticity only)

MEMORY_SCOPE = (
    "Interval ledger under the EXISTING interevent/column convention. STEP (every evolve call, admitted before any state "
    "check or trial against the column's available_bytes_fn on its admitted side): the interevent step's own required_bytes "
    "(accepted + static + ONE trial + reserves) + the caller's extra reserve + one object allowance for progress/host "
    "metadata. The owned PET vector is passed DIRECTLY to the step, so it IS the step's `pet_input` trial allowance and is "
    "counted once (declared alias); there is no per-step PET clone, hash, host download or preparation, and no timestep "
    "history (only the current rate and the accepted + one trial fixed resident state). PREPARE (cold, admitted before "
    "any allocation or read): the step requirement + the owned PET copy + three N-byte validity masks + one object allowance "
    "(+ the caller's extra reserve); CUDA additionally admits the bounded host hash staging min(HASH_CHUNK_BYTES, 8 N) + the "
    "caller's host reserve separately against its host reader. A column memory reading is a whole-owner quota, never extra "
    "free memory. EXCLUDED: allocator pools, CUDA context/NVRTC/Numba JIT workspaces, the caller's own source PET array and "
    "weather/forcing pipeline, native voxel/ledger material outside the water owner, checkpoint staging, LEAF and the "
    "remaining process memory. No whole-process fit is claimed."
)


class IntervalError(iw.InterEventWaterError):
    """Invalid binding, progress, interval, argument or continuation. Raised before any result exists."""


class IntervalTimeError(IntervalError):
    """The requested boundary cannot be reached by a genuine representable `t + dt` progression."""


def _real(value: Any, name: str) -> float:
    try:
        return iw._real(value, name)
    except iw.InterEventWaterError as exc:
        raise IntervalError(str(exc)) from None


def _positive(value: Any, name: str) -> float:
    out = _real(value, name)
    if not out > 0.0:
        raise IntervalError(f"{name} must be > 0, got {out!r}")
    return out


#: Fixed implementation-supported range of every count control (signed 64-bit, like the other profiles).
COUNT_MAX = 2**63 - 1


def _count(value: Any, name: str, *, minimum: int, ceiling: int | None = None) -> int:
    """A canonical Python int in `[minimum, min(ceiling, COUNT_MAX)]`. Booleans/foreign types are refused; the range is judged
    from `bit_length()` so no message ever formats the supplied integer (a huge value cannot raise a raw ValueError)."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise IntervalError(f"{name} must be a non-Boolean integer, got type {type(value).__name__}")
    out = int(value)  # normalizes a genuine NumPy integer; never formatted below
    if out < 0 and minimum >= 0 or out.bit_length() > 63:
        raise IntervalError(f"{name} is outside the supported count range [{minimum}, {COUNT_MAX}] "
                            f"(type {type(value).__name__}, {out.bit_length()} bits)")
    if out < minimum:
        raise IntervalError(f"{name} must be >= {minimum} (got a {out.bit_length()}-bit value)")
    if ceiling is not None and out > ceiling:
        raise IntervalError(f"{name} exceeds the prepared ceiling {ceiling}")
    return out


def _identity(value: Any) -> str:
    """Strict bounded opaque caller binding, validated BEFORE any serialization or copy."""
    if type(value) is not str or not 1 <= len(value) <= MAX_IDENTITY_CHARS or any(not 0x20 <= ord(c) <= 0x7E for c in value):
        raise IntervalError(f"identity must be a str of 1..{MAX_IDENTITY_CHARS} printable ASCII characters")
    return value


# --- memory ------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class IntervalMemory:
    """Pure arithmetic over the interevent operator's own estimate (see MEMORY_SCOPE)."""

    backend: str
    n_cells: int
    step_required_bytes: int  # the interevent step's required bytes (contains ONE PET-sized trial input)
    owned_pet_bytes: int  # the owned rate; ALIASED to that trial input, so not added again
    progress_allowance_bytes: int
    extra_reserve_bytes: int
    required_step_bytes: int  # admitted by every evolve call
    build_bytes: int  # cold owned copy + validity masks + object allowance
    build_host_bytes: int  # CUDA hash staging (0 on CPU: shares the quota)
    required_prepare_bytes: int
    pet_aliased_to_step_input: bool = True
    scope: str = MEMORY_SCOPE

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def estimate_interval_memory(prep: iw.InterEventWaterOperator, *, extra_reserve_bytes: int = 0) -> IntervalMemory:
    extra = ss._checked_int(extra_reserve_bytes, "extra_reserve_bytes")
    n = prep.n_cells
    step = prep.memory.required_bytes
    build = n * 8 + iw.COLD_MASK_BYTES_PER_CELL * n + iw.OBJECT_BYTES
    host = min(ls.HASH_CHUNK_BYTES, n * 8) if prep.backend == "cuda" else 0
    return IntervalMemory(prep.backend, n, step, n * 8, iw.OBJECT_BYTES, extra, step + extra + iw.OBJECT_BYTES, build, host,
                          step + extra + build)


# --- binding -----------------------------------------------------------------------------------------------------------
class IntervalBinding:
    """Immutable factory-only constant-forcing interval binding minted ONLY by `prepare_interval_forcing`: the actual
    interevent operator, an OWNED `(N,)` float64 terrain-applied PET rate (NumPy: immutable bytes-backed array; CUDA:
    private device array), its cold SHA-256, the exact native `[start_s, end_s]`, the opaque caller identity, the caller's
    required `max_dt_s`/`min_dt_s`, the retry/step controls, the memory record and a binding SHA-256. Not a dataclass: no
    `dataclasses.replace`, copy or pickle bypass can change a value without the cold hash."""

    __slots__ = ("_pet", "_pet_digest", "_seal", "backend", "end_s", "extra_reserve_bytes", "identity", "max_dt_s",
                 "max_retries_per_step", "max_steps_ceiling", "memory", "min_dt_s", "n_cells", "prep", "sha256", "start_s",
                 "xp")

    def __init__(self, seal: object, **fields: Any) -> None:
        if seal is not _SEAL:
            raise IntervalError("IntervalBinding is constructed only by prepare_interval_forcing()")
        for name in self.__slots__:
            if name == "_seal":
                continue
            if name not in fields:
                raise IntervalError(f"internal: binding field {name} missing")
            object.__setattr__(self, name, fields.pop(name))
        if fields:
            raise IntervalError(f"internal: unexpected binding fields {sorted(fields)}")
        object.__setattr__(self, "_seal", seal)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("IntervalBinding is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("IntervalBinding is immutable")

    def __reduce__(self) -> Any:
        raise TypeError("IntervalBinding cannot be copied or pickled; prepare a new binding")

    def __repr__(self) -> str:
        return (f"IntervalBinding(identity={self.identity!r}, start_s={self.start_s!r}, end_s={self.end_s!r}, "
                f"sha256={self.sha256[:12]}...)")

    @property
    def pet_rate(self) -> Any:
        """The owned rate: the immutable host array, or on a device a fresh DETACHED copy (admitted first)."""
        if self.xp is np:
            return self._pet
        _check_binding(self)
        iw._admit(self.prep.op, self.memory.required_step_bytes + self.n_cells * 8, "PET inspection")
        return self._pet.copy()

    def describe(self) -> dict[str, Any]:
        return describe(self)


def _check_binding(binding: Any) -> None:
    """Metadata-only revalidation (every call, before any allocation): sealed record, the operator (CUDA: the context
    preflight) and the owned rate's type/dtype/shape/contiguity/device. No element is read."""
    if type(binding) is not IntervalBinding or getattr(binding, "_seal", None) is not _SEAL:
        raise IntervalError(f"binding must be a genuine IntervalBinding, got {type(binding).__name__}")
    prep = binding.prep
    iw._check_prepared(prep)
    if binding.xp is not prep.xp or binding.backend != prep.backend or binding.n_cells != prep.n_cells:
        raise IntervalError("binding disagrees with its interevent operator; refused")
    try:
        ss._check_dynamic(prep.op.column, "owned PET rate", binding._pet, (prep.n_cells,))
    except ss.SubsurfaceError as exc:
        raise IntervalError(str(exc)) from None
    if binding.xp is np and binding._pet.flags.writeable:
        raise IntervalError("the owned host PET rate lost its immutability; refused")
    if (not isinstance(binding.sha256, str) or len(binding.sha256) != 64 or not isinstance(binding._pet_digest, str)
            or not binding.start_s >= 0.0 or not binding.end_s > binding.start_s):
        raise IntervalError("binding carries a replaced identity or interval; refused")


def prepare_interval_forcing(prep: iw.InterEventWaterOperator, pet_terrain_m_s: Any, *, start_s: Any, end_s: Any,
                             identity: Any, max_dt_s: Any, min_dt_s: Any,
                             max_retries_per_step: int = DEFAULT_MAX_RETRIES_PER_STEP,
                             max_steps_ceiling: int = DEFAULT_MAX_STEPS_CEILING, extra_reserve_bytes: int = 0,
                             host_available_bytes_fn: Any = None, host_reserve_bytes: int = 0) -> IntervalBinding:
    """Bind the actual operator and ONE constant terrain-applied PET `(N,)` (m/s, finite, >= 0, already containing aspect
    and cos(slope); copied once and owned) valid over the native interval `[start_s, end_s]`. `max_dt_s` (the hydraulic /
    column step ceiling) and `min_dt_s` (the refinement floor) are REQUIRED: there is no default daily step. Every scalar,
    metadata, device and quota check precedes any allocation, download or hash; the validity read is one counted packet
    and the cold hash streams in bounded chunks. `identity` is an opaque bounded caller binding only."""
    iw._check_prepared(prep)  # genuine operator + CUDA context preflight: metadata only
    op = prep.op
    try:
        ss._check_dynamic(op.column, "pet_terrain_m_s", pet_terrain_m_s, (prep.n_cells,))
    except ss.SubsurfaceError as exc:
        raise IntervalError(str(exc)) from None
    ident = _identity(identity)
    start, end = _real(start_s, "start_s"), _real(end_s, "end_s")
    if start < 0.0 or not end > start:
        raise IntervalError(f"need 0 <= start_s < end_s, got {start!r}, {end!r}")
    hi, lo = _positive(max_dt_s, "max_dt_s"), _positive(min_dt_s, "min_dt_s")
    if lo > hi:
        raise IntervalError("min_dt_s exceeds max_dt_s")
    retries = _count(max_retries_per_step, "max_retries_per_step", minimum=1)
    ceiling = _count(max_steps_ceiling, "max_steps_ceiling", minimum=1)
    extra = ss._checked_int(extra_reserve_bytes, "extra_reserve_bytes")
    host_reserve = ss._checked_int(host_reserve_bytes, "host_reserve_bytes")
    if host_available_bytes_fn is not None and not callable(host_available_bytes_fn):
        raise IntervalError("host_available_bytes_fn must be callable or None")
    if prep.backend != "cuda" and (host_available_bytes_fn is not None or host_reserve != 0):
        raise IntervalError("separate host staging controls are CUDA-only; CPU uses the column quota")
    memory = estimate_interval_memory(prep, extra_reserve_bytes=extra)
    iw._admit(op, memory.required_prepare_bytes, "interval preparation")  # BEFORE any allocation/read/hash
    if prep.backend == "cuda":
        try:
            ls._admit_host_build(memory.build_host_bytes + host_reserve + iw.OBJECT_BYTES,
                                 host_available_bytes_fn or ss.default_host_available_bytes)
        except ls.LayeredMemoryError as exc:
            raise iw.InterEventMemoryError(str(exc)) from None
    xp = prep.xp
    from maple.core.backend import errstate

    with errstate(xp=xp, all="ignore"):
        bad = xp.count_nonzero(~xp.isfinite(pet_terrain_m_s) | (pet_terrain_m_s < 0.0))
    if ls._packet(xp, [bad])[0] != 0.0:
        raise IntervalError("pet_terrain_m_s must be finite and >= 0 on every cell")
    if xp is np:
        owned = ev._immutable_host(np.asarray(pet_terrain_m_s, dtype=np.float64))
    else:
        owned = xp.ascontiguousarray(xp.array(pet_terrain_m_s, dtype=np.float64, copy=True))
    h = hashlib.sha256(b"maple_syrup.interevent_interval.pet/1")
    ls._stream_hash(h, owned)
    pet_digest = h.hexdigest()
    g = hashlib.sha256(b"maple_syrup.interevent_interval.binding/1")
    for piece in (prep.sha256, pet_digest, float(start).hex(), float(end).hex(), ident, float(hi).hex(), float(lo).hex(),
                  str(retries), str(ceiling), str(extra)):
        g.update(piece.encode())
        g.update(b"\0")
    return IntervalBinding(_SEAL, prep=prep, _pet=owned, _pet_digest=pet_digest, start_s=start, end_s=end, identity=ident,
                           max_dt_s=hi, min_dt_s=lo, max_retries_per_step=retries, max_steps_ceiling=ceiling,
                           extra_reserve_bytes=extra, memory=memory, sha256=g.hexdigest(), backend=prep.backend,
                           n_cells=prep.n_cells, xp=prep.xp)


# --- progress ------------------------------------------------------------------------------------------------------------
class IntervalProgress:
    """One whole accepted position inside (or at an edge of) a bound interval: the genuine `InterEventState` (the current
    time is `state.t_s`, nothing else), the binding it belongs to, the persistent next numerical step cap, the accepted-step
    and rejected-attempt counts of THIS piece `pending_rejection` (the cap was lowered by a CFL rejection that no accepted step has confirmed yet, so the next accepted
    step must not grow it: this makes a stop/resume sequence reproduce the one-shot mesh) and `completed` (exactly
    `state.t_s == end_s`). Factory-only, immutable and
    registered at minting: it cannot be built, edited or `dataclasses.replace`d with a forged flag. In-memory authenticity
    only: not a codec, CAS or durable publication."""

    __slots__ = ("__weakref__", "accepted_steps", "binding_sha256", "completed", "next_dt_cap_s", "pending_rejection",
                 "rejected_attempts", "state")

    def __init__(self, seal: object, **fields: Any) -> None:
        if seal is not _SEAL:
            raise IntervalError("IntervalProgress is minted only by start_interval/carry_interval/evolve_interval")
        for name in self.__slots__:
            if name == "__weakref__":
                continue
            if name not in fields:
                raise IntervalError(f"internal: progress field {name} missing")
            object.__setattr__(self, name, fields.pop(name))
        if fields:
            raise IntervalError(f"internal: unexpected progress fields {sorted(fields)}")
        _MINTED.add(self)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("IntervalProgress is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("IntervalProgress is immutable")

    def __reduce__(self) -> Any:
        raise TypeError("IntervalProgress cannot be copied or pickled")

    def __repr__(self) -> str:
        return (f"IntervalProgress(t_s={self.state.t_s!r}, cap={self.next_dt_cap_s!r}, accepted={self.accepted_steps}, "
                f"rejected={self.rejected_attempts}, completed={self.completed})")


def _mint(binding: IntervalBinding, state: iw.InterEventState, cap: float, accepted: int, rejected: int,
          pending: bool = False) -> IntervalProgress:
    return IntervalProgress(_SEAL, state=state, binding_sha256=binding.sha256, next_dt_cap_s=float(cap),
                            accepted_steps=int(accepted), rejected_attempts=int(rejected), pending_rejection=bool(pending),
                            completed=bool(state.t_s == binding.end_s))


def _check_cap(binding: IntervalBinding, cap: float, name: str) -> float:
    if not (binding.min_dt_s <= cap <= binding.max_dt_s):
        raise IntervalError(f"{name} = {cap!r} must lie within [min_dt_s, max_dt_s] = "
                            f"[{binding.min_dt_s!r}, {binding.max_dt_s!r}]")
    return cap


def start_interval(binding: IntervalBinding, state: iw.InterEventState, *, initial_dt_cap_s: Any = None) -> IntervalProgress:
    """Cold start from a genuine, already-entered current water state whose clock is EXACTLY `start_s`. The actual water,
    anchored window closure and source receipt are audited (`audit_interevent`); no storm is re-entered, no ledger rewritten.
    The initial cap defaults to `max_dt_s`; an explicit one must be finite with `min_dt_s <= cap <= max_dt_s`."""
    _check_binding(binding)
    cap = binding.max_dt_s if initial_dt_cap_s is None else _check_cap(binding, _positive(initial_dt_cap_s, "initial_dt_cap_s"),
                                                                       "initial_dt_cap_s")
    iw.audit_interevent(binding.prep, state)
    if state.t_s != binding.start_s:
        raise IntervalError(f"the state's clock {state.t_s!r} is not the interval start {binding.start_s!r}; refused")
    return _mint(binding, state, cap, 0, 0)


def carry_interval(binding: IntervalBinding, previous: IntervalProgress) -> IntervalProgress:
    """Typed transition to the NEXT interval piece: `binding.start_s` must equal the previous physical clock exactly. The
    SAME water state carries on (the actual state is audited again); the cap is `min(previous cap, binding.max_dt_s)` and is
    refused, never raised, when that falls below the new `min_dt_s`. Counts restart for the new piece. The previous
    interval need not be complete, but it must have reached the new start."""
    _check_binding(binding)
    if type(previous) is not IntervalProgress or previous not in _MINTED:
        raise IntervalError("previous must be an IntervalProgress minted by this module")
    cap = min(_real(previous.next_dt_cap_s, "previous.next_dt_cap_s"), binding.max_dt_s)
    if cap < binding.min_dt_s:
        raise IntervalError(f"the carried cap {cap!r} is below this piece's min_dt_s {binding.min_dt_s!r}; declare "
                            "min_dt_s <= the carried cap (a carried cap is never raised)")
    iw.audit_interevent(binding.prep, previous.state)
    if previous.state.t_s != binding.start_s:
        raise IntervalError(f"the previous clock {previous.state.t_s!r} is not this piece's start {binding.start_s!r}; "
                            "prepare the next binding at the exact previous physical end")
    return _mint(binding, previous.state, cap, 0, 0, previous.pending_rejection)


def _check_progress(binding: IntervalBinding, progress: Any) -> float:
    if type(progress) is not IntervalProgress or progress not in _MINTED:
        raise IntervalError("progress must be an IntervalProgress minted by this module")
    if progress.binding_sha256 != binding.sha256:
        raise IntervalError("progress belongs to a different interval binding; refused")
    cap = _positive(progress.next_dt_cap_s, "progress.next_dt_cap_s")
    _check_cap(binding, cap, "progress.next_dt_cap_s")
    iw._integer(progress.accepted_steps, "progress.accepted_steps")
    iw._integer(progress.rejected_attempts, "progress.rejected_attempts")
    if type(progress.pending_rejection) is not bool:
        raise IntervalError("progress.pending_rejection must be a bool")
    t = iw._check_state(binding.prep, progress.state)  # metadata + scalar source receipt: bounded, no array read
    if not binding.start_s <= t <= binding.end_s:
        raise IntervalError(f"the accepted clock {t!r} lies outside the bound interval [{binding.start_s!r}, {binding.end_s!r}]")
    if progress.completed is not (t == binding.end_s):
        raise IntervalError("progress completion flag disagrees with the accepted clock; refused")
    return t


# --- evolve --------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class IntervalResult:
    """Bounded outcome: the last accepted WHOLE progress, why work stopped (`REASONS`), this call's accepted/rejected
    counts and the last accepted step's scalar receipt. No stage arrays, candidates or history are retained."""

    progress: IntervalProgress
    reason: str
    accepted_steps: int
    rejected_attempts: int
    last_receipt: iw.InterEventStepReceipt | None

    @property
    def complete(self) -> bool:
        return self.progress.completed


def _fragment(t: float, boundary: float, cap: float) -> float:
    """The dt with `t + dt == boundary` EXACTLY (a genuine representable progression), searched among the neighbours of
    `boundary - t`, its neighbours and `cap` itself, with `0 < dt <= cap` ALWAYS; otherwise a typed error. Never an overshoot,
    a restamp or a cap enlargement."""
    if not boundary > t:
        raise IntervalTimeError(f"the boundary {boundary!r} is not ahead of the clock {t!r}")
    d = boundary - t
    for c in (d, math.nextafter(d, 0.0), math.nextafter(d, math.inf), cap):
        if 0.0 < c <= cap and t + c == boundary:
            return c
    raise IntervalTimeError(f"the boundary {boundary!r} is not reachable from t = {t!r} by a representable t + dt with "
                            f"0 < dt <= the step cap {cap!r}; the cap is never enlarged")


def evolve_interval(binding: IntervalBinding, progress: IntervalProgress, until_s: Any = None, *, max_accepted_steps: Any,
                    max_rejected_attempts: Any) -> IntervalResult:
    """Advance bounded work from the accepted `progress` toward `until_s` (default: the exact interval end). Every trial is
    ONE `interevent_water_step(..., phase="interevent")` with the owned constant PET; only `HydraulicStepRejected` is
    recoverable (all trial intake/E/history discarded, dt halved from the SAME accepted water, every stage rerun). Every other
    error propagates typed (nonrecoverable) with the last whole accepted progress attached as `exc.interval_progress`.
    Inputs are never written; the caller adopts `result.progress` whole."""
    _check_binding(binding)
    steps_budget = _count(max_accepted_steps, "max_accepted_steps", minimum=1, ceiling=binding.max_steps_ceiling)
    rej_budget = _count(max_rejected_attempts, "max_rejected_attempts", minimum=1, ceiling=binding.max_steps_ceiling)
    iw._admit(binding.prep.op, binding.memory.required_step_bytes, "interval step")  # quota BEFORE any state check / trial
    t = _check_progress(binding, progress)
    end = binding.end_s
    target = end if until_s is None else _real(until_s, "until_s")
    if target < t:
        raise IntervalError(f"until_s = {target!r} precedes the accepted clock {t!r}")
    if target > end:
        raise IntervalError(f"until_s = {target!r} overshoots the interval end {end!r}")
    prep, pet = binding.prep, binding._pet
    min_dt, max_dt = binding.min_dt_s, binding.max_dt_s
    state, cap0 = progress.state, float(progress.next_dt_cap_s)
    pending = progress.pending_rejection
    n = rej_call = 0
    last = None

    def snapshot() -> IntervalProgress:
        if not (n or rej_call or cap0 != progress.next_dt_cap_s):
            return progress
        return _mint(binding, state, cap0, progress.accepted_steps + n, progress.rejected_attempts + rej_call, pending)

    def finish(reason: str) -> IntervalResult:
        return IntervalResult(snapshot(), reason, n, rej_call, last)

    try:
        while t < target:
            if n >= steps_budget:
                return finish("work_limit")
            remaining = target - t
            if remaining <= cap0 or t + cap0 >= target:
                dt = _fragment(t, target, min(cap0, max_dt))
            else:
                dt = cap0
            floor = min(min_dt, dt)
            attempts = 0
            while True:
                if not (0.0 < dt <= cap0 and dt <= max_dt and t + dt > t):  # BEFORE every physical trial
                    raise IntervalTimeError(f"dt = {dt!r} at t = {t!r} violates 0 < dt <= cap {cap0!r} <= max_dt_s "
                                            f"{max_dt!r} or cannot advance floating time; the cap is never enlarged")
                try:
                    candidate = iw.interevent_water_step(prep, state, pet, dt, phase=iw.INTEREVENT)
                    break
                except HydraulicStepRejected:
                    attempts += 1
                    rej_call += 1
                    cap0 = min(cap0, max(min_dt, dt / 2.0))  # the rejected refinement is never forgotten
                    pending = True
                    if attempts >= binding.max_retries_per_step or rej_call >= rej_budget:
                        return finish("retry_limit")
                    if dt / 2.0 < floor:
                        return finish("min_dt")
                    dt /= 2.0
            # accepted: persistent cap (event-owner idiom, never above the old cap except by the 2x growth after a full step)
            if attempts:
                cap = min(cap0, max(min_dt, dt))
            elif pending:
                cap = cap0  # a rejection-lowered cap has not been confirmed by an accepted step: no growth yet
            elif dt >= cap0:
                cap = min(max_dt, 2.0 * cap0)
            else:
                cap = cap0
            last = candidate.receipt
            state, cap0, t = candidate.state, cap, float(candidate.state.t_s)
            pending = False
            n += 1
            del candidate
    except Exception as exc:
        try:
            exc.interval_progress = snapshot()
        except Exception:  # an exception type that refuses attributes still propagates unchanged
            pass
        raise
    return finish("complete" if t == end else "target")


# --- provenance ----------------------------------------------------------------------------------------------------------
def describe(binding: IntervalBinding) -> dict[str, Any]:
    _check_binding(binding)
    return {
        "operation": "pure constant-PET interevent interval evolver over the existing interevent_water_step",
        "interval_native_s": [binding.start_s, binding.end_s], "identity": binding.identity,
        "identity_note": "opaque bounded caller binding; NOT acceptance of calendar/weather/pipeline provenance",
        "pet_sha256": binding._pet_digest, "binding_sha256": binding.sha256, "operator_sha256": binding.prep.sha256,
        "controls": {"max_dt_s": binding.max_dt_s, "min_dt_s": binding.min_dt_s,
                     "max_retries_per_step": binding.max_retries_per_step, "max_steps_ceiling": binding.max_steps_ceiling},
        "reasons": list(REASONS),
        "completion": "only the exact interval end; hydraulic quiet never completes it (E/intake/drainage continue)",
        "recoverable": "HydraulicStepRejected only; same accepted water, halved dt, all stages rerun",
        "memory": binding.memory.as_dict(),
        "excluded": "calendar/date/weather/MASY/LEAF, rain/wind, transpiration/growth, dry shortcut, checkpoint/codec/CAS, "
                    "GPU execution claim, whole-process memory fit",
    }
