"""EB1: the genuine retained-COMPLETION capability of a `PondEvent` (the producer's proof, not the display dataclass).

`PondRetainedHandoff` is a plain frozen display record anyone can construct or `replace()`. A `CompletionCapability` is different: it is minted ONLY here, in a private window, from the EXACT handoff INSTANCE and the EXACT state that
`PondEvent.settle_retaining_water` registered in its private producer registry in its `complete` branch (status "complete": real quiet gates, empty ledger, nothing mobile, published journal, closing audits). A hand-built or copied
handoff, a `requires_recession` handoff, an ordinary (`_blessed`) state, another event / config / forcing, a stale state or a consumed handoff never yields one. The capability stores the producer's own facts (config / forcing sha,
topology generation, accepted steps, clock, state identity) and never the mutable display `accounting`.

What it IS: proof that THIS event instance completed THIS state. What it is NOT: a landscape-lineage guarantee (a functional trial after a complete state neither invalidates the capability nor the accepted parent; the future lifecycle
binds the expected ADOPTED parent separately), a physical audit (the future adoption repeats the full physical checks), a copy of any array or deep bed (it holds references only; arrays follow the existing immutability convention), a
restart artefact or a property refresh.

Single consumption: `consume_completion` marks the HANDOFF consumed in the producing event's private weak set (so several capabilities requested for one handoff cannot bypass it); entries disappear with the handoff. Memory: O(1)
host metadata per completion; the registries are weak (no strong key -> value -> key cycles: the facts hold a weak reference to the state and never the handoff).
"""
from __future__ import annotations

import contextlib
import threading
import weakref
from dataclasses import dataclass
from typing import Any

__all__ = ["CompletionCapability", "CompletionError", "consume_completion", "issue", "verify_completion"]


class CompletionError(RuntimeError):
    """Typed refusal of the completion capability."""


_MINT = threading.local()
_CAPS: weakref.WeakSet = weakref.WeakSet()  # every genuinely minted capability (copies and direct constructions are never members)


@contextlib.contextmanager
def _minting():
    _MINT.ok = True
    try:
        yield
    finally:
        _MINT.ok = False


@dataclass(frozen=True, eq=False)
class CompletionFacts:
    """The producer's private facts, recorded at registration (host scalars; the state only through a weak reference)."""

    state_ref: Any  # weakref.ref(state): never a strong reference (no registry cycle)
    config_sha256: str
    forcing_sha256: str
    topology_generation: int
    n_accepted_steps: int
    t_s: float
    ledger_ref: Any  # weakref.ref(state.sediment)


@dataclass(frozen=True, eq=False)
class CompletionCapability:
    """Opaque, uncopyable, unpicklable producer capability (see the module docstring). Minted only by `PondEvent.completion_capability`."""

    event: Any
    handoff: Any
    state: Any
    facts: CompletionFacts

    def __post_init__(self) -> None:
        if not getattr(_MINT, "ok", False):
            raise CompletionError("a CompletionCapability is minted only by PondEvent.completion_capability (no direct construction)")
        _CAPS.add(self)

    def __copy__(self):
        raise CompletionError("a CompletionCapability cannot be copied")

    def __deepcopy__(self, memo):
        raise CompletionError("a CompletionCapability cannot be copied")

    def __reduce_ex__(self, protocol):
        raise CompletionError("a CompletionCapability is not serialisable")


def _admit_producer(event: Any, handoff: Any, state: Any, facts: Any) -> None:
    """The ADMISSION that precedes every issue and every use: the actual native types (exact `PondEvent`, `PondRetainedHandoff`, `PondEventState`, `CompletionFacts`; no duck types, subclasses or stand-in producers) and the producer's OWN
    registry (a genuine `WeakKeyDictionary`/`WeakSet` pair) holding exactly this facts object for exactly this handoff instance, whose state is exactly this state and whose recorded scalars still equal the state's. A private thread-local
    window alone is never admission: every refusable check runs BEFORE the window opens."""
    from maple_syrup.pond_event import PondEvent, PondEventState, PondRetainedHandoff

    if type(event) is not PondEvent or type(handoff) is not PondRetainedHandoff or type(state) is not PondEventState or type(facts) is not CompletionFacts:
        raise CompletionError("completion needs the actual native PondEvent, PondRetainedHandoff, PondEventState and producer facts (no stand-in producer or duck types)")
    reg, used = getattr(event, "_completions", None), getattr(event, "_completions_consumed", None)
    if type(reg) is not weakref.WeakKeyDictionary or type(used) is not weakref.WeakSet:
        raise CompletionError("the event's producer registries are not the native weak registries")
    if reg.get(handoff) is not facts:
        raise CompletionError("the producer did not register this handoff with these facts (a hand-built or copied handoff, or a requires_recession/ordinary state, is not a completion)")
    f = facts
    if (handoff.status != "complete" or handoff.state is not state or f.state_ref() is not state or state.status != "wet" or state.config_sha256 != f.config_sha256 or state.forcing_sha256 != f.forcing_sha256
            or event.config_sha256 != f.config_sha256 or event.forcing_sha256 != f.forcing_sha256 or int(state.topology.generation) != f.topology_generation
            or int(state.counters.n_accepted_steps) != f.n_accepted_steps or float(state.t_s) != f.t_s or f.ledger_ref() is not state.sediment):
        raise CompletionError("the handoff/state pair differs from the completion the producer recorded (event/config/forcing/topology/steps/clock/ledger)")
    if handoff in used:
        raise CompletionError("this completion was already consumed")


def mint(event: Any, handoff: Any, state: Any, facts: Any) -> CompletionCapability:
    """Compatibility name of the earlier public mint: it now ADMITS first. `facts` is only compared with the producer's own registry entry (it grants nothing); a stand-in producer, an unregistered handoff or foreign facts raise `CompletionError`."""
    _admit_producer(event, handoff, state, facts)
    return issue(event, handoff, state)


def issue(event: Any, handoff: Any, state: Any) -> CompletionCapability:
    """Issue a capability for a completion the producer registered (called by `PondEvent.completion_capability`; also callable by anyone: it takes no caller authority, only the objects, and admits them against the producer's own
    registry first). The facts are READ from that registry, never supplied by the caller."""
    reg = getattr(event, "_completions", None)
    facts = reg.get(handoff) if type(reg) is weakref.WeakKeyDictionary and type(handoff).__name__ == "PondRetainedHandoff" else None
    _admit_producer(event, handoff, state, facts)
    with _minting():  # nothing refusable (no reader, hash or callback) runs inside the window
        return CompletionCapability(event, handoff, state, facts)


def verify_completion(cap: Any, state: Any, *, event: Any) -> CompletionCapability:
    """Non-consuming verification at a boundary: exact genuine capability of THIS event for THIS state, still registered with the producer, unconsumed, facts unchanged. Raises `CompletionError`; changes nothing."""
    if type(cap) is not CompletionCapability or cap not in _CAPS:
        raise CompletionError("not a genuine CompletionCapability (copies, replacements and caller-built objects carry no authority)")
    if cap.event is not event:
        raise CompletionError("the capability was issued by another PondEvent")
    if cap.state is not state:
        raise CompletionError("the capability is bound to a different state instance")
    _admit_producer(event, cap.handoff, state, cap.facts)  # re-admission at every use (also refuses consumed handoffs)
    return cap


def consume_completion(cap: Any, state: Any, *, event: Any) -> None:
    """The ONE consumption mark (the lifecycle's LAST step after every refusable check): verifies, then marks the handoff consumed. A second request or capability for the same handoff is refused."""
    verify_completion(cap, state, event=event)
    event._completions_consumed.add(cap.handoff)
