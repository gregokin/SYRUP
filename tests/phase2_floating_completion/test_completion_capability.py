"""EB1: the genuine retained-completion capability (producer completeness ONLY: no property refresh, adoption, restart, daily or wind). Real wet/splash/quiet/settlement producer on the closed thin lake: Dense and published Windowed,
NumPy and Numba."""
from __future__ import annotations

import copy
import dataclasses
import gc
import pickle
import weakref

import pytest
from test_terminal import lake, settled

from maple_syrup import floating_completion as fcm
from maple_syrup.pond_event import PondRetainedHandoff


@pytest.fixture(params=["numpy", "numba"])
def backend(request):
    if request.param == "numba":
        pytest.importorskip("numba")
    return request.param


def complete(ev, st):
    s = settled(ev, st)
    q = ev.evolve(s, 120.0, stop_when_quiet=True)
    assert q.quiet_ready
    h = ev.settle_retaining_water(q.state)
    assert h.complete
    return h


@pytest.mark.parametrize("windowed", [False, True])
def test_genuine_complete_issues_a_capability_bundle_and_single_consumption(tmp_path, backend, windowed):
    ev, st, vert = lake(tmp_path / "a", backend, windowed)
    h = complete(ev, st)
    cap = ev.completion_capability(h, h.state)
    assert fcm.verify_completion(cap, h.state, event=ev) is cap
    b = vert.event_end_bundle(cap, h.state, ev)
    p = vert.current_payload(h.state.hydraulic.soil_water_m)
    assert b.payload is p and b.depth_m is h.state.hydraulic.depth_m and b.bed is h.state.bed and b.capability is cap  # references, no copies
    assert len(vert._registry) <= 3  # the bundle created no registry entry
    cap2 = ev.completion_capability(h, h.state)  # a second request for the same handoff...
    fcm.consume_completion(cap, h.state, event=ev)
    for c in (cap, cap2):  # ...cannot bypass the single consumption mark
        with pytest.raises(fcm.CompletionError, match="consumed"):
            fcm.verify_completion(c, h.state, event=ev)
    with pytest.raises(fcm.CompletionError, match="consumed"):
        ev.completion_capability(h, h.state)


def test_forged_copied_wrong_and_noncomplete_proofs_are_refused(tmp_path):
    ev, st, vert = lake(tmp_path / "a", "numpy", False)
    first = ev.settle_retaining_water(ev.evolve(st, 60.0, stop_when_quiet=True).state)
    assert first.status == "requires_recession"
    with pytest.raises(fcm.CompletionError):
        ev.completion_capability(first, first.state)  # requires_recession is never a completion
    h = complete(ev, st)
    forged = PondRetainedHandoff(h.state, "complete", h.terminal_deposition_kg, h.accounting, h.config_sha256, h.forcing_sha256, h.t_s, h.topology_generation)  # hand-built around the REAL complete state
    copied = dataclasses.replace(h)
    for bad in (forged, copied):
        with pytest.raises(fcm.CompletionError):
            ev.completion_capability(bad, h.state)
    with pytest.raises(fcm.CompletionError):
        ev.completion_capability(object(), h.state)
    with pytest.raises(fcm.CompletionError, match="pair"):
        ev.completion_capability(h, first.state)  # wrong state for the genuine handoff
    plain = ev.step(h.state, 0.0, 0.2).state  # an ordinary blessed state
    with pytest.raises(fcm.CompletionError):
        ev.completion_capability(copy.copy(h), plain)
    other, _st2, _v2 = lake(tmp_path / "b", "numpy", False)
    with pytest.raises(fcm.CompletionError):
        other.completion_capability(h, h.state)  # another event never registered it
    cap = ev.completion_capability(h, h.state)
    with pytest.raises(fcm.CompletionError, match="another PondEvent"):
        fcm.verify_completion(cap, h.state, event=other)
    with pytest.raises(fcm.CompletionError, match="different state"):
        fcm.verify_completion(cap, plain, event=ev)
    for op in (copy.copy, copy.deepcopy, pickle.dumps, dataclasses.replace):
        with pytest.raises((fcm.CompletionError, TypeError)):
            op(cap)
    with pytest.raises(fcm.CompletionError, match="minted"):
        fcm.CompletionCapability(ev, h, h.state, cap.facts)
    with pytest.raises(fcm.CompletionError):
        fcm.verify_completion(object(), h.state, event=ev)
    with pytest.raises(fcm.CompletionError):
        vert.event_end_bundle(cap, h.state, other)  # an event that does not run this vertical


def test_functional_trial_after_complete_never_invalidates_the_accepted_capability(tmp_path):
    ev, st, _vert = lake(tmp_path / "a", "numpy", False)
    h = complete(ev, st)
    cap = ev.completion_capability(h, h.state)
    trial = ev.step(h.state, 0.0, 0.2).state  # an unadopted functional trial from the complete state
    assert trial is not h.state
    fcm.verify_completion(cap, h.state, event=ev)  # no global "latest" head: the accepted pair still verifies
    cap_b = ev.completion_capability(h, h.state)  # and a second genuine capability for the same pair is still available until consumption
    assert cap_b is not cap
    fcm.consume_completion(cap_b, h.state, event=ev)
    with pytest.raises(fcm.CompletionError, match="consumed"):
        fcm.verify_completion(cap, h.state, event=ev)  # adopting one stales the other (the lifecycle's job to bind the lineage)


def test_failed_consumption_leaves_everything_unconsumed_and_retry_works(tmp_path):
    ev, st, _v = lake(tmp_path / "a", "numpy", False)
    h = complete(ev, st)
    cap = ev.completion_capability(h, h.state)
    plain = ev.step(h.state, 0.0, 0.2).state
    with pytest.raises(fcm.CompletionError):
        fcm.consume_completion(cap, plain, event=ev)
    assert h not in ev._completions_consumed
    fcm.consume_completion(cap, h.state, event=ev)  # retry with the right pair
    assert h in ev._completions_consumed


def test_display_accounting_mutation_grants_no_authority_and_facts_are_producer_private(tmp_path):
    ev, st, _v = lake(tmp_path / "a", "numpy", False)
    h = complete(ev, st)
    cap = ev.completion_capability(h, h.state)
    before = (cap.facts.config_sha256, cap.facts.forcing_sha256, cap.facts.topology_generation, cap.facts.n_accepted_steps, cap.facts.t_s)
    h.accounting["final_audit"] = {"water": {"closes": False}}  # the dict is mutable display data (declared honestly): editing it changes nothing here...
    h.accounting["policy"] = "edited"
    fcm.verify_completion(cap, h.state, event=ev)
    assert before == (cap.facts.config_sha256, cap.facts.forcing_sha256, cap.facts.topology_generation, cap.facts.n_accepted_steps, cap.facts.t_s)
    assert not hasattr(cap, "accounting") and "accounting" not in {f.name for f in dataclasses.fields(fcm.CompletionFacts)}
    with pytest.raises(dataclasses.FrozenInstanceError):
        h.status = "complete"  # ...and the display record's fields are frozen


def test_weak_registries_release_the_state_and_bed_with_the_handoff(tmp_path):
    ev, st, _v = lake(tmp_path / "a", "numpy", False)
    h = complete(ev, st)
    cap = ev.completion_capability(h, h.state)
    fcm.consume_completion(cap, h.state, event=ev)
    assert len(ev._completions) == 1 and len(ev._completions_consumed) == 1
    state_ref, bed_ref = weakref.ref(h.state), weakref.ref(h.state.bed)
    del cap, h, st
    for _ in range(3):
        gc.collect()
    assert len(ev._completions) == 0 and len(ev._completions_consumed) == 0  # entries vanish with the handoff: no unbounded identity registry
    assert state_ref() is None and bed_ref() is None  # and nothing retained the state or its bed (no strong key -> value -> key cycle)


def _fake_producer(ev, vert, h, facts):
    from types import SimpleNamespace

    return SimpleNamespace(vertical=vert, config_sha256=ev.config_sha256, forcing_sha256=ev.forcing_sha256, _completions={h: facts}, _completions_consumed=set())


def _forged(st):
    h = PondRetainedHandoff(st, "complete", None, {}, st.config_sha256, st.forcing_sha256, st.t_s, st.topology.generation)
    facts = fcm.CompletionFacts(weakref.ref(st), st.config_sha256, st.forcing_sha256, int(st.topology.generation), int(st.counters.n_accepted_steps), float(st.t_s), weakref.ref(st.sediment))
    return h, facts


def test_public_mint_with_a_fake_producer_is_refused_everywhere_root_reproducer(tmp_path):
    """Root's reproducer: a REAL incomplete step-0 state, a forged handoff + facts and a SimpleNamespace producer with its OWN registries. Nothing may issue, verify or produce a bundle; the real event registers nothing."""
    ev, st, vert = lake(tmp_path / "a", "numpy", False)
    h, facts = _forged(st)
    fake = _fake_producer(ev, vert, h, facts)
    with pytest.raises(fcm.CompletionError, match="native"):
        fcm.mint(fake, h, st, facts)  # Root's exact call: the public mint now admits first
    with pytest.raises(fcm.CompletionError, match="native"):
        fcm.issue(fake, h, st)
    with fcm._minting():  # even an explicit, caller-opened private window cannot build a USABLE capability: every use re-admits
        cap = fcm.CompletionCapability(fake, h, st, facts)
    with pytest.raises(fcm.CompletionError, match="native"):
        fcm.verify_completion(cap, st, event=fake)
    with pytest.raises(fcm.CompletionError):
        vert.event_end_bundle(cap, st, fake)
    with pytest.raises(fcm.CompletionError):
        fcm.consume_completion(cap, st, event=fake)
    assert len(ev._completions) == 0 and st.counters.n_accepted_steps == 0 and not fake._completions_consumed


def test_real_event_with_unregistered_or_wrong_objects_is_refused_even_through_the_public_issue(tmp_path):
    ev, st, vert = lake(tmp_path / "a", "numpy", False)
    h, facts = _forged(st)
    for args in ((ev, h, st), (ev, object(), st), (ev, h, object()), (object(), h, st)):
        with pytest.raises(fcm.CompletionError):
            fcm.issue(*args)
    with fcm._minting():  # a hand-built capability around the REAL event is refused at use: the producer never registered the handoff
        cap = fcm.CompletionCapability(ev, h, st, facts)
    with pytest.raises(fcm.CompletionError, match="did not register"):
        fcm.verify_completion(cap, st, event=ev)
    with pytest.raises(fcm.CompletionError):
        vert.event_end_bundle(cap, st, ev)
    # a duck-typed registry on the REAL event's class is not admission either: wrong facts / wrong native types
    h2 = complete(ev, st)
    good = ev.completion_capability(h2, h2.state)
    wrong = dataclasses.replace(good.facts, n_accepted_steps=good.facts.n_accepted_steps + 1)
    with pytest.raises(fcm.CompletionError):
        fcm._admit_producer(ev, h2, h2.state, wrong)
    with pytest.raises(fcm.CompletionError):
        fcm._admit_producer(ev, h2, h2.state, object())
    fcm.verify_completion(good, h2.state, event=ev)  # the genuine path is unaffected
