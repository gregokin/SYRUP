"""CW-B30: outer frames (`run_complete_case`, `complete_event`) no longer pin their own start state or the loaded
continuation while the event runs; caller- and hook-owned objects are untouched.

Weak references are taken to the ACTUAL objects (the initial `SedimentEventState`, its `BedState`, ledger, voxel
column, active layer and phase; the loaded `EventProgress`, its `SedimentEventResult` and accumulators). The
observation points are real call sites of the event: the private accounting step (a later attempt, after accepted
states have replaced the start state) and the terminal transaction. Only creation/lifetime is observed; no physics
is replaced or duplicated.

CW-B30 correction 1: `evolve_sediment_event` also drops its `prev` / `continuation` references at their last use
(after the restart copies, validation and boundary concatenation), so on RESUME the loaded prior RESULT, its
accumulators and (once an accepted step has replaced it) its state are no longer alive during the loop. The only
permitted survivor is the proven current alias: the loaded state IS the solver's state until the first accepted
step, so it may be alive at the first attempt only."""
from __future__ import annotations

import gc
import weakref
from pathlib import Path

import numpy as np
import pytest
from b30_cases import KINDS, make_hook, run_options, window_options

import maple_syrup.complete_event as complete_module
import maple_syrup.event_experiment as runner_module
import maple_syrup.sediment_event as event_module
from maple_syrup.event_experiment import run_complete_case
from maple_syrup.sediment_event import SedimentEventError
from maple_syrup.sediment_transport import TransportStepRejected


def options(kind, **extra):
    return (run_options if kind == "dense" else window_options)(**extra)


def initial_refs(state0):
    bed = state0.bed
    refs = {"state0": state0, "bed": bed, "ledger": bed.ledger, "voxel_column": bed.voxel_column,
            "active_layer": bed.active_layer}
    if state0.phase is not None:
        refs["phase"] = state0.phase
    return {name: weakref.ref(obj) for name, obj in refs.items()}


def live(refs):
    gc.collect()
    return sorted(name for name, ref in refs.items() if ref() is not None)


def observe_attempts(monkeypatch, record):
    """Record, at the start of every attempt of the private loop step, which watched objects are alive."""
    real = event_module._sediment_accounting_step

    def observed(*args, **kwargs):
        record.append({key: live(refs) for key, refs in record.watch.items()})
        return real(*args, **kwargs)

    monkeypatch.setattr(event_module, "_sediment_accounting_step", observed)


class Record(list):
    def __init__(self):
        super().__init__()
        self.watch = {}


@pytest.mark.parametrize("kind", KINDS)
def test_initial_state_is_released_once_accepted_states_replace_it(kind, tmp_path, monkeypatch):
    record = Record()
    hook = make_hook(kind, tmp_path / "src", on_state0=lambda s0: record.watch.update(initial=initial_refs(s0)))
    observe_attempts(monkeypatch, record)
    run = run_complete_case(tmp_path / "src" / "case", tmp_path / "out", case_setup=hook, **options(kind))
    assert run.summary["status"] == "complete" and len(record) > 4
    # Attempt 1 starts from the initial state; after two accepted steps nothing outside the solver's own (rebound)
    # state may hold the initial state, its bed, ledger, column, layer or phase.
    assert record[0]["initial"]  # alive while it IS the solver's state
    assert all(entry["initial"] == [] for entry in record[3:]), record[3]
    assert live(record.watch["initial"]) == []


def pause_then(kind, tmp_path, **extra):
    hook = make_hook(kind, tmp_path / "src")
    paused = run_complete_case(tmp_path / "src" / "case", tmp_path / "paused", case_setup=hook,
                               **options(kind, pause_at_s=4.0))
    assert paused.summary["status"] == "paused"
    return paused.summary["last_checkpoint"]


@pytest.mark.parametrize("kind", KINDS)
def test_resume_releases_the_setup_initial_state_and_the_loaded_progress(kind, tmp_path, monkeypatch):
    checkpoint = pause_then(kind, tmp_path)
    record = Record()
    real_load = runner_module.load_checkpoint

    def load(*args, **kwargs):
        progress = real_load(*args, **kwargs)
        accumulators = {"cumulative_rain_m": progress.result.cumulative_rain_m,
                        "cumulative_pickup_kg": progress.result.cumulative_pickup_kg,
                        "hydrograph": progress.result.hydrograph}
        record.watch["progress"] = {"progress": weakref.ref(progress)}
        record.watch["result"] = {"result": weakref.ref(progress.result), "state": weakref.ref(progress.result.state),
                                  **{k: weakref.ref(v) for k, v in accumulators.items()}}
        return progress

    at_finalize = {}
    real_finalize = complete_module.finalize_event

    def finalize(*args, **kwargs):
        at_finalize.update({key: live(refs) for key, refs in record.watch.items()})
        return real_finalize(*args, **kwargs)

    monkeypatch.setattr(runner_module, "load_checkpoint", load)
    monkeypatch.setattr(complete_module, "finalize_event", finalize)
    observe_attempts(monkeypatch, record)
    hook = make_hook(kind, tmp_path / "src2", on_state0=lambda s0: record.watch.update(initial=initial_refs(s0)))
    run = run_complete_case(tmp_path / "src2" / "case", tmp_path / "resumed", case_setup=hook, resume=checkpoint,
                            **options(kind))
    assert run.summary["status"] == "complete"
    later = record[2:]
    assert later
    # The setup's dry initial state is never the event state on resume: released before the event starts.
    assert all(entry["initial"] == [] for entry in record)
    # The loaded EventProgress wrapper is held by no outer frame during the loop.
    assert all(entry["progress"] == [] for entry in later)
    # Attempt 1: the loaded state is still the solver's current state (the one permitted alias); the prior result
    # object and its accumulators are already released.
    assert set(record[0]["result"]) <= {"state"}
    # Once accepted states have replaced it, nothing of the loaded prior result is alive.
    assert all(entry["result"] == [] for entry in later), later[0]
    assert at_finalize == {"initial": [], "progress": [], "result": []}


def snapshot(obj):
    from maple.core.backend import array_leaves

    return [np.asarray(a).copy() for a in array_leaves(obj)]


def unchanged(before, obj):
    from maple.core.backend import array_leaves

    after = array_leaves(obj)
    return len(before) == len(after) and all(
        a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == np.asarray(b).tobytes()
        for a, b in zip(before, after, strict=True))


def run_with_owned_hook(kind, tmp_path, **extra):
    keep = []
    hook = make_hook(kind, tmp_path / "src", keep=keep)
    error = None
    try:
        run_complete_case(tmp_path / "src" / "case", tmp_path / "out", case_setup=hook, **options(kind, **extra))
    except Exception as exc:  # noqa: BLE001 - the tests below inspect the refusal
        error = exc
    return keep, error


def owned_snapshot(mapping):
    return {"keys": list(mapping), "ids": {k: id(v) for k, v in mapping.items()},
            "prepared_keys": list(mapping["prepared"]), "prepared_ids": {k: id(v) for k, v in mapping["prepared"].items()},
            "state0": snapshot(mapping["prepared"]["state0"])}


def assert_owned_unchanged(mapping, before):
    assert list(mapping) == before["keys"] and {k: id(v) for k, v in mapping.items()} == before["ids"]
    assert list(mapping["prepared"]) == before["prepared_keys"]
    assert {k: id(v) for k, v in mapping["prepared"].items()} == before["prepared_ids"]
    assert unchanged(before["state0"], mapping["prepared"]["state0"])


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("failure", ["none", "before_loop", "during_loop", "source_check", "rejection_then_ok"])
def test_hook_owned_mapping_and_initial_state_are_unchanged(kind, failure, tmp_path, monkeypatch):
    """The hook keeps its returned mapping. Whatever happens, the runner removes, adds or replaces nothing in it, and
    the initial state it holds keeps its exact bytes (and stays alive: the hook owns it)."""
    keep, captured = [], {}
    hook = make_hook(kind, tmp_path / "src", keep=keep)

    def capture_hook(**kw):
        mapping = hook(**kw)
        captured["before"] = owned_snapshot(mapping)
        return mapping

    extra = {}
    if failure == "before_loop":
        extra["pause_at_s"] = 3.3  # not an original boundary: refused before the event starts
    real = event_module._sediment_accounting_step
    calls = [0]

    def step(*args, **kwargs):
        calls[0] += 1
        if failure == "during_loop" and calls[0] >= 3:
            raise TransportStepRejected("injected persistent rejection")
        if failure == "rejection_then_ok" and calls[0] == 3:
            raise TransportStepRejected("injected single rejection")
        if failure == "source_check" and calls[0] == 3:
            xml = Path(keep[0]["prepared"]["xml_path"])
            xml.write_text(xml.read_text() + "<!-- changed -->\n")
        return real(*args, **kwargs)

    monkeypatch.setattr(event_module, "_sediment_accounting_step", step)
    out = tmp_path / "out"
    if failure in ("none", "rejection_then_ok"):
        run = run_complete_case(tmp_path / "src" / "case", out, case_setup=capture_hook, **options(kind, **extra))
        assert run.summary["status"] == "complete" and run.summary["sediment_closure"]["closed"]
        if failure == "rejection_then_ok":
            assert run.summary["n_rejected_attempts"] >= 1
    else:
        match = {"before_loop": "pause time", "during_loop": "rejected|halving",
                 "source_check": "MAHLERAN XML changed"}[failure]
        with pytest.raises(SedimentEventError, match=match):
            run_complete_case(tmp_path / "src" / "case", out, case_setup=capture_hook, **options(kind, **extra))
        assert not out.exists()
    assert len(keep) == 1
    assert_owned_unchanged(keep[0], captured["before"])


@pytest.mark.parametrize("kind", KINDS)
def test_restart_identity_refusal_keeps_the_hook_mapping(kind, tmp_path):
    checkpoint = pause_then(kind, tmp_path)
    keep = []
    hook = make_hook(kind, tmp_path / "src3", keep=keep)
    with pytest.raises(SedimentEventError, match="identity"):
        run_complete_case(tmp_path / "src3" / "case", tmp_path / "bad", case_setup=hook, resume=checkpoint,
                          **options(kind, max_dt_s=0.5))
    assert len(keep) == 1 and keep[0]["prepared"]["state0"] is not None


@pytest.mark.parametrize("kind", KINDS)
def test_public_complete_event_caller_references_are_untouched(kind, tmp_path):
    """A direct caller of `complete_event` keeps its own state / continuation objects and their exact bytes."""
    from b30_cases import build_case

    from maple_syrup.complete_event import CompletionPolicy, DryHandoff, complete_event
    from maple_syrup.sediment_event import SedimentEventControl
    from maple_syrup.storm import StormControl

    storage = "dense" if kind == "dense" else "csr"
    control = SedimentEventControl(storm=StormControl(max_dt_s=1, implementation="array", max_steps=10000),
                                   record_exchange_history=True, phase_storage=storage).validated()
    s, ctx, schedule = build_case(kind, control, tmp_path)
    import dataclasses

    ctx = dataclasses.replace(ctx, record_exchange_history=True)  # the bed context must ask what the control records
    state0 = s["state0"]
    before = snapshot(state0)
    args = (ctx, s["column"], s["field"], schedule, s["vegetation"], s["sediment"])
    common = {"max_end_s": 180, "control": control, "policy": CompletionPolicy(hold_s=2), "report_every_s": 2.}
    paused = complete_event(state0, *args, checkpoint_callback=lambda p: p.result.state.t_s >= 4, **common)
    assert unchanged(before, state0)
    held_state, held_result = paused.result.state, paused.result
    before_paused = snapshot(paused)
    done = complete_event(held_state, *args, continuation=paused, **common)
    assert isinstance(done, DryHandoff)
    assert paused.result is held_result and paused.result.state is held_state
    assert unchanged(before_paused, paused)
    assert unchanged(before, state0)
