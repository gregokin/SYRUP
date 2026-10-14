"""CW-D2: opt-in recording of the ACTUAL per-cell/class MAPLE exchange history through every commit.

Semantics under test (MAPLE's, not inferred): a touch is one ledger accumulate call that moved mass at the
cell/class; SYRUP's water step makes two such calls per accepted step (removal, deposition), so the live counter
after one step equals `nonzero(actual pickup) + nonzero(actual deposition)`; the avalanche fold-in is captured
with each commit; commit receipts are summed over commits and the live pending ledger is added exactly once by
`exchange_history_totals`. Default behaviour (no recording) is unchanged, including the checkpoint payload.
Nothing here was run by its author (file-only tools); the root records executed results.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("maple")

from test_coupled_backends import H, setup_xp  # tests/conservative_water via conftest
from test_sediment_event import chain_elevation, make_bed, setup

from maple_syrup import sediment_bed as bed_module
from maple_syrup.checkpoint import load_checkpoint, save_checkpoint, sha256_file
from maple_syrup.complete_event import CompletionPolicy, DryHandoff, complete_event
from maple_syrup.rainfall import constant_rainfall
from maple_syrup.sediment_bed import (
    BedContext,
    commit_bed,
    maple_supports_exchange_history,
)
from maple_syrup.sediment_event import (
    EXCHANGE_HISTORY_KEYS,
    SedimentEventControl,
    SedimentEventError,
    evolve_sediment_event,
    exchange_history_totals,
    fold_commit_exchange_history,
    new_exchange_history,
    sediment_coupled_step,
    validate_exchange_history,
)
from maple_syrup.storm import StormControl

IDENTITY = {"case": "controlled actual MAPLE bed", "forcing": "fixed", "source": "test", "options": {"dt": 1}}


def inputs(xp=np, *, wet=True, record=True, commit_interval_s=None, implementation="array",
           force_final_commit=False):
    """3-cell chain (ksat 1e-5 m/s), optionally wet with 10 s of 100 mm/h, interval commits when asked."""
    from maple.core.parameters.topographic_commit import TopographicCommitSpec

    spec = None if commit_interval_s is None else TopographicCommitSpec(commit_interval_s=commit_interval_s)
    bed, ctx = make_bed(chain_elevation(3), depth_m=.001 if wet else 0., commit_spec=spec)
    if record:
        ctx = replace(ctx, record_exchange_history=True)
    control = SedimentEventControl(storm=StormControl(max_dt_s=1, implementation=implementation, max_steps=10000),
                                   record_exchange_history=record, force_final_commit=force_final_commit)
    s = setup(bed, ctx, ksat=1e-5, control=control) if xp is np else setup_xp(bed, ctx, xp, ksat=1e-5, control=control)
    schedule = constant_rainfall(0, 10 if wet else 2, 100 if wet else 0)
    return s, s["ctx"], control, schedule


def run(s, ctx, control, schedule, end, *, continuation=None, state=None, cadence=2.0):
    return evolve_sediment_event(state or s["state0"], ctx, s["column"], s["field"], schedule, s["vegetation"],
                                 s["sediment"], end, control, report_every_s=cadence, continuation=continuation)


def complete(s, ctx, control, schedule, *, max_end_s=180, **kw):
    return complete_event(s["state0"], ctx, s["column"], s["field"], schedule, s["vegetation"], s["sediment"],
                          max_end_s=max_end_s, control=control, policy=CompletionPolicy(hold_s=2),
                          report_every_s=2., **kw)


def gross_identity_bound(result, totals):
    """MAPLE summation bound for `gross == cumulative pickup + cumulative deposition` per cell/class: every
    accounting call contributed one Kahan term on each side (2 per step plus terminal / commit folds)."""
    from maple.surface.voxels._numerics import summation_error_bound_kg

    gross = H(totals["arrays"]["gross_physical_activity_kg"])
    n_terms = 2 * result.n_accepted_steps + result.n_commits + 2
    return summation_error_bound_kg(n_terms, max(float(gross.max()), 1.0))


# --- default behaviour and dependency support ----------------------------------------------------------------------
def test_default_control_and_context_record_nothing():
    assert BedContext.__dataclass_fields__["record_exchange_history"].default is False
    assert SedimentEventControl().record_exchange_history is False
    s, ctx, control, schedule = inputs(record=False)
    result = run(s, ctx, control, schedule, 4)
    assert result.exchange_history is None
    with pytest.raises(SedimentEventError, match="did not record"):
        exchange_history_totals(result)
    # a forced MAPLE commit through the default context carries no history fields
    state = result.state
    outcome = commit_bed(state.bed, ctx, state.graph, state.terrain, state.t_s, force=True)
    assert outcome.did_commit and outcome.maple_result.pre_commit_n_physical_touches is None
    assert outcome.maple_result.exchange_history_receipt is None


def test_recording_requires_a_supporting_dependency_and_refuses_before_any_step(monkeypatch):
    assert maple_supports_exchange_history()  # the private CW-D2 dependency
    s, ctx, control, schedule = inputs()
    state0 = s["state0"]
    ledger_before = H(state0.bed.ledger.n_physical_touches).copy()
    monkeypatch.setattr(bed_module, "maple_supports_exchange_history", lambda: False)
    with pytest.raises(SedimentEventError, match="cannot record the exchange history"):
        run(s, ctx, control, schedule, 4)
    with pytest.raises(SedimentEventError, match="cannot record the exchange history"):
        complete(s, ctx, control, schedule)
    assert state0.t_s == 0 and np.array_equal(H(state0.bed.ledger.n_physical_touches), ledger_before)
    # the bed adapter itself refuses too, before evaluating any trigger or commit
    with pytest.raises(bed_module.BedIntegrationError, match="cannot record"):
        commit_bed(state0.bed, ctx, state0.graph, state0.terrain, 1.0, force=True)


def test_control_and_context_must_agree():
    s, ctx, control, schedule = inputs()
    with pytest.raises(SedimentEventError, match="must agree"):
        run(s, replace(ctx, record_exchange_history=False), control, schedule, 4)
    with pytest.raises(SedimentEventError, match="must agree"):
        run(s, ctx, replace(control, record_exchange_history=False), schedule, 4)
    with pytest.raises(SedimentEventError, match="needs actual MAPLE commits"):
        SedimentEventControl(record_exchange_history=True, commit=False).validated()
    with pytest.raises(SedimentEventError, match="must be bool"):
        SedimentEventControl(record_exchange_history=1).validated()


# --- analytic touch counts ---------------------------------------------------------------------------------------------
def test_one_step_live_counter_is_the_number_of_nonzero_water_calls():
    s, ctx, control, schedule = inputs()
    state0 = s["state0"]
    assert not np.any(H(state0.bed.ledger.n_physical_touches))
    rate = s["field"].apply(schedule.rate_after_m_per_s(0.0))
    attempt = sediment_coupled_step(state0, ctx, s["column"], rate, s["vegetation"], s["sediment"], 1.0, control)
    pickup = H(attempt.pickup.actual_removal_by_cell_class_kg)
    deposition = H(attempt.deposit.deposition_by_cell_class_kg)
    expected = (pickup != 0).astype(np.int64) + (deposition != 0).astype(np.int64)
    np.testing.assert_array_equal(H(attempt.bed.ledger.n_physical_touches), expected)
    assert expected.max() >= 1 and np.any(pickup > 0)  # genuine activity on the wet, raining chain
    gross = H(attempt.bed.ledger.actual_mass_positive_kg).sum(axis=0) + H(attempt.bed.ledger.actual_mass_negative_kg).sum(axis=0)
    np.testing.assert_allclose(gross, pickup + deposition, rtol=0, atol=8 * np.finfo(float).eps * max(gross.max(), 1.0))


def test_history_over_multiple_commits_matches_the_cumulative_exchange_and_counts_every_commit():
    s, ctx, control, schedule = inputs(commit_interval_s=2.0)
    result = run(s, ctx, control, schedule, 20)
    assert result.n_commits >= 3
    history = result.exchange_history
    assert tuple(history) == EXCHANGE_HISTORY_KEYS and history["n_commits_recorded"] == result.n_commits
    assert history["receipt"]["captured_after_stage"] == "record_avalanche_transfer"
    assert all(rec["commit_stage_log"].index("record_avalanche_transfer") < rec["commit_stage_log"].index("clear_pending_ledger")
               for rec in result.commit_log)
    totals = exchange_history_totals(result)
    assert totals["history_complete"] and totals["includes_pending_ledger"]
    touches = H(totals["arrays"]["n_physical_touches"])
    gross = H(totals["arrays"]["gross_physical_activity_kg"])
    pickup, deposition = H(result.cumulative_pickup_kg), H(result.cumulative_deposition_kg)
    exchanged = pickup + deposition
    assert touches.dtype == np.int64 and touches.min() >= 0
    assert np.all(touches[exchanged > 0] >= 1) and not np.any(touches[exchanged == 0])
    assert touches.max() <= 2 * result.n_accepted_steps  # at most two calls per accepted step
    assert touches.sum() > 0
    bound = gross_identity_bound(result, totals)
    assert np.all(np.abs(gross - exchanged) <= bound)
    # the reference is the event's initial bed per cell/class, as an independent copy
    np.testing.assert_array_equal(H(history["reference_inventory_kg"]), H(result.initial_bed_by_cell_class_kg))
    assert history["reference_inventory_kg"] is not result.initial_bed_by_cell_class_kg
    # committed + pending partition: the live ledger is added exactly once
    committed = H(totals["arrays"]["committed_n_physical_touches"])
    pending = H(result.state.bed.ledger.n_physical_touches)
    np.testing.assert_array_equal(touches, committed + pending)
    assert np.any(pending) or not np.any(H(totals["arrays"]["pending_n_physical_touches"]))


def test_forced_final_commit_empties_the_pending_part():
    s, ctx, control, schedule = inputs(commit_interval_s=2.0, force_final_commit=True)
    result = run(s, ctx, control, schedule, 7)
    totals = exchange_history_totals(result)
    assert totals["history_complete"] and totals["pending_ledger_empty"]
    assert not np.any(H(totals["arrays"]["pending_n_physical_touches"]))
    np.testing.assert_array_equal(H(totals["arrays"]["n_physical_touches"]),
                                  H(totals["arrays"]["committed_n_physical_touches"]))


def test_zero_activity_event_records_a_valid_zero_history():
    s, ctx, control, schedule = inputs(wet=False, commit_interval_s=2.0)
    handoff = complete(s, ctx, control, schedule, max_end_s=12)
    assert isinstance(handoff, DryHandoff)
    totals = exchange_history_totals(handoff.progress.result)
    assert totals["history_complete"] and totals["pending_ledger_empty"]
    assert not np.any(H(totals["arrays"]["n_physical_touches"])) and not np.any(H(totals["arrays"]["gross_physical_activity_kg"]))
    assert handoff.progress.result.exchange_history["n_commits_recorded"] == handoff.progress.result.n_commits


def test_terminal_settlement_and_forced_final_commit_are_in_the_completed_history():
    s, ctx, control, schedule = inputs(commit_interval_s=2.0)
    handoff = complete(s, ctx, control, schedule)
    assert isinstance(handoff, DryHandoff)
    result = handoff.progress.result
    totals = exchange_history_totals(result)
    assert totals["history_complete"] and totals["pending_ledger_empty"]
    touches = H(totals["arrays"]["n_physical_touches"])
    gross = H(totals["arrays"]["gross_physical_activity_kg"])
    exchanged = H(result.cumulative_pickup_kg) + H(result.cumulative_deposition_kg)  # includes the terminal deposit
    terminal = H(handoff.terminal_deposition_kg)
    assert np.all(touches[terminal > 0] >= 1)
    assert np.all(np.abs(gross - exchanged) <= gross_identity_bound(result, totals))
    assert result.n_forced_commits >= 1 and result.exchange_history["n_commits_recorded"] == result.n_commits


# --- checkpoint / resume ---------------------------------------------------------------------------------------------
def test_checkpoint_preserves_history_and_pending_truth_and_resume_equals_uninterrupted(tmp_path):
    s, ctx, control, schedule = inputs(commit_interval_s=2.0)
    paused = complete(s, ctx, control, schedule, checkpoint_callback=lambda p: True)
    assert paused.result.state.t_s == 2
    before = paused.result.exchange_history
    pending_before = H(paused.result.state.bed.ledger.n_physical_touches).copy()
    assert np.any(pending_before) or np.any(H(before["committed_n_physical_touches"]))
    path = tmp_path / "checkpoint"
    save_checkpoint(path, paused, ctx, s["column"], s["sediment"], IDENTITY)
    manifest = json.loads((path / "checkpoint.json").read_text())
    result_fields = manifest["payload"]["fields"]["result"]["fields"]
    assert "exchange_history" in result_fields and "record_exchange_history" in manifest["payload"]["fields"]["control"]["fields"]
    restored = load_checkpoint(path, ctx, s["column"], s["sediment"], IDENTITY)
    after = restored.result.exchange_history
    for name in ("reference_inventory_kg", "committed_n_physical_touches", "committed_gross_physical_activity_kg"):
        np.testing.assert_array_equal(H(after[name]), H(before[name]))
    assert after["n_commits_recorded"] == before["n_commits_recorded"] and after["receipt"] == before["receipt"]
    np.testing.assert_array_equal(H(restored.result.state.bed.ledger.n_physical_touches), pending_before)
    full = complete(s, ctx, control, schedule)
    resumed = complete_event(restored.result.state, ctx, s["column"], s["field"], schedule, s["vegetation"],
                             s["sediment"], max_end_s=180, control=control, policy=CompletionPolicy(hold_s=2),
                             report_every_s=2., continuation=restored)
    a, b = exchange_history_totals(full.progress.result), exchange_history_totals(resumed.progress.result)
    for name in a["arrays"]:
        assert H(a["arrays"][name]).tobytes() == H(b["arrays"][name]).tobytes(), name
    assert a["n_commits_recorded"] == b["n_commits_recorded"] and a["history_complete"] and b["history_complete"]


def test_default_checkpoint_payload_has_no_history_keys(tmp_path):
    s, ctx, control, schedule = inputs(record=False, commit_interval_s=2.0)
    paused = complete(s, ctx, control, schedule, checkpoint_callback=lambda p: True)
    path = tmp_path / "default"
    save_checkpoint(path, paused, ctx, s["column"], s["sediment"], IDENTITY)
    manifest = json.loads((path / "checkpoint.json").read_text())
    assert "exchange_history" not in manifest["payload"]["fields"]["result"]["fields"]
    assert "record_exchange_history" not in manifest["payload"]["fields"]["control"]["fields"]
    restored = load_checkpoint(path, ctx, s["column"], s["sediment"], IDENTITY)
    assert restored.result.exchange_history is None and restored.control.record_exchange_history is False


def test_prehistory_checkpoint_cannot_claim_a_history(tmp_path):
    s, ctx, control, schedule = inputs(record=False, commit_interval_s=2.0)
    paused = complete(s, ctx, control, schedule, checkpoint_callback=lambda p: True)
    path = tmp_path / "prehistory"
    save_checkpoint(path, paused, ctx, s["column"], s["sediment"], IDENTITY)
    restored = load_checkpoint(path, ctx, s["column"], s["sediment"], IDENTITY)
    recording_ctx = replace(ctx, record_exchange_history=True)
    recording_control = replace(control, record_exchange_history=True)
    # the solver refuses to invent a history for commits it did not record
    with pytest.raises(SedimentEventError, match="no complete exchange history"):
        run(s, recording_ctx, recording_control, schedule, 6, continuation=restored.result,
            state=restored.result.state)
    # the completion driver refuses the changed control outright
    with pytest.raises(SedimentEventError, match="controls changed"):
        complete_event(restored.result.state, recording_ctx, s["column"], s["field"], schedule, s["vegetation"],
                       s["sediment"], max_end_s=180, control=recording_control, policy=CompletionPolicy(hold_s=2),
                       report_every_s=2., continuation=restored)


def test_recorded_history_cannot_be_dropped_silently_by_a_non_recording_control():
    s, ctx, control, schedule = inputs(commit_interval_s=2.0)
    first = run(s, ctx, control, schedule, 4)
    with pytest.raises(SedimentEventError, match="refusing to drop"):
        run(s, replace(ctx, record_exchange_history=False), replace(control, record_exchange_history=False),
            schedule, 6, continuation=first, state=first.state)


@pytest.mark.parametrize("kind", ["count", "negative", "receipt_stage", "control_flag", "shape"])
def test_malformed_history_in_a_checkpoint_is_refused(tmp_path, kind):
    s, ctx, control, schedule = inputs(commit_interval_s=2.0)
    paused = complete(s, ctx, control, schedule, checkpoint_callback=lambda p: True)
    path = tmp_path / "corrupt"
    save_checkpoint(path, paused, ctx, s["column"], s["sediment"], IDENTITY)
    metadata = path / "checkpoint.json"
    m = json.loads(metadata.read_text())
    fields = m["payload"]["fields"]
    history = fields["result"]["fields"]["exchange_history"]["dict"]
    with np.load(path / "continuation.npz") as archive:
        data = {k: archive[k].copy() for k in archive.files}
    if kind == "count":
        history["n_commits_recorded"] += 1
    elif kind == "receipt_stage":
        history["receipt"]["dict"]["captured_after_stage"] = "validate_transaction"
    elif kind == "control_flag":
        fields["control"]["fields"]["record_exchange_history"] = False
    else:
        node = history["committed_n_physical_touches"]
        name = node["array"]
        if kind == "negative":
            data[name].flat[0] = -1
        else:
            data[name] = data[name][0:1].copy()
            node["shape"] = list(data[name].shape)
        np.savez(path / "continuation.npz", **data)
        m["files"]["continuation.npz"] = sha256_file(path / "continuation.npz")
    metadata.write_text(json.dumps(m))
    with pytest.raises((SedimentEventError, ValueError)):
        load_checkpoint(path, ctx, s["column"], s["sediment"], IDENTITY)


# --- runner outputs ----------------------------------------------------------------------------------------------------
def test_runner_writes_history_outputs_only_when_asked(plot1_case, tmp_path):
    from maple_syrup.event_experiment import run_complete_plot1

    default = run_complete_plot1(plot1_case, tmp_path / "default", implementation="array", max_end_s=120,
                                 pause_at_s=60, checkpoint_every_s=60, report_every_s=30)
    assert "exchange_history" not in default.summary and not (tmp_path / "default" / "exchange_history.npz").exists()
    assert "record_exchange_history" not in default.summary["identity"]["control"]
    recorded = run_complete_plot1(plot1_case, tmp_path / "recorded", implementation="array", max_end_s=120,
                                  pause_at_s=60, checkpoint_every_s=60, report_every_s=30, record_exchange_history=True)
    block = recorded.summary["exchange_history"]
    assert block["history_complete"] and block["includes_pending_ledger"]
    assert block["n_commits_recorded"] == recorded.summary["n_commits"]
    assert recorded.summary["identity"]["control"]["record_exchange_history"] is True
    with np.load(tmp_path / "recorded" / "exchange_history.npz") as archive:
        touches = archive["n_physical_touches"]
        assert touches.dtype == np.int64 and touches.min() >= 0 and touches.sum() > 0
        np.testing.assert_array_equal(touches, archive["committed_n_physical_touches"] + archive["pending_n_physical_touches"])
        for name in block["arrays"]:
            assert block["arrays"][name]["sha256"] == hashlib.sha256(archive[name].tobytes()).hexdigest()
    assert block["binding"]["syrup_source_sha256"] == recorded.summary["identity"]["syrup_source_sha256"]
    # the physical outputs do not depend on recording
    for name in ("final_state.npz", "hydrograph.npz"):
        with np.load(tmp_path / "default" / name) as a, np.load(tmp_path / "recorded" / name) as b:
            for key in a.files:
                assert a[key].tobytes() == b[key].tobytes(), (name, key)


# --- guards: result finiteness, counter wrap, receipt contract (correction 2) -------------------------------------------
def _recorded(commit_interval_s=2.0, end=6):
    s, ctx, control, schedule = inputs(commit_interval_s=commit_interval_s)
    result = run(s, ctx, control, schedule, end)
    assert result.exchange_history["n_commits_recorded"] >= 1
    return s, ctx, control, schedule, result


def _fake_commit(xp, shape3, receipt, *, touches=0, gross=0.0):
    """A commit-result stand-in carrying the EXACT MAPLE receipt of a real commit and chosen receipt arrays."""
    return SimpleNamespace(pre_commit_n_physical_touches=xp.full(shape3, touches, dtype=np.int64),
                           pre_commit_gross_physical_activity_kg=xp.full(shape3, gross, dtype=np.float64),
                           exchange_history_receipt=dict(receipt))


def _snapshot(history):
    return {k: (H(v).copy() if k in ("reference_inventory_kg", "committed_n_physical_touches",
                                     "committed_gross_physical_activity_kg") else v) for k, v in history.items()}


def _assert_unchanged(history, snapshot):
    for k, v in snapshot.items():
        if isinstance(v, np.ndarray):
            np.testing.assert_array_equal(H(history[k]), v)
        else:
            assert history[k] == v


def test_fold_refuses_an_fp64_overflow_of_the_summed_gross_and_keeps_the_history_unchanged():
    _s, _ctx, _control, _schedule, result = _recorded()
    history = result.exchange_history
    shape3 = tuple(history["committed_n_physical_touches"].shape)
    huge = dict(history, committed_gross_physical_activity_kg=np.full(shape3, 1e308))
    snapshot = _snapshot(huge)
    # the deliberate FP64 overflow raises NumPy's RuntimeWarning in the guarded add; it is expected here only
    with pytest.warns(RuntimeWarning, match="overflow"), pytest.raises(SedimentEventError, match="overflowed FP64"):
        fold_commit_exchange_history(huge, _fake_commit(np, shape3, history["receipt"], gross=1e308), np)
    _assert_unchanged(huge, snapshot)
    # the same receipt onto a legitimate near-finite history is accepted: finite inputs, finite result
    folded = fold_commit_exchange_history(huge, _fake_commit(np, shape3, history["receipt"], gross=0.0), np)
    assert np.all(np.isfinite(folded["committed_gross_physical_activity_kg"]))
    assert np.all(folded["committed_gross_physical_activity_kg"] == 1e308)
    assert folded["n_commits_recorded"] == huge["n_commits_recorded"] + 1
    _assert_unchanged(huge, snapshot)  # inputs immutable


def test_fold_refuses_an_int64_counter_wrap_and_accepts_the_largest_representable_count():
    _s, _ctx, _control, _schedule, result = _recorded()
    history = result.exchange_history
    shape3 = tuple(history["committed_n_physical_touches"].shape)
    top = dict(history, committed_n_physical_touches=np.full(shape3, np.iinfo(np.int64).max, dtype=np.int64))
    snapshot = _snapshot(top)
    with pytest.raises(SedimentEventError, match="counter overflowed"):
        fold_commit_exchange_history(top, _fake_commit(np, shape3, history["receipt"], touches=1), np)
    _assert_unchanged(top, snapshot)
    folded = fold_commit_exchange_history(top, _fake_commit(np, shape3, history["receipt"], touches=0), np)
    assert np.all(folded["committed_n_physical_touches"] == np.iinfo(np.int64).max)


def _pending_ledger_with_huge_slots(ledger, process_indices, value=1e308):
    """The genuine live ledger with `value` written into the positive-activity channel of the given process slots
    only; every other slot keeps its genuine small activity. One huge slot keeps MAPLE's fixed-order process fold
    finite (1e308 + small = 1e308); two huge slots make that fold itself non-finite."""
    from maple.core.types.sediment_ledger import process_index

    positive = H(ledger.actual_mass_positive_kg).copy()
    for name in process_indices:
        positive[process_index(name)] = value
    return replace(ledger, actual_mass_positive_kg=positive)


def test_totals_refuse_an_fp64_overflow_from_the_pending_ledger_and_accept_near_finite_values():
    _s, _ctx, _control, _schedule, result = _recorded(end=5)  # t = 5: one step after the 4 s commit is pending
    ledger = result.state.bed.ledger
    assert np.any(H(ledger.n_physical_touches))  # genuine pending activity
    shape3 = tuple(ledger.n_physical_touches.shape)
    huge_history = dict(result.exchange_history, committed_gross_physical_activity_kg=np.full(shape3, 1e308))
    # genuinely FINITE pending gross of 1e308: one actual process slot huge, the others genuine
    one_slot = _pending_ledger_with_huge_slots(ledger, ("water_erosion_deposition",))
    pending_gross = H(one_slot.actual_mass_positive_kg).sum(axis=0) + H(one_slot.actual_mass_negative_kg).sum(axis=0)
    assert np.all(np.isfinite(pending_gross)) and np.all(pending_gross >= 1e308)
    with_pending = replace(result.state, bed=replace(result.state.bed, ledger=one_slot))
    broken = replace(result, exchange_history=huge_history, state=with_pending)
    # finite committed 1e308 + finite pending 1e308: the FINAL summed total overflows and is refused
    with pytest.warns(RuntimeWarning, match="overflow"), pytest.raises(SedimentEventError, match="summed gross activity overflowed FP64"):
        exchange_history_totals(broken)
    # separate negative control: a pending gross that is itself non-finite (two huge slots) is refused as such
    two_slots = _pending_ledger_with_huge_slots(ledger, ("water_erosion_deposition", "avalanching"))
    nonfinite = replace(result, state=replace(result.state, bed=replace(result.state.bed, ledger=two_slots)))
    with pytest.warns(RuntimeWarning, match="overflow"), pytest.raises(SedimentEventError, match="receipt gross activity must be finite"):
        exchange_history_totals(nonfinite)
    # legitimate near-finite controls: 1e308 committed with the genuine small pending part, and the genuine
    # committed history with the finite 1e308 pending part, both finite
    near = replace(result, exchange_history=huge_history)
    totals = exchange_history_totals(near)
    assert np.all(np.isfinite(H(totals["arrays"]["gross_physical_activity_kg"])))
    assert totals["n_commits_recorded"] == result.exchange_history["n_commits_recorded"]
    totals = exchange_history_totals(replace(result, state=with_pending))
    assert np.all(np.isfinite(H(totals["arrays"]["gross_physical_activity_kg"])))
    assert np.all(H(totals["arrays"]["pending_gross_physical_activity_kg"]) >= 1e308)
    # the genuine result is untouched by the stand-ins
    assert np.all(np.isfinite(H(result.state.bed.ledger.actual_mass_positive_kg)))
    assert float(H(result.state.bed.ledger.actual_mass_positive_kg).max()) < 1e3


@pytest.mark.parametrize("key", ["touch_semantics", "gross_semantics", "captured_after_stage", "captured_before_stage"])
def test_first_fold_refuses_a_receipt_whose_contract_strings_differ_from_maple(key):
    s, ctx, _control, _schedule, result = _recorded()
    real = result.exchange_history["receipt"]
    state0 = s["state0"]
    fresh = new_exchange_history(np, state0.graph.shape, 6, result.initial_bed_by_cell_class_kg)
    snapshot = _snapshot(fresh)
    shape3 = tuple(fresh["committed_n_physical_touches"].shape)
    altered = dict(real, **{key: real[key] + " (relabelled)"})
    with pytest.raises(SedimentEventError, match="capture contract"):
        fold_commit_exchange_history(fresh, _fake_commit(np, shape3, altered), np)
    _assert_unchanged(fresh, snapshot)
    with pytest.raises(SedimentEventError, match="capture contract"):
        validate_exchange_history(dict(fresh, n_commits_recorded=1, receipt=altered), np, shape3, 1)
    for bad in (dict(real, extra="x"), {k: v for k, v in real.items() if k != "gross_semantics"}):
        with pytest.raises(SedimentEventError, match="unexpected keys"):
            fold_commit_exchange_history(fresh, _fake_commit(np, shape3, bad), np)
    _assert_unchanged(fresh, snapshot)
    # the genuine receipt of a real MAPLE commit is accepted into the fresh history
    outcome = commit_bed(result.state.bed, ctx, result.state.graph, result.state.terrain, result.state.t_s + 1.0,
                         force=True)
    assert outcome.did_commit
    folded = fold_commit_exchange_history(fresh, outcome.maple_result, np)
    assert folded["n_commits_recorded"] == 1 and folded["receipt"] == outcome.maple_result.exchange_history_receipt
    np.testing.assert_array_equal(folded["committed_n_physical_touches"], H(outcome.maple_result.pre_commit_n_physical_touches))


def test_unsupported_source_contract_is_refused_explicitly(monkeypatch):
    import maple.surface.topographic_commit.commit as commit_module

    s, _ctx, _control, _schedule, result = _recorded()
    real = result.exchange_history["receipt"]
    fresh = new_exchange_history(np, s["state0"].graph.shape, 6, result.initial_bed_by_cell_class_kg)
    shape3 = tuple(fresh["committed_n_physical_touches"].shape)
    monkeypatch.delattr(commit_module, "EXCHANGE_HISTORY_TOUCH_SEMANTICS")
    with pytest.raises(SedimentEventError, match="no exchange history capture contract"):
        fold_commit_exchange_history(fresh, _fake_commit(np, shape3, real), np)
    with pytest.raises(SedimentEventError, match="no exchange history capture contract"):
        validate_exchange_history(result.exchange_history, np, shape3, result.n_commits)


def test_checkpoint_refuses_relabelled_receipt_semantics(tmp_path):
    s, ctx, control, schedule = inputs(commit_interval_s=2.0)
    paused = complete(s, ctx, control, schedule, checkpoint_callback=lambda p: True)
    path = tmp_path / "relabelled"
    save_checkpoint(path, paused, ctx, s["column"], s["sediment"], IDENTITY)
    metadata = path / "checkpoint.json"
    m = json.loads(metadata.read_text())
    receipt = m["payload"]["fields"]["result"]["fields"]["exchange_history"]["dict"]["receipt"]["dict"]
    receipt["touch_semantics"] = "count of something else"
    metadata.write_text(json.dumps(m))
    with pytest.raises(SedimentEventError, match="capture contract"):
        load_checkpoint(path, ctx, s["column"], s["sediment"], IDENTITY)


def test_device_fold_and_totals_guard_overflow_too(gpu):
    s, ctx, control, schedule = inputs(gpu, commit_interval_s=2.0, implementation="cuda")
    result = run(s, ctx, control, schedule, 6)
    history = result.exchange_history
    shape3 = tuple(history["committed_n_physical_touches"].shape)
    huge = dict(history, committed_gross_physical_activity_kg=gpu.full(shape3, 1e308))
    with pytest.raises(SedimentEventError, match="overflowed FP64"):
        fold_commit_exchange_history(huge, _fake_commit(gpu, shape3, history["receipt"], gross=1e308), gpu)
    folded = fold_commit_exchange_history(huge, _fake_commit(gpu, shape3, history["receipt"], gross=0.0), gpu)
    assert folded["committed_gross_physical_activity_kg"].__class__ is gpu.ndarray
    assert np.all(np.isfinite(H(folded["committed_gross_physical_activity_kg"])))
    top = dict(history, committed_n_physical_touches=gpu.full(shape3, np.iinfo(np.int64).max, dtype=np.int64))
    with pytest.raises(SedimentEventError, match="counter overflowed"):
        fold_commit_exchange_history(top, _fake_commit(gpu, shape3, history["receipt"], touches=1), gpu)
    # a host receipt onto a device history is a mixed namespace, refused before any arithmetic
    with pytest.raises(SedimentEventError, match="mixed namespace"):
        fold_commit_exchange_history(history, _fake_commit(np, shape3, history["receipt"]), gpu)


# --- device -------------------------------------------------------------------------------------------------------------
def test_device_history_matches_the_host_event(gpu):
    host_s, host_ctx, host_control, schedule = inputs(commit_interval_s=2.0)
    host = run(host_s, host_ctx, host_control, schedule, 20)
    dev_s, dev_ctx, dev_control, _ = inputs(gpu, commit_interval_s=2.0, implementation="cuda")
    dev = run(dev_s, dev_ctx, dev_control, schedule, 20)
    assert dev.exchange_history["committed_n_physical_touches"].__class__ is gpu.ndarray
    assert dev.exchange_history["receipt"]["namespace"] == "cupy"
    assert host.n_commits == dev.n_commits == dev.exchange_history["n_commits_recorded"]
    a, b = exchange_history_totals(host), exchange_history_totals(dev)
    assert b["history_complete"] and b["arrays"]["n_physical_touches"].__class__ is gpu.ndarray
    assert H(a["arrays"]["n_physical_touches"]).tobytes() == H(b["arrays"]["n_physical_touches"]).tobytes()
    gross_a, gross_b = H(a["arrays"]["gross_physical_activity_kg"]), H(b["arrays"]["gross_physical_activity_kg"])
    assert np.all(np.abs(gross_a - gross_b) <= gross_identity_bound(host, a))
