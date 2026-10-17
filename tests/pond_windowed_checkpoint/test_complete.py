"""Completed (dry) windowed pond handoffs: the terminal MAPLE deposition and the recorded surface + soil reset are data, never
replayed; a wet state at the quiet boundary finalises exactly like the original (not run by its author)."""

from __future__ import annotations

import numpy as np
import pytest

from maple_syrup.pond_checkpoint import PondCheckpointError, load_pond_checkpoint, save_pond_checkpoint
from maple_syrup.pond_event import PondDryHandoff, PondEventError

from .wfixture import IDENTITY, assert_tree_equal, dry_case, fingerprint, host, lake_case, published, store_pins

END_S = 120.0


def completed(case):
    event = case.event()
    start = event.initial_state(case.bed, case.soil)
    return event, start, event.complete(start, END_S)


def test_dry_handoff_roundtrip_is_data_and_cannot_resume(tmp_path):
    case = dry_case(tmp_path)
    event, _start, handoff = completed(case)
    assert isinstance(handoff, PondDryHandoff) and handoff.dry_state.status == "dry"
    assert handoff.dry_state.bed.voxel_column.pending_deep is None, "finalize must have published the storage"
    assert float(host(handoff.pre_reset_soil_m).sum()) > 0.0  # a recorded, nonzero soil reset
    before, pins = fingerprint(handoff), store_pins(case.store_dir)
    path = save_pond_checkpoint(tmp_path / "done", handoff, event, IDENTITY)
    assert fingerprint(handoff) == before and store_pins(case.store_dir) == pins

    fresh_event = case.event()
    with pytest.raises(PondCheckpointError, match="completed"):
        load_pond_checkpoint(path, fresh_event, IDENTITY)
    loaded = load_pond_checkpoint(path, fresh_event, IDENTITY, allow_complete=True)
    assert isinstance(loaded, PondDryHandoff) and loaded.status == "complete"
    assert_tree_equal(handoff, loaded)
    assert loaded.accounting == handoff.accounting
    assert store_pins(case.store_dir) == pins

    # no repeated finalisation / reset: the dry state is not wet, and the reset arrays are exactly the recorded ones
    with pytest.raises(PondEventError, match="wet"):
        fresh_event.finalize(loaded.dry_state)
    np.testing.assert_array_equal(host(loaded.pre_reset_surface_m), host(loaded.dry_state.water.reset_depth))
    np.testing.assert_array_equal(host(loaded.pre_reset_soil_m), host(loaded.dry_state.water.reset_soil))
    assert loaded.dry_state.counters.n_final_commits == handoff.dry_state.counters.n_final_commits
    assert fresh_event.audit(loaded.dry_state)["water"]["closes"]
    assert store_pins(case.store_dir) == pins, "reading the handoff changed the deep store"


def test_closed_lake_nonzero_terminal_deposition_and_surface_reset_survive(tmp_path):
    case = lake_case(tmp_path)
    event = case.event()
    start = event.initial_state(case.bed, case.soil)
    assert float(host(start.hydraulic.depth_m).max()) > 0.0 and float(host(start.bed.water.mobile_mass_by_cell_class_kg).sum()) > 0.0
    handoff = event.complete(start, END_S)
    assert float(host(handoff.terminal_deposition_kg).sum()) > 0.0, "the flow-end settlement deposited nothing"
    assert float(host(handoff.pre_reset_surface_m).sum()) > 0.0
    path = save_pond_checkpoint(tmp_path / "lake", handoff, event, IDENTITY)
    fresh_event = case.event()
    loaded = load_pond_checkpoint(path, fresh_event, IDENTITY, allow_complete=True)
    assert_tree_equal(handoff, loaded)
    np.testing.assert_array_equal(host(loaded.terminal_deposition_kg), host(handoff.terminal_deposition_kg))
    assert loaded.accounting["terminal_deposition_by_class_kg"] == handoff.accounting["terminal_deposition_by_class_kg"]
    np.testing.assert_array_equal(host(loaded.dry_state.sediment.deposited_terminal),
                                  host(handoff.dry_state.sediment.deposited_terminal))
    assert loaded.dry_state.counters.n_final_commits == handoff.dry_state.counters.n_final_commits
    assert fresh_event.audit(loaded.dry_state)["sediment"]["closes"]
    # the loaded handoff is only data: a second finalisation (a second terminal deposition / reset) is impossible
    with pytest.raises(PondEventError, match="wet"):
        fresh_event.finalize(loaded.dry_state)


def test_bare_dry_state_is_not_a_handoff_and_is_refused(tmp_path):
    case = dry_case(tmp_path)
    event, _start, handoff = completed(case)
    with pytest.raises(PondCheckpointError, match="not a PondDryHandoff"):
        save_pond_checkpoint(tmp_path / "bare", handoff.dry_state, event, IDENTITY)
    assert not (tmp_path / "bare").exists()


def test_wet_state_at_quiet_checkpoints_then_finalises_exactly_like_the_original(tmp_path):
    case = dry_case(tmp_path)
    event = case.event()
    result = event.evolve(event.initial_state(case.bed, case.soil), END_S, stop_when_quiet=True)
    assert result.quiet_ready, f"fixture did not reach a quiet state: {result.stop_reason}"
    wet = published(result.state, event)
    assert wet.status == "wet"
    path = save_pond_checkpoint(tmp_path / "quiet", wet, event, IDENTITY)
    fresh_event = case.event()
    loaded = load_pond_checkpoint(path, fresh_event, IDENTITY)
    assert_tree_equal(wet, loaded)
    assert loaded.quiet == wet.quiet and loaded.quiet.ready
    a, b = event.finalize(wet), fresh_event.finalize(loaded)
    assert_tree_equal(a, b)
    assert a.dry_state.bed.voxel_column.pending_deep is None


def test_complete_flag_type_and_pending_dry_state_is_refused(tmp_path):
    case = dry_case(tmp_path)
    event, _start, handoff = completed(case)
    path = save_pond_checkpoint(tmp_path / "done", handoff, event, IDENTITY)
    with pytest.raises(PondCheckpointError, match="allow_complete"):
        load_pond_checkpoint(path, case.event(), IDENTITY, allow_complete="yes")
