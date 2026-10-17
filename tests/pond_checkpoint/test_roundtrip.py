"""Same-backend restart of the pond event: save a wet state, load it into a FRESH event, continue, and require the exact
continuation of the uninterrupted run (nothing here was run by its author)."""

from __future__ import annotations

import numpy as np
import pytest

from maple_syrup import checkpoint as old_checkpoint
from maple_syrup import pond_checkpoint as pc
from maple_syrup.pond_checkpoint import load_pond_checkpoint, save_pond_checkpoint

from .pond_fixture import IDENTITY, advance, assert_tree_equal, fingerprint, host, make_case, tree_hash

N_BEFORE, N_AFTER = 6, 6


def physical(state) -> None:
    """The fixture must have done real work, or equality would be vacuous."""
    assert state.counters.n_accepted_steps > 0 and state.t_s > 0
    assert float(host(state.hydraulic.depth_m).max()) > 0.0
    assert float(np.abs(host(state.hydraulic.qx_m2_s)).max() + np.abs(host(state.hydraulic.qy_m2_s)).max()) > 0.0
    sl = state.sediment
    moved = float(host(sl.wet_actual).sum() + host(state.bed.water.mobile_mass_by_cell_class_kg).sum())
    assert moved > 0.0, "no sediment moved: the fixture does not exercise the sediment continuation"


def test_default_bound_is_the_existing_one_gib():
    assert pc.DEFAULT_MAX_CHECKPOINT_BYTES == old_checkpoint.DEFAULT_MAX_CHECKPOINT_BYTES == 1024**3
    assert pc.SCHEMA != old_checkpoint.SCHEMA and pc.MANIFEST_NAME != "checkpoint.json"


def test_wet_pending_ledger_below_threshold_keeps_source_and_continues_exactly(tmp_path):
    """A real native pending ledger below MAPLE's trigger: the held source / generation 0 are preserved, not refreshed."""
    case = make_case(pickup_kg_per_class=5e-4)
    event = case.event()
    start = event.initial_state(case.bed, case.soil)
    paused = advance(event, start, N_BEFORE)
    physical(paused)
    assert float(np.abs(host(paused.bed.ledger.pending_bed_mass_change_kg)).max()) > 0.0, "no pending ledger to carry"
    assert paused.counters.n_commits == 0 and paused.topology.generation == 0
    before = fingerprint(paused)

    path = save_pond_checkpoint(tmp_path / "ck", paused, event, IDENTITY)
    assert fingerprint(paused) == before, "save mutated the state"
    assert path.is_dir() and not [p for p in tmp_path.iterdir() if p.name.startswith(".")]

    fresh_event = case.event()  # a restarting process: new event from the ORIGINAL topology
    loaded = load_pond_checkpoint(path, fresh_event, IDENTITY)
    assert_tree_equal(paused, loaded)
    assert loaded.topology.generation == 0
    np.testing.assert_array_equal(host(loaded.held_source_m), host(paused.held_source_m))
    assert loaded.topology.source_sha256 == paused.topology.source_sha256

    full = advance(event, paused, N_AFTER)
    resumed = advance(fresh_event, loaded, N_AFTER)
    assert_tree_equal(full, resumed)
    assert fingerprint(paused) == before


def test_generation_survives_an_actual_threshold_crossing(tmp_path):
    """The native pickup is far above the pending-elevation trigger: the first accepted step commits and refreshes the
    topology. The checkpoint stores the HELD source of that generation, rebuilds from it (not from the latest bed) and continues."""
    case = make_case(pickup_kg_per_class=0.1)
    event = case.event()
    start = event.initial_state(case.bed, case.soil)
    paused = advance(event, start, N_BEFORE)
    physical(paused)
    assert paused.counters.n_commits >= 1 and paused.counters.n_refreshes >= 1
    assert paused.topology.generation == paused.counters.n_refreshes >= 1
    assert paused.topology.source_sha256 != start.topology.source_sha256

    path = save_pond_checkpoint(tmp_path / "ck", paused, event, IDENTITY)
    fresh_event = case.event()
    assert fresh_event._initial_topology.generation == 0  # the event is built from the ORIGINAL topology
    loaded = load_pond_checkpoint(path, fresh_event, IDENTITY)
    assert loaded.topology.generation == paused.topology.generation
    assert loaded.topology.source_sha256 == paused.topology.source_sha256
    assert loaded.topology.topology_sha256 == paused.topology.topology_sha256
    assert_tree_equal(paused, loaded)
    assert_tree_equal(advance(event, paused, N_AFTER), advance(fresh_event, loaded, N_AFTER))


def test_adaptive_cap_clock_and_knots_survive_evolve(tmp_path):
    """`evolve` stores the adaptive step cap in the state; the restart continues with it, with the same rain knots."""
    case = make_case(pickup_kg_per_class=5e-4, rain_end_s=1.5)
    event = case.event()
    start = event.initial_state(case.bed, case.soil)
    first = event.evolve(start, 1.0)
    assert first.reached_target
    paused = first.state
    assert paused.hydraulic.next_dt_cap_s is not None
    path = save_pond_checkpoint(tmp_path / "ck", paused, event, IDENTITY)
    fresh_event = case.event()
    loaded = load_pond_checkpoint(path, fresh_event, IDENTITY)
    assert loaded.hydraulic.next_dt_cap_s == paused.hydraulic.next_dt_cap_s and loaded.t_s == paused.t_s
    # the second half crosses the rain edge at 1.5 s: the knot is split exactly the same way
    a = event.evolve(paused, 3.0)
    b = fresh_event.evolve(loaded, 3.0)
    assert a.reached_target and b.reached_target
    assert (a.n_accepted_steps, a.min_accepted_dt_s, a.max_accepted_dt_s) == (b.n_accepted_steps, b.min_accepted_dt_s, b.max_accepted_dt_s)
    assert_tree_equal(a.state, b.state)


def test_audit_budgets_of_the_restarted_run_equal_the_uninterrupted_run(tmp_path):
    case = make_case(pickup_kg_per_class=5e-4)
    event = case.event()
    paused = advance(event, event.initial_state(case.bed, case.soil), N_BEFORE)
    path = save_pond_checkpoint(tmp_path / "ck", paused, event, IDENTITY)
    fresh_event = case.event()
    loaded = load_pond_checkpoint(path, fresh_event, IDENTITY)
    full = advance(event, paused, N_AFTER)
    resumed = advance(fresh_event, loaded, N_AFTER)
    a, b = event.audit(full), fresh_event.audit(resumed)
    assert a["water"]["closes"] and a["sediment"]["closes"] and b["water"]["closes"] and b["sediment"]["closes"]
    assert a == b


def test_load_does_not_commit_reroute_reset_or_finalise(tmp_path):
    case = make_case(pickup_kg_per_class=0.1)
    event = case.event()
    paused = advance(event, event.initial_state(case.bed, case.soil), N_BEFORE)
    path = save_pond_checkpoint(tmp_path / "ck", paused, event, IDENTITY)
    loaded = load_pond_checkpoint(path, case.event(), IDENTITY)
    for name in ("n_commits", "n_refreshes", "n_final_commits", "n_accepted_steps"):
        assert getattr(loaded.counters, name) == getattr(paused.counters, name)
    assert loaded.status == "wet" and loaded.water.reset_depth is None
    assert loaded.bed.committed_topography.commit_count == paused.bed.committed_topography.commit_count


def test_checkpoint_directory_layout_and_manifest(tmp_path):
    import json

    case = make_case(pickup_kg_per_class=5e-4)
    event = case.event()
    paused = advance(event, event.initial_state(case.bed, case.soil), 2)
    path = save_pond_checkpoint(tmp_path / "ck", paused, event, IDENTITY)
    assert sorted(p.name for p in path.iterdir()) == sorted([pc.MANIFEST_NAME, *pc.PAYLOADS])
    manifest = json.loads((path / pc.MANIFEST_NAME).read_text())
    assert manifest["schema"] == pc.SCHEMA and manifest["status"] == "wet" and manifest["array_backend"] == "numpy"
    assert manifest["identity"] == IDENTITY and set(manifest["source"]) >= {"maple_syrup_tree_sha256", "maple_package_sha256"}
    assert manifest["source"] == pc.pond_checkpoint_source_pair()
    # the native snapshot owns the bed; the companion does not repeat the voxel / active inventory
    with np.load(path / pc.PAYLOADS[1]) as companion:
        assert not [n for n in companion.files if "voxel" in n or "active_mass" in n]
    # the accepted checkpoint cannot decode this format (no checkpoint.json) and this module refuses that one
    assert not (path / "checkpoint.json").exists()
    assert tree_hash(path)
