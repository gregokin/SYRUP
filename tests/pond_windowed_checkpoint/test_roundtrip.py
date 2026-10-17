"""Same-backend restart of the pond event over a REAL resident-window bed: save a published wet state, load it into a FRESH
event, continue, and require the exact continuation of the uninterrupted run (nothing here was run by its author)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from maple_syrup import pond_checkpoint as pc
from maple_syrup.pond_checkpoint import load_pond_checkpoint, save_pond_checkpoint

from .wfixture import (
    IDENTITY,
    advance,
    assert_tree_equal,
    fingerprint,
    host,
    logical_column_bytes,
    make_case,
    physical,
    published,
    store_pins,
    tree_hash,
)

N_BEFORE, N_AFTER = 6, 6


def test_schemas_are_distinct_and_dense_is_unchanged():
    assert pc.SCHEMA == "maple-syrup-pond-checkpoint/1" and pc.WINDOWED_SCHEMA == "maple-syrup-pond-checkpoint/2"
    assert pc.DEFAULT_MAX_CHECKPOINT_BYTES == 1024**3
    assert pc.MANIFEST_NAME == "pond_checkpoint.json" and pc.PAYLOADS == ("maple_state.npz", "pond_continuation.npz")
    assert "bed_storage" not in pc.MANIFEST_KEYS and pc.WINDOWED_MANIFEST_KEYS == pc.MANIFEST_KEYS | {"bed_storage", "deep_storage"}


def test_wet_published_roundtrip_and_continued_physical_steps(tmp_path):
    case = make_case(tmp_path, pickup_kg_per_class=5e-4)
    event = case.event()
    paused = published(advance(event, event.initial_state(case.bed, case.soil), N_BEFORE), event)
    physical(paused)
    assert np.any(host(paused.bed.voxel_column.base) > 0), "no buried deep region: the fixture would be vacuous"
    before, pins = fingerprint(paused), store_pins(case.store_dir)

    path = save_pond_checkpoint(tmp_path / "ck", paused, event, IDENTITY)
    assert fingerprint(paused) == before, "save mutated the state"
    assert store_pins(case.store_dir) == pins, "save touched the deep store"
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".")]

    fresh_event = case.event()
    loaded = load_pond_checkpoint(path, fresh_event, IDENTITY, expected_bed_storage="windowed")
    assert store_pins(case.store_dir) == pins, "load touched the deep store"
    assert_tree_equal(paused, loaded)
    assert loaded.bed.voxel_column.pending_deep is None
    assert logical_column_bytes(loaded.bed.voxel_column) == logical_column_bytes(paused.bed.voxel_column)

    full = advance(event, paused, N_AFTER)
    resumed = advance(fresh_event, loaded, N_AFTER)
    assert_tree_equal(full, resumed)
    assert fingerprint(paused) == before
    a, b = event.audit(published(full, event)), fresh_event.audit(published(resumed, fresh_event))
    assert a["water"]["closes"] and a["sediment"]["closes"] and a == b


def test_layout_manifest_and_deep_record(tmp_path):
    case = make_case(tmp_path, pickup_kg_per_class=5e-4)
    event = case.event()
    paused = published(advance(event, event.initial_state(case.bed, case.soil), 3), event)
    path = save_pond_checkpoint(tmp_path / "ck", paused, event, IDENTITY)
    assert sorted(p.name for p in path.iterdir()) == sorted([pc.MANIFEST_NAME, *pc.PAYLOADS])
    manifest = json.loads((path / pc.MANIFEST_NAME).read_text())
    assert manifest["schema"] == pc.WINDOWED_SCHEMA and set(manifest) == pc.WINDOWED_MANIFEST_KEYS
    assert manifest["bed_storage"] == "windowed" and manifest["status"] == "wet" and manifest["source"] == pc.pond_checkpoint_source_pair()
    store = paused.bed.voxel_column.deep_store
    deep = manifest["deep_storage"]
    assert set(deep) == pc.DEEP_RECORD_KEYS
    assert deep["deep_chunk_encoding"] == "full" and deep["deep_overlay_max_bytes"] is None and deep["store_format_version"] == 1
    assert deep["physical_manifest_digest"] == deep["logical_manifest_digest"] == store.manifest_digest
    assert deep["header_digest"] == store.header.digest() and deep["generation"] == store.version_id
    assert deep["store_dir"] == str(case.store_dir.resolve()) and deep["class_ids"] == list(store.header.class_ids)
    assert deep["window_voxels"] == 4 and deep["n_voxels"] > deep["window_voxels"]
    # ONE bed authority: the native windowed snapshot; no dense column, no buried-array duplicate, no deep chunk copy
    with np.load(path / pc.PAYLOADS[0]) as native:
        assert {"window_kg", "base", "deep_by_class_kg", "prefix_packet", "metadata"} <= set(native.files)
        assert "mass_kg" not in native.files
        assert native["window_kg"].shape[2] == 4
        assert all(native[n].ndim < 4 or native[n].shape[2] == 4 for n in native.files if n != "metadata"), "a logical column"
    with np.load(path / pc.PAYLOADS[1]) as companion:
        assert not [n for n in companion.files if "voxel" in n or "active_mass" in n or "window" in n or "deep" in n]
    stored = {p.name for p in (case.store_dir / "chunks").iterdir()}  # the deep chunks stay in the store, never in the checkpoint
    assert stored and not stored & {p.name for p in path.iterdir()}
    assert not (path / "checkpoint.json").exists()


def test_pending_ledger_below_threshold_is_retained_and_nothing_is_refreshed(tmp_path):
    case = make_case(tmp_path, pickup_kg_per_class=5e-4)
    event = case.event()
    paused = published(advance(event, event.initial_state(case.bed, case.soil), N_BEFORE), event)
    assert float(np.abs(host(paused.bed.ledger.pending_bed_mass_change_kg)).max()) > 0.0, "no pending ledger to carry"
    assert paused.counters.n_commits == 0 and paused.topology.generation == 0
    path = save_pond_checkpoint(tmp_path / "ck", paused, event, IDENTITY)
    fresh_event = case.event()
    loaded = load_pond_checkpoint(path, fresh_event, IDENTITY)
    assert_tree_equal(paused, loaded)
    assert loaded.topology.generation == 0 and loaded.topology.source_sha256 == paused.topology.source_sha256
    np.testing.assert_array_equal(host(loaded.held_source_m), host(paused.held_source_m))
    np.testing.assert_array_equal(host(loaded.bed.ledger.pending_bed_mass_change_kg),
                                  host(paused.bed.ledger.pending_bed_mass_change_kg))
    assert_tree_equal(advance(event, paused, N_AFTER), advance(fresh_event, loaded, N_AFTER))


def test_actual_net_refresh_then_checkpoint_keeps_graph_source_generation_phase_and_cap(tmp_path):
    """The native pickup is far above the pending-elevation trigger: an accepted step commits and refreshes the topology from
    the exact height-packet net gate. The checkpoint stores the HELD source of that generation, rebuilds from it (not from the
    latest bed) and neither commits nor refreshes again."""
    case = make_case(tmp_path, pickup_kg_per_class=0.1)
    event = case.event()
    result = event.evolve(event.initial_state(case.bed, case.soil), 1.0)
    paused = published(result.state, event)
    physical(paused)
    assert paused.counters.n_commits >= 1 and paused.counters.n_refreshes >= 1
    assert paused.topology.generation == paused.counters.n_refreshes
    assert paused.hydraulic.next_dt_cap_s is not None
    path = save_pond_checkpoint(tmp_path / "ck", paused, event, IDENTITY)
    fresh_event = case.event()
    assert fresh_event._initial_topology.generation == 0
    loaded = load_pond_checkpoint(path, fresh_event, IDENTITY)
    for name in ("n_commits", "n_refreshes", "n_final_commits", "n_net_elevation_commits", "n_accepted_steps"):
        assert getattr(loaded.counters, name) == getattr(paused.counters, name), name
    assert loaded.bed.committed_topography.commit_count == paused.bed.committed_topography.commit_count
    assert loaded.topology.generation == paused.topology.generation
    assert (loaded.topology.source_sha256, loaded.topology.topology_sha256) == (paused.topology.source_sha256,
                                                                                 paused.topology.topology_sha256)
    assert loaded.hydraulic.next_dt_cap_s == paused.hydraulic.next_dt_cap_s and loaded.t_s == paused.t_s
    np.testing.assert_array_equal(host(loaded.phase_fraction), host(paused.phase_fraction))
    np.testing.assert_array_equal(host(loaded.phase_position_m), host(paused.phase_position_m))
    np.testing.assert_array_equal(host(loaded.hydraulic.qx_m2_s), host(paused.hydraulic.qx_m2_s))
    assert_tree_equal(paused, loaded)
    a, b = event.evolve(paused, 3.0), fresh_event.evolve(loaded, 3.0)
    assert (a.n_accepted_steps, a.min_accepted_dt_s, a.max_accepted_dt_s) == (b.n_accepted_steps, b.min_accepted_dt_s, b.max_accepted_dt_s)
    assert_tree_equal(published(a.state, event), published(b.state, fresh_event))


def test_save_and_load_do_not_publish_commit_reroute_or_write_into_the_store(tmp_path, monkeypatch):
    import maple.surface.voxels.deep_store as deep_store
    import maple.surface.voxels.windowed_pending as pending

    case = make_case(tmp_path, pickup_kg_per_class=0.1)
    event = case.event()
    paused = published(advance(event, event.initial_state(case.bed, case.soil), N_BEFORE), event)
    pins = store_pins(case.store_dir)

    def forbidden(*_a, **_k):
        raise AssertionError("a checkpoint operation published / applied a deep journal")

    monkeypatch.setattr(deep_store.DeepStoreVersion, "apply_journal", forbidden)
    monkeypatch.setattr(pending, "publish_pending_column", forbidden)
    monkeypatch.setattr(event.__class__, "step", forbidden)
    monkeypatch.setattr(event.__class__, "finalize", forbidden)
    path = save_pond_checkpoint(tmp_path / "ck", paused, event, IDENTITY)
    loaded = load_pond_checkpoint(path, case.event(), IDENTITY)
    assert store_pins(case.store_dir) == pins
    assert loaded.counters.n_commits == paused.counters.n_commits and loaded.status == "wet" and loaded.water.reset_depth is None
    assert tree_hash(path)


def test_max_bytes_bounds_the_resident_payload_not_the_external_store(tmp_path, monkeypatch):
    import maple.surface.voxels.windowed_snapshot as snap

    case = make_case(tmp_path, pickup_kg_per_class=5e-4)
    event = case.event()
    paused = published(advance(event, event.initial_state(case.bed, case.soil), 3), event)
    pins = store_pins(case.store_dir)
    for bound in (1, 1000):
        with pytest.raises(pc.PondCheckpointError, match="max_bytes|bounded uncompressed|exceeds"):
            save_pond_checkpoint(tmp_path / f"small{bound}", paused, event, IDENTITY, max_bytes=bound)
        assert not (tmp_path / f"small{bound}").exists() and not [p for p in tmp_path.iterdir() if p.name.startswith(".")]
    assert store_pins(case.store_dir) == pins
    path = save_pond_checkpoint(tmp_path / "ck", paused, event, IDENTITY)
    size = sum(p.stat().st_size for p in path.iterdir())
    # a bound far below the payload is refused from the manifest / archive directory, before the native loader runs
    monkeypatch.setattr(snap, "load_windowed_state_snapshot", lambda *a, **k: (_ for _ in ()).throw(AssertionError("native load ran")))
    before = tree_hash(path)
    with pytest.raises(pc.PondCheckpointError, match="max_bytes|bounded uncompressed|exceed"):
        load_pond_checkpoint(path, case.event(), IDENTITY, max_bytes=max(1, size // 4))
    assert tree_hash(path) == before
    for bad in (0, -1, True, 1.5, None):
        with pytest.raises(pc.PondCheckpointError, match="max_bytes"):
            load_pond_checkpoint(path, case.event(), IDENTITY, max_bytes=bad)


def test_save_estimate_is_resident_only(tmp_path, monkeypatch):
    """The cap check never forms the logical (cells x voxels x classes) column: densifying is forbidden during save."""
    import maple.surface.voxels.windowed_inventory as inventory

    case = make_case(tmp_path, pickup_kg_per_class=5e-4)
    event = case.event()
    paused = published(advance(event, event.initial_state(case.bed, case.soil), 2), event)
    monkeypatch.setattr(inventory, "windowed_to_dense",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("save densified the column")))
    path = save_pond_checkpoint(tmp_path / "ck", paused, event, IDENTITY)
    assert path.is_dir()


def test_adaptive_cap_clock_and_knots_survive_evolve(tmp_path):
    case = make_case(tmp_path, pickup_kg_per_class=5e-4, rain_end_s=1.5)
    event = case.event()
    first = event.evolve(event.initial_state(case.bed, case.soil), 1.0)
    assert first.reached_target
    paused = published(first.state, event)
    assert paused.hydraulic.next_dt_cap_s is not None
    path = save_pond_checkpoint(tmp_path / "ck", paused, event, IDENTITY)
    fresh_event = case.event()
    loaded = load_pond_checkpoint(path, fresh_event, IDENTITY)
    assert loaded.hydraulic.next_dt_cap_s == paused.hydraulic.next_dt_cap_s and loaded.t_s == paused.t_s
    a, b = event.evolve(paused, 3.0), fresh_event.evolve(loaded, 3.0)
    assert a.reached_target and b.reached_target
    assert (a.n_accepted_steps, a.min_accepted_dt_s, a.max_accepted_dt_s) == (b.n_accepted_steps, b.min_accepted_dt_s, b.max_accepted_dt_s)
    assert_tree_equal(published(a.state, event), published(b.state, fresh_event))


def test_numba_hydraulics_roundtrip_and_numpy_numba_are_not_interchangeable(tmp_path):
    pytest.importorskip("numba")
    from dataclasses import replace

    case = make_case(tmp_path, pickup_kg_per_class=5e-4, backend="numba")
    event = case.event()
    paused = published(advance(event, event.initial_state(case.bed, case.soil), 4), event)
    path = save_pond_checkpoint(tmp_path / "ck", paused, event, IDENTITY)
    fresh = case.event()
    loaded = load_pond_checkpoint(path, fresh, IDENTITY)
    assert_tree_equal(paused, loaded)
    assert_tree_equal(advance(event, paused, 4), advance(fresh, loaded, 4))
    with pytest.raises(pc.PondCheckpointError, match="hydraulic backend"):
        load_pond_checkpoint(path, case.event(control=replace(case.control, backend="numpy")), IDENTITY)
