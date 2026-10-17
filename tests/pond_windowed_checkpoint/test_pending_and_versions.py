"""Pending deep journals are refused without I/O or publication; explicitly published states checkpoint; the external immutable
store versions are retained; full (v1) and overlay (v2) version bindings (nothing here was run by its author)."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from maple_syrup import pond_checkpoint as pc
from maple_syrup.pond_checkpoint import PondCheckpointError, load_pond_checkpoint, save_pond_checkpoint
from .wfixture import (
    IDENTITY,
    advance,
    assert_tree_equal,
    fingerprint,
    logical_column_bytes,
    make_case,
    host,
    published,
    store_pins,
    tree_hash,
)


def with_pending_journal(state, event):
    """The accepted state with a REAL native pending deep journal (one canonical record rewriting a buried voxel with its own
    current value), attached through MAPLE's `attach_pending_journal` on a copy of the column. Returns `(state, column)`."""
    import dataclasses

    from maple.surface.voxels.deep_store import DeepJournal
    from maple.surface.voxels.windowed_pending import attach_pending_journal

    column = state.bed.voxel_column
    store = column.deep_store
    nc = store.header.n_classes
    value = np.asarray(store.gather_columns(np.array([0], dtype=np.int64)))[0, 0].reshape(1, nc).astype(np.float64)
    journal = DeepJournal.from_records(np.array([0], dtype=np.int64), np.array([0], dtype=np.int64), value, header=store.header)
    copy = dataclasses.replace(column)
    object.__setattr__(copy, "height_packet", column.height_packet)  # same arrays: the packets stay bound to this base
    pending = attach_pending_journal(copy, journal)
    assert pending.pending_deep is not None
    return replace(state, bed=replace(state.bed, voxel_column=pending)), pending


@pytest.fixture
def scenario(tmp_path):
    case = make_case(tmp_path, pickup_kg_per_class=5e-4)
    event = case.event()
    state = published(advance(event, event.initial_state(case.bed, case.soil), 4), event)
    return case, event, state


def test_pending_journal_is_refused_without_io_publication_or_mutation(tmp_path, scenario, monkeypatch):
    import maple.surface.voxels.deep_store as deep_store
    import maple.surface.voxels.windowed_snapshot as snap

    case, event, state = scenario
    pending_state, column = with_pending_journal(state, event)
    pins, before = store_pins(case.store_dir), fingerprint(pending_state.bed.voxel_column.window_kg)
    listing = sorted(p.name for p in tmp_path.iterdir())

    def forbidden(*_a, **_k):
        raise AssertionError("I/O or publication happened for a pending column")

    monkeypatch.setattr(snap, "save_windowed_state_snapshot", forbidden)
    monkeypatch.setattr(deep_store.DeepStoreVersion, "apply_journal", forbidden)
    with pytest.raises(PondCheckpointError, match="publish"):
        save_pond_checkpoint(tmp_path / "ck", pending_state, event, IDENTITY)
    assert sorted(p.name for p in tmp_path.iterdir()) == listing and not (tmp_path / "ck").exists()
    assert store_pins(case.store_dir) == pins
    assert pending_state.bed.voxel_column.pending_deep is column.pending_deep is not None  # the journal was not consumed
    assert fingerprint(pending_state.bed.voxel_column.window_kg) == before


def test_explicit_publication_then_checkpoint_and_old_versions_are_retained(tmp_path, scenario):
    case, event, state = scenario
    old_path = save_pond_checkpoint(tmp_path / "old", state, event, IDENTITY)
    old_digest = state.bed.voxel_column.manifest_digest
    pins_old = store_pins(case.store_dir)

    pending_state, _ = with_pending_journal(state, event)
    new_state = event.publish_storage(pending_state)  # the explicit, caller-owned core publication
    new_column = new_state.bed.voxel_column
    assert new_column.pending_deep is None and new_column.generation == state.bed.voxel_column.generation + 1
    assert new_column.manifest_digest != old_digest
    pins_new = store_pins(case.store_dir)
    assert set(pins_old) < set(pins_new) and all(pins_new[k] == v for k, v in pins_old.items()), "a version was altered or removed"

    new_path = save_pond_checkpoint(tmp_path / "new", new_state, event, IDENTITY)
    assert store_pins(case.store_dir) == pins_new
    # both checkpoints name immutable versions of the SAME external store and both still load
    fresh = case.event()
    older = load_pond_checkpoint(old_path, fresh, IDENTITY)
    newer = load_pond_checkpoint(new_path, case.event(), IDENTITY)
    assert older.bed.voxel_column.manifest_digest == old_digest
    assert newer.bed.voxel_column.manifest_digest == new_column.manifest_digest
    assert older.bed.voxel_column.generation + 1 == newer.bed.voxel_column.generation
    assert logical_column_bytes(older.bed.voxel_column) == logical_column_bytes(newer.bed.voxel_column)  # a no-change journal
    assert_tree_equal(state, older)
    assert store_pins(case.store_dir) == pins_new, "loading changed the store"


def test_boundary_mode_pending_save_refused_then_explicit_publication_and_exact_resume(tmp_path):
    """`deep_publication="boundary"` (explicit positive high-water records) after a real mobile deposition: the pending native
    journal is refused by the checkpoint, published only by the caller (`PondEvent.publish_storage`), then saved, loaded and
    continued exactly like the uninterrupted published run."""
    case = make_case(tmp_path, mobile_kg=5e-3, pickup_kg_per_class=5e-4,
                     control_kw={"deep_publication": "boundary", "deep_publication_high_water_records": 1000})
    assert case.control.bed_storage == "windowed" and case.control.deep_publication == "boundary"
    event = case.event()
    state = advance(event, event.initial_state(case.bed, case.soil), 6)
    assert float(host(state.sediment.wet_actual).sum() + host(state.bed.water.mobile_mass_by_cell_class_kg).sum()) > 0.0, \
        "no mobile sediment: the boundary-mode fixture does nothing"
    if state.bed.voxel_column.pending_deep is None:
        # the small fixture's deposition may stay inside the resident window / active layer: attach a REAL native journal
        # (canonical record through MAPLE's attach_pending_journal) so the pending path is exercised either way
        state, _ = with_pending_journal(state, event)
    assert state.bed.voxel_column.pending_deep is not None
    pins, listing = store_pins(case.store_dir), sorted(p.name for p in tmp_path.iterdir())
    with pytest.raises(PondCheckpointError, match="publish"):
        save_pond_checkpoint(tmp_path / "ck", state, event, IDENTITY)
    assert sorted(p.name for p in tmp_path.iterdir()) == listing and store_pins(case.store_dir) == pins

    accepted = event.publish_storage(state)  # explicit; no physical change
    assert accepted.bed.voxel_column.pending_deep is None
    assert accepted.t_s == state.t_s and accepted.counters.n_accepted_steps == state.counters.n_accepted_steps
    path = save_pond_checkpoint(tmp_path / "ck", accepted, event, IDENTITY)
    fresh = case.event()
    loaded = load_pond_checkpoint(path, fresh, IDENTITY)
    assert_tree_equal(accepted, loaded)
    assert_tree_equal(published(advance(event, accepted, 4), event), published(advance(fresh, loaded, 4), fresh))


def test_removing_the_referenced_version_is_refused_not_silently_recovered(tmp_path, scenario):
    case, event, state = scenario
    path = save_pond_checkpoint(tmp_path / "ck", state, event, IDENTITY)
    manifest = state.bed.voxel_column.deep_store.manifest_path
    saved = manifest.read_bytes()
    manifest.unlink()
    before = tree_hash(path)
    with pytest.raises(PondCheckpointError, match="deep-store|windowed snapshot cannot be loaded"):
        load_pond_checkpoint(path, case.event(), IDENTITY)
    assert tree_hash(path) == before
    manifest.write_bytes(saved)  # the retained immutable version makes the same checkpoint loadable again
    assert load_pond_checkpoint(path, case.event(), IDENTITY).counters.n_accepted_steps == 4


def edit(path, change):
    meta = json.loads((path / pc.MANIFEST_NAME).read_text())
    change(meta)
    from maple_syrup.checkpoint import sha256_file

    meta["files"] = {n: sha256_file(path / n) for n in pc.PAYLOADS}
    (path / pc.MANIFEST_NAME).write_text(json.dumps(pc._seal_manifest(meta), indent=1, sort_keys=True))


@pytest.fixture
def overlay_scenario(tmp_path):
    case = make_case(tmp_path, pickup_kg_per_class=5e-4, encoding="overlay")
    event = case.event()
    state = published(advance(event, event.initial_state(case.bed, case.soil), 4), event)
    assert state.bed.voxel_column.deep_store.format_version == 2
    return case, event, state


def test_overlay_version_roundtrip_binds_physical_and_logical_identity(tmp_path, overlay_scenario):
    case, event, state = overlay_scenario
    store = state.bed.voxel_column.deep_store
    assert store.manifest_digest != store.logical_manifest_digest  # a v2 version has two distinct identities
    pins = store_pins(case.store_dir)
    path = save_pond_checkpoint(tmp_path / "ck", state, event, IDENTITY)
    manifest = json.loads((path / pc.MANIFEST_NAME).read_text())
    deep = manifest["deep_storage"]
    assert deep["deep_chunk_encoding"] == "overlay" and deep["store_format_version"] == 2
    assert deep["physical_manifest_digest"] == store.manifest_digest and deep["logical_manifest_digest"] == store.logical_manifest_digest
    fresh = case.event()
    loaded = load_pond_checkpoint(path, fresh, IDENTITY)
    assert loaded.bed.voxel_column.deep_store.format_version == 2
    assert_tree_equal(state, loaded)
    assert_tree_equal(advance(event, state, 4), advance(fresh, loaded, 4))
    assert store_pins(case.store_dir) == pins


@pytest.mark.parametrize("variant", ["physical_is_v1", "logical_swapped", "format_downgraded", "genesis_header"])
def test_overlay_version_bindings_are_enforced(tmp_path, overlay_scenario, variant):
    case, event, state = overlay_scenario
    path = save_pond_checkpoint(tmp_path / "ck", state, event, IDENTITY)

    def change(m):
        deep = m["deep_storage"]
        if variant == "physical_is_v1":
            deep["physical_manifest_digest"] = case.v1_manifest_digest  # the genesis version of the same store
        elif variant == "logical_swapped":
            deep["logical_manifest_digest"] = deep["physical_manifest_digest"]
        elif variant == "format_downgraded":
            deep["store_format_version"] = 1
        else:
            deep["header_digest"] = "0" * 64

    edit(path, change)
    before = tree_hash(path)
    with pytest.raises(PondCheckpointError):
        load_pond_checkpoint(path, case.event(), IDENTITY)
    assert tree_hash(path) == before


def test_full_and_overlay_contexts_do_not_cross(tmp_path, overlay_scenario):
    case, event, state = overlay_scenario
    path = save_pond_checkpoint(tmp_path / "ck", state, event, IDENTITY)
    full_ctx = replace(case.ctx, deep_chunk_encoding="full")
    with pytest.raises(PondCheckpointError, match="binding|context_sha256|deep_storage_option"):
        load_pond_checkpoint(path, case.event(ctx=full_ctx), IDENTITY)
    bounded = replace(case.ctx, deep_overlay_max_bytes=10**6)
    with pytest.raises(PondCheckpointError, match="binding|context_sha256|deep_storage_option"):
        load_pond_checkpoint(path, case.event(ctx=bounded), IDENTITY)


def test_full_context_cannot_save_an_overlay_version(tmp_path, overlay_scenario):
    case, event, state = overlay_scenario
    full_event = case.event(ctx=replace(case.ctx, deep_chunk_encoding="full"))
    state_full_cfg = replace(state, config_sha256=full_event.config_sha256)
    with pytest.raises(PondCheckpointError, match="overlay"):
        save_pond_checkpoint(tmp_path / "ck", state_full_cfg, full_event, IDENTITY)
    assert not (tmp_path / "ck").exists() and not [p for p in tmp_path.iterdir() if p.name.startswith(".")]


def test_full_checkpoint_cannot_be_read_as_overlay(tmp_path, scenario):
    case, event, state = scenario
    path = save_pond_checkpoint(tmp_path / "ck", state, event, IDENTITY)
    with pytest.raises(PondCheckpointError):
        load_pond_checkpoint(path, case.event(ctx=replace(case.ctx, deep_chunk_encoding="overlay")), IDENTITY)
