"""Refusals of the windowed pond checkpoint: manifest and deep-record edits, source / backend / context / encoding / class
order, native snapshot tampering, a missing or corrupt deep chunk, byte bounds, write failure; and non-mutation (not run by its
author). Every edit that should reach a deeper validator re-seals the hashes, so the refusal comes from THAT validator."""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from maple_syrup import pond_checkpoint as pc
from maple_syrup.checkpoint import sha256_file
from maple_syrup.pond_checkpoint import PondCheckpointError, load_pond_checkpoint, save_pond_checkpoint

from .wfixture import IDENTITY, advance, fingerprint, make_case, published, store_pins, tree_hash


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    root = tmp_path_factory.mktemp("wpond_ck_base")
    case = make_case(root, pickup_kg_per_class=5e-4)
    event = case.event()
    state = published(advance(event, event.initial_state(case.bed, case.soil), 5), event)
    return case, event, state, save_pond_checkpoint(root / "base", state, event, IDENTITY)


@pytest.fixture
def ck(base, tmp_path):
    case, _event, _state, path = base
    target = tmp_path / "ck"
    shutil.copytree(path, target)
    return case, target


def rewrite(path: Path, *, manifest=None, companion=None, native=None, refresh_files=True) -> None:
    """Edit the manifest, the companion arrays and / or the native snapshot arrays of a checkpoint and re-seal."""
    meta = json.loads((path / pc.MANIFEST_NAME).read_text())
    for name, edit in ((pc.PAYLOADS[1], companion), (pc.PAYLOADS[0], native)):
        if edit is not None:
            with np.load(path / name, allow_pickle=True) as z:
                data = {k: z[k].copy() for k in z.files}
            edit(data, meta)
            np.savez(path / name, **data)
    if manifest is not None:
        manifest(meta)
    if refresh_files:
        meta["files"] = {n: sha256_file(path / n) for n in pc.PAYLOADS}
    (path / pc.MANIFEST_NAME).write_text(json.dumps(pc._seal_manifest(meta), indent=1, sort_keys=True))


def refused(case, path, match, **kw):
    event = case.event()
    before = tree_hash(path)
    pins = store_pins(case.store_dir)
    with pytest.raises(PondCheckpointError, match=match):
        load_pond_checkpoint(path, event, kw.pop("identity", IDENTITY), **kw)
    assert tree_hash(path) == before, "a refused load modified the checkpoint"
    assert store_pins(case.store_dir) == pins, "a refused load modified the deep store"


def test_untouched_copy_loads(ck):
    case, path = ck
    assert load_pond_checkpoint(path, case.event(), IDENTITY).counters.n_accepted_steps == 5


# --- manifest-level refusals: nothing of the payload is read -------------------------------------------------------------
MANIFEST_EDITS = [
    ("schema_unknown", lambda m: m.update(schema="future/version"), "keys|schema"),
    ("schema_relabelled_dense", lambda m: m.update(schema=pc.SCHEMA), "keys"),
    ("unknown_manifest_key", lambda m: m.update(extra=1), "keys"),
    ("bed_storage_dense", lambda m: m.update(bed_storage="dense"), "bed_storage"),
    ("unknown_payload_key", lambda m: m["payload"].update(extra=1), "payload keys"),
    ("status_unknown", lambda m: m.update(status="finished"), "status"),
    ("array_backend", lambda m: m.update(array_backend="cupy"), "cross-backend"),
    ("hydraulic_backend", lambda m: m.update(hydraulic_backend="numba"), "hydraulic backend"),
    ("binding_context", lambda m: m["binding"].update(context_sha256="0" * 64), "binding"),
    ("binding_shape", lambda m: m["binding"].update(shape=[9, 9]), "binding"),
    ("binding_storage", lambda m: m["binding"].update(bed_storage="dense"), "binding"),
    ("binding_deep_option", lambda m: m["binding"]["deep_storage_option"].update(deep_chunk_encoding="overlay"), "binding"),
]


@pytest.mark.parametrize(("name", "edit", "match"), MANIFEST_EDITS, ids=[e[0] for e in MANIFEST_EDITS])
def test_manifest_edit_is_refused(ck, name, edit, match):
    case, path = ck
    rewrite(path, manifest=edit)
    refused(case, path, match)


def _deep(**changes):
    def edit(m):
        m["deep_storage"].update(changes)

    return edit


DEEP_EDITS = [
    ("encoding_overlay", _deep(deep_chunk_encoding="overlay"), "cross-encoding|differs"),
    ("overlay_bound_with_full", _deep(deep_overlay_max_bytes=5), "invalid"),
    ("format_v2_under_full_context", _deep(store_format_version=2), "overlay"),
    ("format_bool", _deep(store_format_version=True), "store_format_version"),
    ("format_str", _deep(store_format_version="1"), "store_format_version"),
    ("v1_with_distinct_logical", _deep(logical_manifest_digest="a" * 64), "equal physical and logical"),
    ("digest_not_hex", _deep(physical_manifest_digest="XYZ", logical_manifest_digest="XYZ"), "64-hex"),
    ("other_version", _deep(physical_manifest_digest="a" * 64, logical_manifest_digest="a" * 64), "provenance|recorded"),
    ("header_digest", _deep(header_digest="b" * 64), "recorded"),
    ("store_dir_relative", _deep(store_dir="store"), "absolute"),
    ("store_dir_elsewhere", _deep(store_dir="/nonexistent/elsewhere"), "recorded"),
    ("generation", _deep(generation=7), "recorded"),
    ("n_voxels_bool", _deep(n_voxels=True), "n_voxels"),
    ("window_voxels", _deep(window_voxels=3), "differs from the expected|packet|window"),
    ("chunk_columns", _deep(chunk_columns=32), "chunk count|recorded"),
    ("n_chunks", _deep(n_chunks=3), "chunk count"),
    ("class_ids_reversed", lambda m: m["deep_storage"].update(class_ids=m["deep_storage"]["class_ids"][::-1]), "class ordering"),
    ("packet_terms", lambda m: m["deep_storage"].update(packet_terms=m["deep_storage"]["packet_terms"] + 1), "packet_terms"),
    ("unknown_deep_key", _deep(extra=1), "deep_storage record keys"),
    ("deep_record_missing_key", lambda m: m["deep_storage"].pop("header_digest"), "deep_storage record keys"),
    ("deep_record_not_dict", lambda m: m.update(deep_storage=[]), "deep_storage record keys"),
]


@pytest.mark.parametrize(("name", "edit", "match"), DEEP_EDITS, ids=[e[0] for e in DEEP_EDITS])
def test_deep_storage_record_edit_is_refused(ck, name, edit, match):
    case, path = ck
    rewrite(path, manifest=edit)
    refused(case, path, match)


HUGE = [
    ("huge_window", dict(n_voxels=2**30, window_voxels=2**29)),  # window bytes alone exceed max_bytes
    ("huge_voxels_small_window", dict(n_voxels=2**30, window_voxels=4)),  # native packet-term bound, no O(n) walk
    ("huge_voxels_window_one", dict(n_voxels=2**31 - 1, window_voxels=1)),
    ("just_above_native_bound", dict(n_voxels=40000, window_voxels=4)),
]


@pytest.mark.parametrize(("name", "geometry"), HUGE, ids=[h[0] for h in HUGE])
def test_huge_declared_geometry_is_refused_quickly_before_any_payload_is_opened(ck, monkeypatch, name, geometry):
    """The untrusted geometry is bounded by plain integer arithmetic and the native term bound BEFORE any native size
    function (`packet_terms_for` walks a tree that grows with n_voxels) is called; a call would fail the test."""
    import time

    import maple.surface.voxels.windowed_prefix as prefix
    import maple.surface.voxels.windowed_snapshot as snap

    case, path = ck
    rewrite(path, manifest=lambda m: m["deep_storage"].update(geometry), refresh_files=False)
    (path / pc.PAYLOADS[0]).write_bytes(b"garbage that must never be opened")

    def forbidden(*_a, **_k):
        raise AssertionError("an expensive or payload-reading native call ran on untrusted geometry")

    monkeypatch.setattr(prefix, "packet_terms_for", forbidden)
    monkeypatch.setattr(snap, "load_windowed_state_snapshot", forbidden)
    start = time.monotonic()
    refused(case, path, "refused from the manifest alone")
    assert time.monotonic() - start < 30.0, "the refusal was not cheap"


def test_source_pair_mismatch_is_refused_before_any_payload_is_read(ck):
    case, path = ck
    for field in ("maple_syrup_tree_sha256", "maple_package_sha256"):
        rewrite(path, manifest=lambda m, f=field: m["source"].update({f: "f" * 64}), refresh_files=False)
        (path / pc.PAYLOADS[1]).write_bytes(b"garbage that must never be opened")
        refused(case, path, "source pair")


def test_identity_mismatch_is_refused_before_any_payload_is_read(ck):
    case, path = ck
    (path / pc.PAYLOADS[0]).write_bytes(b"garbage that must never be opened")
    for key in ("case", "forcing", "source", "options"):
        expected = json.loads(json.dumps(IDENTITY))
        expected[key] = "changed"
        refused(case, path, "identity", identity=expected)
    with pytest.raises(PondCheckpointError, match="identity"):
        load_pond_checkpoint(path, case.event(), {"bad": float("nan")})


def test_manifest_modified_without_reseal_and_non_strict_json_are_refused(ck):
    case, path = ck
    text = (path / pc.MANIFEST_NAME).read_text()
    meta = json.loads(text)
    meta["identity"]["case"] = "edited"
    (path / pc.MANIFEST_NAME).write_text(json.dumps(meta))
    refused(case, path, "manifest hash")
    (path / pc.MANIFEST_NAME).write_text(text.replace('"schema"', '"schema": NaN, "schema"', 1))
    refused(case, path, "strict JSON")
    (path / pc.MANIFEST_NAME).write_text(text[:-2] + ', "schema": "x"}')
    refused(case, path, "strict JSON")


def test_complete_status_without_allow_complete_is_refused(ck):
    case, path = ck
    rewrite(path, manifest=lambda m: m.update(status="complete"))
    refused(case, path, "completed")


def test_directory_layout_is_exact(ck):
    case, path = ck
    (path / "extra.txt").write_text("x")
    refused(case, path, "exactly the manifest")
    (path / "extra.txt").unlink()
    (path / pc.PAYLOADS[1]).unlink()
    refused(case, path, "exactly the manifest")


# --- event binding -------------------------------------------------------------------------------------------------------
def test_native_context_difference_is_refused_even_if_event_config_digest_is_equal(ck):
    case, path = ck
    other_ctx = replace(case.ctx, initial_available_fraction=np.full_like(case.ctx.initial_available_fraction, 0.5))
    with pytest.raises(PondCheckpointError, match="context_sha256"):
        load_pond_checkpoint(path, case.event(ctx=other_ctx), IDENTITY)


def test_event_control_storage_and_publication_mode_are_bound(ck):
    case, path = ck
    with pytest.raises(PondCheckpointError, match="binding|options_sha256|control_storage"):
        load_pond_checkpoint(path, case.event(control=replace(case.control, bed_storage="dense")), IDENTITY)
    # a VALID altered control (boundary mode needs an explicit positive high-water record count) reaches the binding check
    boundary = replace(case.control, deep_publication="boundary", deep_publication_high_water_records=100)
    with pytest.raises(PondCheckpointError, match="binding|options_sha256|control_storage"):
        load_pond_checkpoint(path, case.event(control=boundary), IDENTITY)
    # the high-water count is only valid together with boundary mode, so change BOTH from the 'step' baseline
    other = replace(case.control, deep_publication="boundary", deep_publication_high_water_records=200)
    with pytest.raises(PondCheckpointError, match="binding|options_sha256|control_storage"):
        load_pond_checkpoint(path, case.event(control=other), IDENTITY)


def test_invalid_controls_fail_at_the_event_constructor_before_any_load(ck):
    """A control that is itself invalid is refused by `PondEvent` (a core guard), so no load is even attempted."""
    from maple_syrup.pond_event import PondEventError

    case, path = ck
    with pytest.raises(PondEventError, match="high_water"):
        case.event(control=replace(case.control, deep_publication="boundary"))
    with pytest.raises(PondEventError, match="windowed"):
        case.event(control=replace(case.control, bed_storage="dense", deep_publication="boundary",
                                   deep_publication_high_water_records=5))


def test_windowed_state_is_refused_by_a_dense_strict_validation(base):
    """The dense default of the core's `require_supported_bed` stays strict: only the windowed path opts in."""
    from maple_syrup.pond_bed import PondBedError, require_supported_bed

    case, _event, state, _path = base
    with pytest.raises(PondBedError):
        require_supported_bed(state.bed, case.ctx)
    require_supported_bed(state.bed, case.ctx, allow_windowed=True)


def test_numba_hydraulics_is_not_numpy(ck):
    pytest.importorskip("numba")
    case, path = ck
    with pytest.raises(PondCheckpointError, match="hydraulic backend"):
        load_pond_checkpoint(path, case.event(control=replace(case.control, backend="numba")), IDENTITY)


# --- payload / native snapshot tampering ---------------------------------------------------------------------------------
def test_payload_hash_mismatch_and_truncated_archive(ck):
    case, path = ck
    raw = bytearray((path / pc.PAYLOADS[0]).read_bytes())
    raw[len(raw) // 2] ^= 0xFF
    (path / pc.PAYLOADS[0]).write_bytes(bytes(raw))
    refused(case, path, "payload hash mismatch")
    rewrite(path)  # hashes now match the corrupted bytes: the archive / header audit must refuse
    with pytest.raises(PondCheckpointError):
        load_pond_checkpoint(path, case.event(), IDENTITY)
    (path / pc.PAYLOADS[0]).write_bytes((path / pc.PAYLOADS[0]).read_bytes()[:100])
    rewrite(path)
    refused(case, path, "not a valid archive|malformed|cannot")


def meta_of(data):
    return json.loads(str(data["metadata"].item()))


def put_meta(data, meta):
    data["metadata"] = np.array(json.dumps(meta, sort_keys=True))


NATIVE_EDITS = [
    ("extra_member", lambda d, m: d.update(extra=np.zeros(3)), "members .* differ"),
    ("missing_member", lambda d, m: d.pop("bound_mass_kg"), "members .* differ"),
    ("window_shape", lambda d, m: d.update(window_kg=d["window_kg"][:, :, :2].copy()), "differs from the expected"),
    ("packet_dtype", lambda d, m: d.update(prefix_packet=d["prefix_packet"].astype(np.float32)), "differs from the expected"),
    ("base_dtype", lambda d, m: d.update(base=d["base"].astype(np.int64)), "differs from the expected"),
    ("metadata_object_dtype", lambda d, m: d.update(metadata=np.array({"a": 1}, dtype=object)), "not allowed"),
    ("metadata_not_scalar", lambda d, m: d.update(metadata=np.array(["a", "b"])), "0-d Unicode"),
    ("class_ids_reversed", lambda d, m: put_meta(d, {**meta_of(d), "class_ids": meta_of(d)["class_ids"][::-1]}),
     "windowed snapshot cannot be loaded"),
    ("metadata_schema", lambda d, m: put_meta(d, {**meta_of(d), "schema": "x"}), "windowed snapshot cannot be loaded"),
    ("metadata_provenance", lambda d, m: put_meta(d, {**meta_of(d), "provenance": {}}), "cannot be loaded|provenance"),
    ("metadata_time", lambda d, m: put_meta(d, {**meta_of(d), "time_s": meta_of(d)["time_s"] + 1.0}), "clock|differ"),
    ("metadata_step", lambda d, m: put_meta(d, {**meta_of(d), "step": meta_of(d)["step"] + 1}), "clock|step|differ"),
    ("packet_values", lambda d, m: d.update(prefix_packet=np.where(np.arange(d["prefix_packet"].size).reshape(
        d["prefix_packet"].shape) == 0, d["prefix_packet"] + 1.0, d["prefix_packet"])), "cannot be loaded"),
    ("window_values", lambda d, m: d.update(window_kg=d["window_kg"] * 1.5), "cannot be loaded"),
    ("base_values", lambda d, m: d.update(base=np.maximum(d["base"] - 1, 0).astype(np.int32)), "cannot be loaded"),
]


@pytest.mark.parametrize(("name", "edit", "match"), NATIVE_EDITS, ids=[e[0] for e in NATIVE_EDITS])
def test_native_snapshot_tampering_is_refused(ck, name, edit, match):
    case, path = ck
    rewrite(path, native=edit)
    refused(case, path, match)


def test_unicode_metadata_is_size_bounded_before_the_native_loader_runs(ck, monkeypatch):
    import maple.surface.voxels.windowed_snapshot as snap

    case, path = ck
    monkeypatch.setattr(pc, "_MAX_NATIVE_META_BYTES", 16)
    monkeypatch.setattr(snap, "load_windowed_state_snapshot", lambda *a, **k: (_ for _ in ()).throw(AssertionError("native load")))
    refused(case, path, "bounded 0-d Unicode")


def test_native_metadata_pointing_at_a_copy_of_the_store_is_refused_by_location(ck, tmp_path):
    case, path = ck
    other = tmp_path / "copy_of_store"
    shutil.copytree(case.store_dir, other)
    rewrite(path, native=lambda d, m: put_meta(d, {**meta_of(d), "store_dir": str(other.resolve())}))
    refused(case, path, "not the one recorded")


def test_companion_tampering_is_refused(ck):
    case, path = ck

    def drop(d, m):
        d.pop("hyd_depth")
        m["payload"]["arrays"].pop("hyd_depth")

    rewrite(path, companion=drop)
    refused(case, path, "missing required arrays")


def test_companion_extra_array_and_shape_edit_are_refused(base, tmp_path):
    case, _event, _state, src = base
    path = tmp_path / "ck"
    shutil.copytree(src, path)

    def extra(d, m):
        d["extra_array"] = np.zeros(2)
        m["payload"]["arrays"]["extra_array"] = {"dtype": "<f8", "shape": [2]}

    rewrite(path, companion=extra)
    refused(case, path, "unused or missing companion arrays")
    shutil.rmtree(path)
    shutil.copytree(src, path)

    def reshape(d, m):
        d["hyd_depth"] = d["hyd_depth"][:-1].copy()
        m["payload"]["arrays"]["hyd_depth"]["shape"] = list(d["hyd_depth"].shape)

    rewrite(path, companion=reshape)
    refused(case, path, "declared dtype / shape differ")


# --- the external deep store ---------------------------------------------------------------------------------------------
@pytest.fixture
def multichunk(tmp_path):
    """9 x 8 = 72 cells over 64-column chunks: two real deep chunks, so a single chunk can go missing."""
    case = make_case(tmp_path, ny=9, nx=8, pickup_kg_per_class=5e-4, store_name="mstore")
    event = case.event()
    state = published(advance(event, event.initial_state(case.bed, case.soil), 2), event)
    path = save_pond_checkpoint(tmp_path / "mck", state, event, IDENTITY)
    assert state.bed.voxel_column.deep_store.n_chunks == 2
    return case, state, path


def test_missing_and_corrupt_deep_chunks_are_refused_and_nothing_is_modified(multichunk):
    case, state, path = multichunk
    store = state.bed.voxel_column.deep_store
    chunk = store.chunk_path(store.chunks[1])
    original = chunk.read_bytes()
    before = tree_hash(path)
    chunk.unlink()
    with pytest.raises(PondCheckpointError, match="windowed snapshot cannot be loaded"):
        load_pond_checkpoint(path, case.event(), IDENTITY)
    assert tree_hash(path) == before and not chunk.exists()
    chunk.write_bytes(original)
    flipped = bytearray(original)
    flipped[len(flipped) // 2] ^= 0x01
    chunk.write_bytes(bytes(flipped))
    with pytest.raises(PondCheckpointError, match="windowed snapshot cannot be loaded"):
        load_pond_checkpoint(path, case.event(), IDENTITY)
    chunk.write_bytes(original)  # the retained, unmodified version loads again
    assert load_pond_checkpoint(path, case.event(), IDENTITY).counters.n_accepted_steps == 2


def test_corrupt_store_manifest_is_refused(multichunk):
    case, state, path = multichunk
    manifest = state.bed.voxel_column.deep_store.manifest_path
    original = manifest.read_bytes()
    mode = manifest.stat().st_mode
    manifest.chmod(0o644)  # the native store publishes immutable (read-only) files; this OWNED fixture copy is tampered on purpose
    try:
        manifest.write_bytes(original.replace(b"version", b"vers1on", 1))
        with pytest.raises(PondCheckpointError, match="windowed snapshot cannot be loaded"):
            load_pond_checkpoint(path, case.event(), IDENTITY)
    finally:
        manifest.write_bytes(original)
        manifest.chmod(mode & 0o777)
    load_pond_checkpoint(path, case.event(), IDENTITY)


# --- write failure: only the own staging directory is removed ------------------------------------------------------------
@pytest.fixture
def wet(tmp_path):
    case = make_case(tmp_path, pickup_kg_per_class=5e-4)
    event = case.event()
    return case, event, published(advance(event, event.initial_state(case.bed, case.soil), 3), event)


def test_existing_destination_is_never_overwritten(wet, tmp_path):
    case, event, state = wet
    path = save_pond_checkpoint(tmp_path / "ck", state, event, IDENTITY)
    before = tree_hash(path)
    with pytest.raises(PondCheckpointError, match="already exists"):
        save_pond_checkpoint(path, state, event, IDENTITY)
    assert tree_hash(path) == before
    with pytest.raises(PondCheckpointError, match="parent directory"):
        save_pond_checkpoint(tmp_path / "missing" / "ck", state, event, IDENTITY)


@pytest.mark.parametrize("stage", ["native_snapshot", "manifest", "rename"])
def test_late_write_failure_cleans_only_the_staging_directory(wet, tmp_path, monkeypatch, stage):
    import maple.surface.voxels.windowed_snapshot as snap

    case, event, state = wet
    bystander = tmp_path / ".unrelated_hidden_dir"
    bystander.mkdir()
    (bystander / "keep.txt").write_text("keep")
    pins, before = store_pins(case.store_dir), fingerprint(state)
    listing = sorted(p.name for p in tmp_path.iterdir())
    if stage == "native_snapshot":
        real = snap.save_windowed_state_snapshot

        def partial(column, active, time_s, target, **kw):
            real(column, active, time_s, target, **kw)
            raise OSError("disk full after the native snapshot")

        monkeypatch.setattr(snap, "save_windowed_state_snapshot", partial)
    elif stage == "manifest":
        monkeypatch.setattr(pc, "_write_manifest", lambda *a, **k: (_ for _ in ()).throw(OSError("manifest write failed")))
    else:
        monkeypatch.setattr(pc.os, "rename", lambda *a, **k: (_ for _ in ()).throw(OSError("rename failed")))
    with pytest.raises(OSError):
        save_pond_checkpoint(tmp_path / "ck", state, event, IDENTITY)
    monkeypatch.undo()
    assert sorted(p.name for p in tmp_path.iterdir()) == listing, "a staging directory or file was left behind"
    assert (bystander / "keep.txt").read_text() == "keep"
    assert store_pins(case.store_dir) == pins and fingerprint(state) == before
    assert save_pond_checkpoint(tmp_path / "ck", state, event, IDENTITY).is_dir()  # and the same save then succeeds
