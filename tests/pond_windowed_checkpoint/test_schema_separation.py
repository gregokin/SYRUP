"""Dense `/1` and windowed `/2` checkpoints never decode each other and no storage or encoding is converted; the dense
format, files and key sets are exactly what they were (not run by its author)."""

from __future__ import annotations

import json
import shutil

import numpy as np
import pytest

from maple_syrup import pond_checkpoint as pc
from maple_syrup.checkpoint import sha256_file
from maple_syrup.pond_checkpoint import PondCheckpointError, load_pond_checkpoint, save_pond_checkpoint

from .wfixture import IDENTITY, advance, assert_tree_equal, make_case, published, tree_hash


@pytest.fixture
def pair(tmp_path):
    (tmp_path / "d").mkdir()
    (tmp_path / "w").mkdir()
    dense = make_case(tmp_path / "d", storage="dense", pickup_kg_per_class=5e-4)
    wind = make_case(tmp_path / "w", pickup_kg_per_class=5e-4)
    de, we = dense.event(), wind.event()
    ds = advance(de, de.initial_state(dense.bed, dense.soil), 4)
    ws = published(advance(we, we.initial_state(wind.bed, wind.soil), 4), we)
    dpath = save_pond_checkpoint(tmp_path / "dense_ck", ds, de, IDENTITY)
    wpath = save_pond_checkpoint(tmp_path / "windowed_ck", ws, we, IDENTITY)
    return dense, ds, dpath, wind, ws, wpath


def test_the_dense_checkpoint_is_the_unchanged_v1_format(pair):
    dense, ds, dpath, _wind, _ws, _wpath = pair
    manifest = json.loads((dpath / pc.MANIFEST_NAME).read_text())
    assert manifest["schema"] == pc.SCHEMA == "maple-syrup-pond-checkpoint/1"
    assert set(manifest) == pc.MANIFEST_KEYS and "bed_storage" not in manifest and "deep_storage" not in manifest
    assert "bed_storage" not in manifest["binding"] and "deep_storage_option" not in manifest["binding"]
    with np.load(dpath / pc.PAYLOADS[0]) as native:  # the dense native snapshot keeps the full voxel column
        assert any(n.endswith("mass_kg") or "voxel" in n for n in native.files) and "window_kg" not in native.files
    loaded = load_pond_checkpoint(dpath, dense.event(), IDENTITY)
    assert_tree_equal(ds, loaded)
    assert load_pond_checkpoint(dpath, dense.event(), IDENTITY, expected_bed_storage="dense").t_s == ds.t_s


def test_each_loader_refuses_the_other_storage(pair):
    dense, _ds, dpath, wind, _ws, wpath = pair
    before_d, before_w = tree_hash(dpath), tree_hash(wpath)
    with pytest.raises(PondCheckpointError, match="windowed bed was expected"):
        load_pond_checkpoint(dpath, dense.event(), IDENTITY, expected_bed_storage="windowed")
    with pytest.raises(PondCheckpointError, match="dense bed was expected"):
        load_pond_checkpoint(wpath, wind.event(), IDENTITY, expected_bed_storage="dense")
    with pytest.raises(PondCheckpointError, match="expected_bed_storage"):
        load_pond_checkpoint(wpath, wind.event(), IDENTITY, expected_bed_storage="csr")
    assert tree_hash(dpath) == before_d and tree_hash(wpath) == before_w


def test_a_windowed_payload_is_never_decoded_as_dense(pair, tmp_path):
    _dense, _ds, _dpath, wind, _ws, wpath = pair
    # relabel the windowed manifest as the dense schema and drop the windowed keys: the dense path must refuse it before
    # or at the native dense snapshot loader, never reconstruct a dense bed
    target = tmp_path / "relabelled"
    shutil.copytree(wpath, target)
    meta = json.loads((target / pc.MANIFEST_NAME).read_text())
    meta["schema"] = pc.SCHEMA
    for key in ("bed_storage", "deep_storage"):
        meta.pop(key)
    meta["binding"]["schema"] = pc.SCHEMA
    meta["files"] = {n: sha256_file(target / n) for n in pc.PAYLOADS}
    (target / pc.MANIFEST_NAME).write_text(json.dumps(pc._seal_manifest(meta), indent=1, sort_keys=True))
    before = tree_hash(target)
    with pytest.raises(PondCheckpointError):
        load_pond_checkpoint(target, wind.event(), IDENTITY)
    assert tree_hash(target) == before


def test_a_dense_payload_is_never_decoded_as_windowed(pair, tmp_path):
    dense, _ds, dpath, _wind, _ws, _wpath = pair
    target = tmp_path / "relabelled"
    shutil.copytree(dpath, target)
    meta = json.loads((target / pc.MANIFEST_NAME).read_text())
    meta["schema"] = pc.WINDOWED_SCHEMA
    meta["binding"]["schema"] = pc.WINDOWED_SCHEMA
    meta["bed_storage"] = "windowed"
    meta["deep_storage"] = {}
    meta["files"] = {n: sha256_file(target / n) for n in pc.PAYLOADS}
    (target / pc.MANIFEST_NAME).write_text(json.dumps(pc._seal_manifest(meta), indent=1, sort_keys=True))
    with pytest.raises(PondCheckpointError):
        load_pond_checkpoint(target, dense.event(), IDENTITY)


def test_a_dense_bed_cannot_be_validated_as_windowed_nor_the_reverse(pair):
    dense, ds, _dpath, wind, ws, _wpath = pair
    with pytest.raises(PondCheckpointError, match="bed storage"):
        pc._validate_state(dense.event(), ds, None, "windowed")
    with pytest.raises(PondCheckpointError, match="bed storage"):
        pc._validate_state(wind.event(), ws, None, "dense")


def test_checkpoint_of_another_module_is_still_refused(pair, tmp_path):
    _dense, _ds, _dpath, wind, _ws, wpath = pair
    target = tmp_path / "other"
    shutil.copytree(wpath, target)
    meta = json.loads((target / pc.MANIFEST_NAME).read_text())
    meta["schema"] = "maple-syrup-checkpoint/2"
    (target / pc.MANIFEST_NAME).write_text(json.dumps(meta))
    with pytest.raises(PondCheckpointError, match="accepted checkpoint module"):
        load_pond_checkpoint(target, wind.event(), IDENTITY)
