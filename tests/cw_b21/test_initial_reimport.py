"""CW-B21: INITIAL-only reimport under a reviewed MAPLE source transition, on genuine compiled synthetic Chastre tiles.

The PRODUCER is a genuine MAPLE: a private copy of the imported (consumer) MAPLE package with an exact fixture source
change (one modified module comment, one producer-only file), identified through its own editable-distribution shim
and executed in a subprocess that assembles and persists the original initial assembly. The CONSUMER is the MAPLE
imported by this test process. Nothing is faked: actual MAPLE tiles, loader, validators and BedContext are used."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("maple")

import maple_syrup.initial_reimport as ir
from maple_syrup.chastre_case import hash_tree
from maple_syrup.sediment_bed import BedIntegrationError
from maple_syrup.windowed_case import (
    INITIAL_BINDING_NAME,
    INITIAL_FILES,
    INITIAL_REIMPORT_FAILED_NAME,
    assemble_windowed_case,
    chastre_row_tiles,
    initial_binding_identity,
    initial_identity_record,
    restore_initial_assembly,
)

TREE = Path(__file__).resolve().parents[2]
HOST = 2_000_000_000
REVIEW = {"reviewer": "fixture root reviewer", "reviewed_at": "2026-10-08T00:00:00Z",
          "evidence": "CW-B21 test fixture: exact producer/consumer source transition"}
PRODUCER_ONLY = "_cw_b21_fixture_producer_only.py"
MODIFIED = "surface/voxels/windowed_snapshot.py"

_ASSEMBLE = """
import json, sys
import maple
from maple_syrup.chastre_case import verify_chastre_case
from maple_syrup.windowed_case import (assemble_windowed_case, chastre_row_tiles, initial_identity_record,
                                       persist_initial_assembly)
v = verify_chastre_case(sys.argv[1], allow_maple_source_change=True)
grid = v.binding['grid']
a = assemble_windowed_case(chastre_row_tiles(v), v.case.config.geometry, v.case.config.grain_classes, grid['nz'],
                           store_dir=sys.argv[2], datum_offset_m=grid['datum_offset_m'], max_store_bytes=100_000_000,
                           max_host_bytes=2_000_000_000)
print(json.dumps({'maple_file': maple.__file__, 'identity': initial_identity_record(persist_initial_assembly(a))}))
"""


# --- fixtures --------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def verified(chastre_case_dir):
    from maple_syrup.chastre_case import verify_chastre_case

    return verify_chastre_case(chastre_case_dir, allow_maple_source_change=True)


@pytest.fixture(scope="module")
def producer(tmp_path_factory):
    """A genuine producer MAPLE differing from the consumer by an exact fixture change."""
    import importlib.metadata

    from maple_syrup.dependency import resolve_maple_dependency

    consumer = resolve_maple_dependency()
    base = tmp_path_factory.mktemp("cw_b21_producer").resolve()
    source = base / "source"
    shutil.copytree(consumer.package_dir, source / "src" / "maple",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"))
    shutil.copy2(consumer.source_root / "pyproject.toml", source / "pyproject.toml")
    package = source / "src" / "maple"
    with (package / MODIFIED).open("a") as handle:
        handle.write("\n# CW-B21 fixture producer: comment-only change\n")
    (package / PRODUCER_ONLY).write_text('"""CW-B21 fixture: present only in the producer."""\n')
    dist = Path(importlib.metadata.distribution("maple")._path)
    shim = base / "editable" / dist.name
    shim.mkdir(parents=True)
    for name in ("METADATA", "top_level.txt"):
        shutil.copy2(dist / name, shim / name)
    (shim / "direct_url.json").write_text(json.dumps({"dir_info": {"editable": True}, "url": f"file://{source}"}))
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "MAPLE_SYRUP_EXPECTED_MAPLE_ROOT")}
    env.update(PYTHONPATH=os.pathsep.join((str(base / "editable"), str(source / "src"), str(TREE / "src"))),
               PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="")
    return {"package_dir": package, "env": env}


@pytest.fixture(scope="module")
def old_store(producer, chastre_case_dir, tmp_path_factory):
    """The ORIGINAL initial assembly, assembled and persisted by the producer MAPLE in a subprocess."""
    store = tmp_path_factory.mktemp("cw_b21_old") / "store"
    done = subprocess.run([sys.executable, "-c", _ASSEMBLE, str(chastre_case_dir), str(store)], env=producer["env"],
                          capture_output=True, text=True, timeout=1800, check=False)
    assert done.returncode == 0, done.stderr[-4000:]
    out = json.loads(done.stdout.strip().splitlines()[-1])
    assert Path(out["maple_file"]).resolve().parent == producer["package_dir"]
    receipt = json.loads((store / "assembly_receipt.json").read_text())
    assert receipt["source_maple"]["package_source_digest"]["root"] == str(producer["package_dir"])
    return store, out["identity"]


@pytest.fixture()
def old_tree(old_store):
    """Every file of the original store (operands, receipt, binding, deep store) is unchanged by each test."""
    store, _ = old_store
    before = hash_tree(store), {p.name: stat.S_IMODE(p.stat().st_mode) for p in store.iterdir() if p.is_file()}
    yield store
    assert (hash_tree(store), {p.name: stat.S_IMODE(p.stat().st_mode) for p in store.iterdir() if p.is_file()}) == before


def tiles_of(verified):
    return chastre_row_tiles(verified)


def datum(verified):
    return float(verified.binding["grid"]["datum_offset_m"])


def reviewed(store, producer, path, *, edit=None):
    artifact = ir.propose_source_transition(store, producer["package_dir"], path.with_suffix(".proposal.json"))
    artifact = {**artifact, "review": dict(REVIEW)}
    if edit is not None:
        edit(artifact)
    path.write_text(json.dumps(artifact, sort_keys=True, indent=1) + "\n")
    return path


def reimport(verified, store, metadata, transition, producer, **kw):
    options = {"store_dir": store, "metadata_dir": metadata, "transition_path": transition,
               "producer_package_dir": producer["package_dir"], "datum_offset_m": datum(verified),
               "max_host_bytes": HOST}
    options.update(kw)
    return ir.reimport_initial_assembly(tiles_of(verified), verified.case.config.geometry,
                                        verified.case.config.grain_classes, **options)


def restore(verified, store):
    return restore_initial_assembly(tiles_of(verified), verified.case.config.geometry,
                                    verified.case.config.grain_classes, store_dir=store, max_host_bytes=HOST,
                                    datum_offset_m=datum(verified))


def resolution_of(verified):
    from maple.case_tools.compilers.case_compiler import load_compiled_case

    return load_compiled_case(tiles_of(verified)[0].case_dir).config.numerics.mass_resolution_kg


def writable(path):
    path.chmod(path.stat().st_mode | stat.S_IWUSR)


def rehash(store, name):
    path = store / INITIAL_BINDING_NAME
    binding = json.loads(path.read_text())
    binding["files"][name] = hashlib.sha256((store / name).read_bytes()).hexdigest()
    writable(path)
    path.write_text(json.dumps(binding, sort_keys=True) + "\n")


def private_copy(old, tmp_path):
    """A writable copy of the metadata files; its snapshot still references the ORIGINAL deep version."""
    copy = tmp_path / "copy"
    copy.mkdir()
    for name in (*INITIAL_FILES, INITIAL_BINDING_NAME):
        shutil.copyfile(old / name, copy / name)
    shutil.copyfile(old / "store.json", copy / "store.json")
    return copy


def leaves(state):
    from maple.core.backend import array_leaves

    return [(a.dtype, a.shape, np.ascontiguousarray(a).tobytes()) for a in array_leaves(state)]


# --- the accepted path -----------------------------------------------------------------------------------------------
def test_original_strict_restore_refuses_the_changed_source(verified, old_tree):
    with pytest.raises(BedIntegrationError, match="MAPLE source changed"):
        restore(verified, old_tree)


def test_reimport_binds_the_consumer_and_the_ordinary_strict_restore_accepts_it(verified, old_tree, old_store,
                                                                              producer, tmp_path):
    from maple_syrup.dependency import resolve_maple_dependency
    from maple_syrup.provenance import capture_maple_provenance

    transition = reviewed(old_tree, producer, tmp_path / "transition.json")
    artifact = json.loads(transition.read_text())
    assert artifact["changes"] == {
        "added": {}, "metadata": {},
        "removed": {PRODUCER_ONLY: hashlib.sha256((producer["package_dir"] / PRODUCER_ONLY).read_bytes()).hexdigest()},
        "modified": {MODIFIED: {"producer": hashlib.sha256((producer["package_dir"] / MODIFIED).read_bytes()).hexdigest(),
                                "consumer": artifact["consumer"]["files"][MODIFIED]}}}
    metadata = tmp_path / "meta"
    out = reimport(verified, old_tree, metadata, transition, producer)

    old_receipt = json.loads((old_tree / "assembly_receipt.json").read_text())
    new_receipt = json.loads((metadata / "assembly_receipt.json").read_text())
    current = capture_maple_provenance(resolve_maple_dependency())
    assert new_receipt["source_maple"]["package_source_digest"] == current["package_source_digest"]
    assert new_receipt["source_maple"]["dependency"] == current["dependency"]
    record = new_receipt["initial_reimport"]
    assert record["producer"] == old_receipt["source_maple"]  # the producer is NOT relabelled
    assert record["consumer"] == new_receipt["source_maple"]
    assert record["transition"]["sha256"] == hashlib.sha256(transition.read_bytes()).hexdigest() == out["transition_sha256"]
    assert record["original_initial_binding"] == old_store[1] == out["original_initial_binding"]
    assert {k: v for k, v in new_receipt.items() if k not in ("source_maple", "initial_reimport")} == \
        {k: v for k, v in old_receipt.items() if k != "source_maple"}
    for name in ("initial_resident.npz", "initial_baseline.npz"):
        assert (metadata / name).read_bytes() == (old_tree / name).read_bytes()
    for name in (*INITIAL_FILES, INITIAL_BINDING_NAME):
        assert stat.S_IMODE((metadata / name).stat().st_mode) == 0o444
    assert sorted(p.name for p in metadata.iterdir()) == sorted((*INITIAL_FILES, INITIAL_BINDING_NAME))
    assert out["initial_binding"] != old_store[1]  # a new initial identity: original-run wet checkpoints stay strict

    again = restore(verified, metadata)  # the ordinary strict restore, under the consumer
    assert again.initial_binding == out["initial_binding"]
    assert again.state.voxel_column.deep_store.store_dir.resolve() == old_tree.resolve()  # external, not copied
    assert again.state.voxel_column.generation == 0
    fresh = assemble_windowed_case(tiles_of(verified), verified.case.config.geometry,
                                   verified.case.config.grain_classes, verified.binding["grid"]["nz"],
                                   store_dir=tmp_path / "consumer_fresh", datum_offset_m=datum(verified),
                                   max_store_bytes=100_000_000, max_host_bytes=HOST)
    assert leaves(again.state) == leaves(fresh.state)  # every physical byte equals a fresh consumer assembly
    assert again.state.voxel_column.manifest_digest == fresh.state.voxel_column.manifest_digest
    assert np.asarray(again.context.initial_available_fraction).tobytes() == \
        np.asarray(fresh.context.initial_available_fraction).tobytes()
    assert again.state.voxel_column.height_packet.packet.tobytes() == \
        fresh.state.voxel_column.height_packet.packet.tobytes()


def test_windowed_setup_reuses_the_new_metadata_and_refuses_the_original(verified, chastre_case_dir, old_tree,
                                                                        producer, tmp_path):
    import maple_syrup.windowed_experiment as we
    from maple_syrup.sediment_event import SedimentEventControl

    metadata = tmp_path / "meta"
    out = reimport(verified, old_tree, metadata, reviewed(old_tree, producer, tmp_path / "t.json"), producer)
    control = SedimentEventControl(phase_storage="csr", phase_max_nnz=100_000).validated()
    options = {"case_kind": "chastre", "window_voxels": 4, "max_store_bytes": 100_000_000, "max_host_bytes": HOST,
               "max_journal_records": 10_000, "backend": "numpy", "control": control, "expected_maple_root": None,
               "allow_maple_source_change": True, "mahleran_root": None, "end_s": None, "restore_initial": True}
    setup = we._setup_windowed(chastre_case_dir, store_dir=metadata, **options)
    assert setup["run_configuration"]["initial_assembly"] == out["initial_binding"] != out["original_initial_binding"]
    with pytest.raises(BedIntegrationError, match="MAPLE source changed"):
        we._setup_windowed(chastre_case_dir, store_dir=old_tree, **options)


def test_cli_propose_then_reviewed_import(verified, chastre_case_dir, old_tree, producer, tmp_path, capsys):
    proposal = tmp_path / "proposal.json"
    assert ir.main(["propose", "--store-dir", str(old_tree), "--producer-package-dir", str(producer["package_dir"]),
                    "--out", str(proposal)]) == 0
    unreviewed = json.loads(proposal.read_text())
    assert unreviewed["review"] is None
    args = ["import", "--store-dir", str(old_tree), "--producer-package-dir", str(producer["package_dir"]),
            "--case-dir", str(chastre_case_dir), "--max-host-bytes", str(HOST)]
    with pytest.raises(BedIntegrationError, match="not reviewed"):
        ir.main([*args, "--metadata-dir", str(tmp_path / "refused"), "--transition", str(proposal)])
    assert not (tmp_path / "refused").exists()
    final = tmp_path / "final.json"
    final.write_text(json.dumps({**unreviewed, "review": REVIEW}))
    capsys.readouterr()
    assert ir.main([*args, "--metadata-dir", str(tmp_path / "meta"), "--transition", str(final)]) == 0
    printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert printed["initial_binding"] == restore(verified, tmp_path / "meta").initial_binding


# --- refusals before anything is written ----------------------------------------------------------------------------
def _drop_removed(a):
    a["changes"]["removed"] = {}


def _extra_modified(a):
    a["changes"]["modified"]["core/backend/__init__.py"] = {"producer": "0" * 64, "consumer": "1" * 64}


def _consumer_file(a):
    a["consumer"]["files"][MODIFIED] = "0" * 64


def _producer_file(a):
    a["producer"]["files"][MODIFIED] = "0" * 64


def _tool(a):
    a["tool_sha256"]["initial_reimport.py"] = "0" * 64


def _subject(a):
    a["subject"]["original_initial_binding_sha256"] = "0" * 64


def _unreviewed(a):
    a["review"] = None


def _blank_review(a):
    a["review"]["reviewer"] = " "


def _extra_key(a):
    a["allow_any_source"] = True


@pytest.mark.parametrize("edit,match", [
    (_drop_removed, "missing actual change"), (_extra_modified, "not actual"),
    (_consumer_file, "consumer manifest"), (_producer_file, "producer manifest"), (_tool, "tool digest"),
    (_subject, "does not name these original"), (_unreviewed, "not reviewed"), (_blank_review, "not reviewed"),
    (_extra_key, "keys differ")])
def test_inexact_or_unreviewed_transitions_are_refused(verified, old_tree, producer, tmp_path, edit, match):
    transition = reviewed(old_tree, producer, tmp_path / "t.json", edit=edit)
    with pytest.raises(BedIntegrationError, match=match):
        reimport(verified, old_tree, tmp_path / "meta", transition, producer)
    assert not (tmp_path / "meta").exists()


def test_missing_transition_existing_or_inner_metadata_dir_are_refused(verified, old_tree, producer, tmp_path):
    transition = reviewed(old_tree, producer, tmp_path / "t.json")
    with pytest.raises(BedIntegrationError, match="artifact is missing"):
        reimport(verified, old_tree, tmp_path / "meta", tmp_path / "absent.json", producer)
    existing = tmp_path / "exists"
    existing.mkdir()
    with pytest.raises(BedIntegrationError, match="already exists"):
        reimport(verified, old_tree, existing, transition, producer)
    assert list(existing.iterdir()) == []
    with pytest.raises(BedIntegrationError, match="outside the original store"):
        reimport(verified, old_tree, old_tree / "meta", transition, producer)


def test_changed_producer_source_or_path_is_refused(verified, old_tree, producer, tmp_path):
    transition = reviewed(old_tree, producer, tmp_path / "t.json")
    target = producer["package_dir"] / PRODUCER_ONLY
    original = target.read_bytes()
    try:
        target.write_bytes(original + b"# changed after the original assembly\n")
        with pytest.raises(BedIntegrationError, match="producer MAPLE source does not match"):
            reimport(verified, old_tree, tmp_path / "meta", transition, producer)
    finally:
        target.write_bytes(original)
    other = tmp_path / "other_maple"
    shutil.copytree(producer["package_dir"], other)
    with pytest.raises(BedIntegrationError, match="producer package directory"):
        reimport(verified, old_tree, tmp_path / "meta", transition, {"package_dir": other})
    assert not (tmp_path / "meta").exists()


def test_changed_tiles_datum_and_geometry_are_refused(verified, old_tree, producer, tmp_path):
    transition = reviewed(old_tree, producer, tmp_path / "t.json")
    geometry, classes = verified.case.config.geometry, verified.case.config.grain_classes
    common = {"store_dir": old_tree, "transition_path": transition, "producer_package_dir": producer["package_dir"],
              "max_host_bytes": HOST}
    tiles = list(tiles_of(verified))
    changed = dict(tiles[1].files)
    changed[next(iter(changed))] = "0" * 64
    tiles[1] = dataclasses.replace(tiles[1], files=changed)
    with pytest.raises(BedIntegrationError, match="source tile changed"):
        ir.reimport_initial_assembly(tiles, geometry, classes, metadata_dir=tmp_path / "m1",
                                     datum_offset_m=datum(verified), **common)
    with pytest.raises(BedIntegrationError, match="datum"):
        ir.reimport_initial_assembly(tiles_of(verified), geometry, classes, metadata_dir=tmp_path / "m2",
                                     datum_offset_m=datum(verified) + 1.0, **common)
    with pytest.raises((BedIntegrationError, ValueError)):
        ir.reimport_initial_assembly(tiles_of(verified), dataclasses.replace(geometry, dx_m=geometry.dx_m * 2), classes,
                                     metadata_dir=tmp_path / "m3", datum_offset_m=datum(verified), **common)
    assert not any((tmp_path / m).exists() for m in ("m1", "m2", "m3"))


def test_tampered_tile_file_is_refused(verified, old_tree, producer, tmp_path):
    transition = reviewed(old_tree, producer, tmp_path / "t.json")
    tile = tiles_of(verified)[-1]
    path = tile.case_dir / "case.yaml"
    original = path.read_bytes()
    try:
        path.write_bytes(original + b"\n# tampered\n")
        with pytest.raises(BedIntegrationError, match="source tile changed"):
            reimport(verified, old_tree, tmp_path / "meta", transition, producer)
    finally:
        path.write_bytes(original)
    assert not (tmp_path / "meta").exists()


def test_changed_configuration_in_a_coherently_rehashed_copy_is_refused(verified, old_tree, producer, tmp_path):
    copy = private_copy(old_tree, tmp_path)
    receipt_path = copy / "assembly_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["maple_configuration"]["seed"] = receipt["maple_configuration"]["seed"] + 1
    receipt_path.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    rehash(copy, "assembly_receipt.json")
    transition = reviewed(copy, producer, tmp_path / "t.json")  # an exact, reviewed transition for THIS copy
    with pytest.raises(BedIntegrationError, match="configuration changed"):
        reimport(verified, copy, tmp_path / "meta", transition, producer)
    assert not (tmp_path / "meta").exists()


# --- non-initial states: rewritten through the GENUINE current MAPLE snapshot writer, coherently re-hashed ---------
def _rewrite_snapshot(verified, store, *, step=0, time_s=0.0, depth=0.0, mobile=0.0, column=None):
    from maple.core.types.water import WaterState
    from maple.surface.voxels.windowed_snapshot import (
        load_windowed_state_snapshot,
        save_windowed_state_snapshot,
    )

    g = verified.case.config.geometry
    resolution = resolution_of(verified)
    path = store / "initial_resident.npz"
    snap = load_windowed_state_snapshot(path, geometry=g, grain_classes=verified.case.config.grain_classes,
                                        mass_resolution_kg=resolution, max_bytes=HOST)
    d = snap["water"].depth_m.copy()
    m = snap["water"].mobile_mass_by_cell_class_kg.copy()
    cell = tuple(np.argwhere(np.asarray(verified.fields["active"], dtype=bool))[0])
    d[cell] += depth
    m[cell][0] += mobile
    path.unlink()
    save_windowed_state_snapshot(snap["voxel_column"] if column is None else column(snap), snap["active_layer"], time_s,
                                 path, step=step, geometry=g, mass_resolution_kg=resolution,
                                 sediment_availability=snap["sediment_availability"], water=WaterState(d, m),
                                 provenance={"assembly_receipt": "assembly_receipt.json"})
    rehash(store, "initial_resident.npz")


def _evolved_column(verified, store):
    """A GENUINE published version 1 (one same-value deep record via `apply_journal`) in a private store copy."""
    from maple.core.types.voxel import WindowedVoxelColumnState
    from maple.surface.voxels.deep_store import DeepJournal, load_version
    from maple.surface.voxels.windowed_prefix import attach_prefix_packets

    g = verified.case.config.geometry
    resolution = resolution_of(verified)

    def build(snap):
        old = snap["voxel_column"]
        v0 = load_version(store, old.manifest_digest, geometry=g, mass_resolution_kg=resolution)
        value = np.asarray(v0.gather_columns(np.array([0], dtype=np.int64)))[0, 0][None, :].astype(np.float64)
        v1 = v0.apply_journal(DeepJournal.from_records(np.array([0]), np.array([0]), value, header=v0.header),
                              g, resolution)
        assert v1.version_id == 1 and v1.parent_manifest_digest == old.manifest_digest
        column = WindowedVoxelColumnState(window_kg=old.window_kg, base=old.base, deep_by_class_kg=old.deep_by_class_kg,
                                          deep_total_kg=old.deep_total_kg, deep_room_kg=old.deep_room_kg,
                                          n_voxels=old.n_voxels, deep_store=v1, generation=1,
                                          manifest_digest=v1.manifest_digest)
        return attach_prefix_packets(column, np.array(old.height_packet.packet))

    return build


@pytest.mark.parametrize("kind,match", [("wet", "wet water"), ("mobile", "mobile sediment"),
                                        ("step", "time 0, step 0"), ("time", "time 0, step 0"),
                                        ("generation", "generation-zero")])
def test_non_initial_snapshots_are_refused(verified, old_tree, producer, tmp_path, kind, match):
    if kind == "generation":
        store = tmp_path / "store_copy"
        shutil.copytree(old_tree, store)
        for p in store.iterdir():
            if p.is_file():
                writable(p)
        options = {"column": _evolved_column(verified, store)}
    else:
        store = private_copy(old_tree, tmp_path)
        options = {"wet": {"depth": 0.01}, "mobile": {"depth": 0.01, "mobile": 1e-3}, "step": {"step": 1},
                   "time": {"time_s": 5.0}}[kind]
    _rewrite_snapshot(verified, store, **options)
    transition = reviewed(store, producer, tmp_path / "t.json")  # exact and reviewed: only the state is wrong
    with pytest.raises(BedIntegrationError, match=match):
        reimport(verified, store, tmp_path / "meta", transition, producer)
    assert not (tmp_path / "meta").exists()


# --- failure after files were written; inconsistent reimport record -------------------------------------------------
def test_late_failure_leaves_no_executable_binding(verified, old_tree, producer, tmp_path, monkeypatch):
    transition = reviewed(old_tree, producer, tmp_path / "t.json")

    def refuse(*args, **kwargs):
        raise BedIntegrationError("injected late failure")

    monkeypatch.setattr(ir, "restore_initial_assembly", refuse)
    metadata = tmp_path / "meta"
    with pytest.raises(BedIntegrationError, match="injected late failure"):
        reimport(verified, old_tree, metadata, transition, producer)
    assert not (metadata / INITIAL_BINDING_NAME).exists()
    assert json.loads((metadata / INITIAL_REIMPORT_FAILED_NAME).read_text())["status"] == "failed"
    monkeypatch.undo()
    with pytest.raises(BedIntegrationError, match="missing"):
        restore(verified, metadata)


def test_restore_refuses_a_failure_record_or_inconsistent_reimport_record(verified, old_tree, producer, tmp_path):
    metadata = tmp_path / "meta"
    reimport(verified, old_tree, metadata, reviewed(old_tree, producer, tmp_path / "t.json"), producer)
    copy = tmp_path / "copy"
    shutil.copytree(metadata, copy)
    (copy / INITIAL_REIMPORT_FAILED_NAME).write_text("{}")
    with pytest.raises(BedIntegrationError, match="failure record"):
        restore(verified, copy)
    (copy / INITIAL_REIMPORT_FAILED_NAME).unlink()
    receipt_path = copy / "assembly_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["initial_reimport"]["copied_operands"]["initial_baseline.npz"] = "0" * 64
    writable(receipt_path)
    receipt_path.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    rehash(copy, "assembly_receipt.json")
    with pytest.raises(BedIntegrationError, match="initial reimport record is inconsistent"):
        restore(verified, copy)
    assert restore(verified, metadata).initial_binding == initial_identity_record(initial_binding_identity(metadata))
