"""CW-E2 correction 1: forged-but-self-consistent manifests, path confinement, staged verification before publication.

Each regression runs against the corrected module AND, as a negative control, against the ORIGINAL E2 module
(loaded read-only from the completed E2 tree, sha256-pinned): the control must exhibit the defect the root
reproduced (cw_e2_independent_version_repro.json), proving the test can fail."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

from maple_syrup import dependency_bundle as db

ORIGINAL_TREE = Path("/home/okin/SYRUP/outputs/conservative_water/cw_e2_dependency_delivery/src/maple_syrup")
ORIGINAL_SHA256 = "7628454bc53cf54f6da32221062847b09b48803a212858a280ac57b4677ffb17"
HERE = Path(db.__file__).resolve().parent


@pytest.fixture(scope="module")
def original():
    path = ORIGINAL_TREE / "dependency_bundle.py"
    if not path.is_file():
        pytest.skip("original E2 tree absent")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == ORIGINAL_SHA256, "original E2 module changed"
    for shared in ("dependency.py", "provenance.py"):  # the control resolves these from THIS tree: same bytes
        assert (ORIGINAL_TREE / shared).read_bytes() == (HERE / shared).read_bytes()
    spec = importlib.util.spec_from_file_location("cw_e2_original_dependency_bundle", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve their module by name; never the production name
    previous, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


@pytest.fixture
def bundle(project):
    out = project["tmp"] / "genuine"
    record = db.export_bundle(project["root"], project["dist"], project["origin"], out)
    return record, out


def forge(record: dict, out: Path, dest: Path, mutate) -> Path:
    """A forged bundle: same archive, mutated manifest, bundle_id RECOMPUTED (self-consistent hash)."""
    body = {k: v for k, v in record.items() if k != "bundle_id"}
    body = json.loads(json.dumps(body))
    mutate(body)
    dest.mkdir()
    (dest / db.ARCHIVE_NAME).write_bytes((out / db.ARCHIVE_NAME).read_bytes())
    full = {"bundle_id": hashlib.sha256(db._canonical(body)).hexdigest(), **body}
    (dest / db.MANIFEST_NAME).write_bytes(db._canonical(full) + b"\n")
    return dest


def everything(root: Path) -> set[str]:
    return {str(p.relative_to(root)) for p in root.rglob("*")}


def set_version(value):
    def mutate(body):
        body["project"]["version"] = value
    return mutate


@pytest.mark.parametrize("version", ["../../../../escaped", "a/b", "/abs", "", "..", ".", "1.0\n", "1.0 x",
                                     "\x00", "1.0/../../x", "v1.0"])
def test_unsafe_version_is_refused_before_any_write(bundle, project, version):
    record, out = bundle
    forged = forge(record, out, project["tmp"] / "forged", set_version(version))
    with pytest.raises(db.BundleError):
        db.verify_bundle(forged)
    nest = project["tmp"] / "a" / "b" / "c"
    cache = nest / "cache"
    cache.mkdir(parents=True)
    before = everything(project["tmp"])
    with pytest.raises(db.BundleError):
        db.materialize_bundle(forged, cache)
    assert os.listdir(cache) == []  # no target, no staging
    assert everything(project["tmp"]) == before  # no escaped sibling or ancestor entries anywhere


def test_negative_control_original_writes_outside_and_publishes(bundle, project, original):
    record, out = bundle
    forged = forge(record, out, project["tmp"] / "forged", set_version("../../../../escaped"))
    assert original.verify_bundle(forged)["project"]["version"] == "../../../../escaped"  # accepted
    cache = project["tmp"] / "a" / "b" / "c" / "cache"
    cache.mkdir(parents=True)
    with pytest.raises(original.BundleError):
        original.materialize_bundle(forged, cache)
    escaped = [p for p in project["tmp"].rglob("escaped*")]
    assert escaped, "control did not reproduce the outside write"
    assert all(not p.is_relative_to(cache / record["bundle_id"]) for p in escaped)
    published = [p for p in cache.iterdir() if len(p.name) == 64]
    assert published, "control did not reproduce the late (post-publication) refusal"


@pytest.mark.parametrize("version", ["9.9", "0.0.2"])
def test_safe_version_contradicting_the_archived_source_is_refused(bundle, project, original, version):
    record, out = bundle
    forged = forge(record, out, project["tmp"] / "forged", set_version(version))
    with pytest.raises(db.BundleError, match="contradicts the archived pyproject"):
        db.verify_bundle(forged)
    assert original.verify_bundle(forged)  # control: accepted by the original
    cache = project["tmp"] / "cache"
    cache.mkdir()
    with pytest.raises(db.BundleError):
        db.materialize_bundle(forged, cache)
    assert os.listdir(cache) == []


def _digest_mutations():
    def project_digest(body):
        body["project"]["project_source_digest"]["digest_sha256"] = "0" * 64

    def project_count(body):
        body["project"]["project_source_digest"]["file_count"] += 1

    def project_bytes(body):
        body["project"]["project_source_digest"]["total_bytes"] -= 1

    def package_digest(body):
        body["project"]["package_source_digest"]["digest_sha256"] = "f" * 64

    def package_count(body):
        body["project"]["package_source_digest"]["file_count"] = 0

    def package_bytes(body):
        body["project"]["package_source_digest"]["total_bytes"] += 7

    def package_root(body):
        body["project"]["package_source_digest"]["root"] = "/elsewhere/maple"

    def pyproject_hash(body):
        body["project"]["pyproject_sha256"] = "1" * 64

    return {f.__name__: f for f in (project_digest, project_count, project_bytes, package_digest, package_count,
                                    package_bytes, package_root, pyproject_hash)}


@pytest.mark.parametrize("name", list(_digest_mutations()))
def test_declared_digests_counts_and_bytes_must_match_the_members(bundle, project, original, name):
    record, out = bundle
    forged = forge(record, out, project["tmp"] / "forged", _digest_mutations()[name])
    with pytest.raises(db.BundleError):
        db.verify_bundle(forged)
    if name in ("project_digest", "package_digest", "package_bytes"):
        assert original.verify_bundle(forged)  # control: the original accepted these


def _metadata_mutations():
    def origin_no_reason(body):
        body["origin"]["original_source"]["revision_absence_reason"] = None

    def origin_chain_item(body):
        body["origin"]["patch_chain"] = [{"id": "p1"}]  # description missing

    def origin_extra(body):
        body["origin"]["invented_git_head"] = "abc"

    def origin_missing(body):
        del body["origin"]["patch_chain"]

    def tool_bad(body):
        body["tool"]["module_sha256"] = "not-a-hash"

    def limits_bad(body):
        body["limits"]["max_files"] = -1

    def rule_changed(body):
        body["selection_rule"]["excluded_dir_names"] = []

    def git_invented(body):
        body["git_supplementary"] = {"status": "ok", "head_commit": "0" * 40}

    def archive_format(body):
        body["archive"]["format"] = "gzip"

    def dist_dupe(body):
        body["distribution"]["files"] = body["distribution"]["files"] + ["METADATA"]

    return {f.__name__: f for f in (origin_no_reason, origin_chain_item, origin_extra, origin_missing, tool_bad,
                                    limits_bad, rule_changed, git_invented, archive_format, dist_dupe)}


@pytest.mark.parametrize("name", list(_metadata_mutations()))
def test_metadata_structures_must_meet_the_documented_schema(bundle, project, original, name):
    record, out = bundle
    forged = forge(record, out, project["tmp"] / "forged", _metadata_mutations()[name])
    with pytest.raises(db.BundleError):
        db.verify_bundle(forged)
    if name.startswith("origin") or name in ("tool_bad", "rule_changed", "git_invented"):
        assert original.verify_bundle(forged)  # control: accepted because only the hash was recomputed


def corrupt_staged(module, monkeypatch, rel):
    """Corrupt one staged file just before the tree is frozen (late write corruption)."""
    real = module._freeze_tree

    def wrapped(root):
        path = Path(root) / rel
        if path.exists():
            path.write_bytes(path.read_bytes() + b"#corrupt")
        real(root)

    monkeypatch.setattr(module, "_freeze_tree", wrapped)


@pytest.mark.parametrize("rel", ["distribution/METADATA", "editable/maple-0.0.1.dist-info/direct_url.json",
                                 "materialization.json", "source/src/maple/__init__.py"])
def test_staged_tree_is_verified_before_publication(bundle, project, monkeypatch, rel):
    _record, out = bundle
    cache = project["tmp"] / "cache"
    cache.mkdir()
    corrupt_staged(db, monkeypatch, rel)
    with pytest.raises(db.BundleError):
        db.materialize_bundle(out, cache)
    assert os.listdir(cache) == []  # nothing published, staging removed


def test_negative_control_original_publishes_before_validating(bundle, project, monkeypatch, original):
    record, out = bundle
    cache = project["tmp"] / "cache"
    cache.mkdir()
    corrupt_staged(original, monkeypatch, "distribution/METADATA")
    with pytest.raises(original.BundleError):
        original.materialize_bundle(out, cache)
    assert (cache / record["bundle_id"]).is_dir(), "control did not reproduce the late refusal"


def short_writes(monkeypatch, at_most=7):
    real = os.write

    def write(fd, data):
        return real(fd, bytes(data[:at_most]))

    monkeypatch.setattr(os, "write", write)


def test_short_writes_are_completed(bundle, project, monkeypatch):
    record, out = bundle
    cache = project["tmp"] / "cache"
    cache.mkdir()
    with monkeypatch.context() as m:
        short_writes(m)
        target = db.materialize_bundle(out, cache)
    db.verify_materialized(target)
    for entry in record["files"]:
        if entry["path"].startswith("source/"):
            rel = entry["path"][len("source/"):]
            assert (target / "source" / rel).read_bytes() == (project["root"] / rel).read_bytes()


def test_negative_control_original_truncates_on_short_writes(bundle, project, monkeypatch, original):
    _record, out = bundle
    cache = project["tmp"] / "cache"
    cache.mkdir()
    with monkeypatch.context() as m:
        short_writes(m)
        with pytest.raises(original.BundleError):
            original.materialize_bundle(out, cache)


def test_genuine_bundle_still_verifies_and_round_trips(bundle, project):
    record, out = bundle
    assert db.verify_bundle(out)["bundle_id"] == record["bundle_id"]
    cache = project["tmp"] / "cache"
    cache.mkdir()
    target = db.materialize_bundle(out, cache)
    assert db.verify_materialized(target)["bundle_id"] == record["bundle_id"]
    assert sorted(os.listdir(cache)) == [record["bundle_id"]]
