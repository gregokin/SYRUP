"""CW-B25: the consumer MAPLE source and the reimport tool are re-captured AFTER the genuine acceptance restore.

Fault injection is applied only to the late provenance capture (`initial_reimport.capture_maple_provenance`) or the
tool digest, and only once the real `restore_initial_assembly` has returned successfully. No physical validator,
loader or restore check is replaced. The restore's own source check (in `windowed_case`) is not affected."""
from __future__ import annotations

import json

import pytest

pytest.importorskip("maple")

import maple_syrup.initial_reimport as ir
from maple_syrup.sediment_bed import BedIntegrationError
from maple_syrup.windowed_case import INITIAL_BINDING_NAME, INITIAL_REIMPORT_FAILED_NAME

from . import test_initial_reimport as base

# The genuine producer/consumer fixtures and helpers of the CW-B21 tests (re-exported so pytest collects them here).
verified, producer, old_store, old_tree = base.verified, base.producer, base.old_store, base.old_tree
reimport, restore, reviewed = base.reimport, base.restore, base.reviewed


def _after_genuine_acceptance(monkeypatch):
    """A flag set only when the REAL acceptance restore has returned."""
    state = {"accepted": False}
    real = ir.restore_initial_assembly

    def genuine(*args, **kwargs):
        out = real(*args, **kwargs)
        state["accepted"] = True
        return out

    monkeypatch.setattr(ir, "restore_initial_assembly", genuine)
    return state


def _assert_refused_and_clean(verified, metadata):
    assert not (metadata / INITIAL_BINDING_NAME).exists()
    assert json.loads((metadata / INITIAL_REIMPORT_FAILED_NAME).read_text())["status"] == "failed"
    with pytest.raises(BedIntegrationError):
        restore(verified, metadata)


def test_consumer_changed_after_acceptance_restore_is_refused(verified, old_tree, producer, tmp_path, monkeypatch):
    transition = reviewed(old_tree, producer, tmp_path / "t.json")
    state = _after_genuine_acceptance(monkeypatch)
    real_capture = ir.capture_maple_provenance
    late = []

    def capture(*args, **kwargs):
        record = real_capture(*args, **kwargs)
        if state["accepted"]:  # the consumer package changed after the acceptance restore captured it
            late.append(True)
            digest = record["package_source_digest"]
            rel, sha = digest["files"][0]
            digest["files"][0] = (rel, "0" * 64 if sha != "0" * 64 else "1" * 64)
        return record

    monkeypatch.setattr(ir, "capture_maple_provenance", capture)
    metadata = tmp_path / "meta"
    with pytest.raises(BedIntegrationError, match="imported MAPLE source changed before the reimport completed"):
        reimport(verified, old_tree, metadata, transition, producer)
    assert state["accepted"] and late  # the genuine restore passed; only the late capture differed
    monkeypatch.undo()
    _assert_refused_and_clean(verified, metadata)


def test_consumer_pyproject_changed_after_acceptance_restore_is_refused(verified, old_tree, producer, tmp_path,
                                                                       monkeypatch):
    transition = reviewed(old_tree, producer, tmp_path / "t.json")
    state = _after_genuine_acceptance(monkeypatch)
    real_capture = ir.capture_maple_provenance

    def capture(*args, **kwargs):
        record = real_capture(*args, **kwargs)
        if state["accepted"]:
            record["metadata_file_sha256"] = {name: "0" * 64 for name in record["metadata_file_sha256"]}
        return record

    monkeypatch.setattr(ir, "capture_maple_provenance", capture)
    metadata = tmp_path / "meta"
    with pytest.raises(BedIntegrationError, match="imported MAPLE source changed before the reimport completed"):
        reimport(verified, old_tree, metadata, transition, producer)
    assert state["accepted"]
    monkeypatch.undo()
    _assert_refused_and_clean(verified, metadata)


def test_tool_changed_after_acceptance_restore_is_refused(verified, old_tree, producer, tmp_path, monkeypatch):
    transition = reviewed(old_tree, producer, tmp_path / "t.json")
    state = _after_genuine_acceptance(monkeypatch)
    real_digest = ir.tool_digest

    def digest():
        out = real_digest()
        if state["accepted"]:  # the tool file changed during the final copy/load
            out["initial_reimport.py"] = "0" * 64
        return out

    monkeypatch.setattr(ir, "tool_digest", digest)
    metadata = tmp_path / "meta"
    with pytest.raises(BedIntegrationError, match="reimport tool changed before the reimport completed"):
        reimport(verified, old_tree, metadata, transition, producer)
    assert state["accepted"]
    monkeypatch.undo()
    _assert_refused_and_clean(verified, metadata)
