"""Portability of the REQUIRED regression fixtures: the minimal trigger controls are packaged with the suite (exact payload, attribution, digests), a missing or malformed required fixture fails explicitly instead
of silently parameterising nothing, and the packaged payload equals Root's immutable original whenever that task file is present. Author-written."""
from __future__ import annotations

import hashlib
import json

import pytest
from conftest import MINIMAL_TRIGGERS, REQUIRED_LABELS, ROOM_CONTROLS, ROOT_ORIGINAL, FixtureError, load_minimal_triggers, load_room_controls

canon = lambda x: json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=True)  # noqa: E731


def test_the_packaged_minimal_trigger_fixture_is_present_complete_and_attributed():
    d = json.loads(MINIMAL_TRIGGERS.read_text())
    meta = d["fixture"]
    assert meta["name"] == "minimal_trigger_controls" and meta["source_file_sha256"] == "5516faae5475f5d8e8c79abff3ef92cb480a174a896940b1e1cb10f1f838e584"
    assert "Root" in meta["attribution"] and "no third-party model code" in meta["attribution"] and "immutable original" in meta["source_file"]
    assert tuple(meta["required_labels"]) == REQUIRED_LABELS
    controls = load_minimal_triggers()
    assert tuple(c["label"] for c in controls) == REQUIRED_LABELS and len(controls) == 2
    assert hashlib.sha256(canon(d["controls"]).encode()).hexdigest() == meta["controls_canonical_sha256"] == "6ecf0b0b110340aa5be410a6e5d5499b0eb7f4195b2921ed97c161550e67b58a"  # the payload is exactly as copied
    by = {c["label"]: c for c in controls}
    a, b = by["L3_N128_all_shared"], by["L5_N257_scalar_thetas"]
    assert (a["cell"], a["pool_m"], a["old_raw"]["intake_m"], a["old_raw"]["pool_out_m"]) == (103, 0.04838095732241791, 0.048380957322417914, -6.938893903907228e-18)
    assert (b["cell"], b["pool_m"], b["old_raw"]["intake_m"], b["old_raw"]["pool_out_m"]) == (248, 0.029264621197886812, 0.029264621197886816, -3.469446951953614e-18)
    assert all(c["expected_old_error"] == {"bit": 17, "n_flagged": 1} for c in controls)


def test_the_packaged_room_fixture_is_present_and_nonempty():
    assert ROOM_CONTROLS.is_file() and len(load_room_controls()) == 3


@pytest.mark.skipif(not ROOT_ORIGINAL.exists(), reason="Root's immutable original task file is not distributed with this checkout (the packaged fixture is then the only source)")
def test_the_packaged_payload_equals_roots_immutable_original():
    raw = ROOT_ORIGINAL.read_bytes()
    orig = json.loads(raw)
    d = json.loads(MINIMAL_TRIGGERS.read_text())
    assert hashlib.sha256(raw).hexdigest() == d["fixture"]["source_file_sha256"]  # the original is the one the fixture was copied from, unedited
    assert d["controls"] == orig["controls"] and d["source_snapshot"] == orig["source_snapshot"]  # exact numeric inputs, names and attribution fields


def test_a_missing_required_fixture_fails_explicitly(tmp_path):
    with pytest.raises(FixtureError, match="missing"):
        load_minimal_triggers(tmp_path / "absent.json")
    with pytest.raises(FixtureError, match="missing"):
        load_room_controls(tmp_path / "absent.json")


def test_a_malformed_or_incomplete_required_fixture_fails_explicitly(tmp_path):
    good = json.loads(MINIMAL_TRIGGERS.read_text())
    cases = {
        "not_json.json": "{ not json",
        "no_controls.json": json.dumps({"fixture": {}}),
        "empty.json": json.dumps({"fixture": {}, "controls": []}),
        "one_label.json": json.dumps({**good, "controls": good["controls"][:1]}),
        "renamed.json": json.dumps({**good, "controls": [dict(good["controls"][0], label="other"), good["controls"][1]]}),
        "missing_key.json": json.dumps({**good, "controls": [{k: v for k, v in good["controls"][0].items() if k != "pool_m"}, good["controls"][1]]}),
    }
    for name, text in cases.items():
        path = tmp_path / name
        path.write_text(text)
        with pytest.raises(FixtureError):
            load_minimal_triggers(path)
    (tmp_path / "room_bad.json").write_text(json.dumps([{"kind": "room"}]))
    with pytest.raises(FixtureError):
        load_room_controls(tmp_path / "room_bad.json")
    (tmp_path / "room_empty.json").write_text("[]")
    with pytest.raises(FixtureError):
        load_room_controls(tmp_path / "room_empty.json")
