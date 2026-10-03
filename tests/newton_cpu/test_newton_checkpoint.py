"""Restart compatibility of the Newton root-solver option: the default (bisection) control encodes exactly as before
the option existed, so older checkpoints load and new default checkpoints are unchanged; a Newton control is stored,
restored and continues equivalently; malformed solver metadata is refused."""
from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

pytest.importorskip("maple")
from test_sediment_event import chain_elevation, make_bed, setup

from maple_syrup.checkpoint import load_checkpoint, save_checkpoint
from maple_syrup.complete_event import CompletionPolicy, complete_event
from maple_syrup.rainfall import constant_rainfall
from maple_syrup.sediment_event import SedimentEventControl, SedimentEventError
from maple_syrup.storm import StormControl, StormError

IDENTITY = {"case": "controlled actual MAPLE bed", "forcing": "fixed", "source": "test", "options": {"dt": 1}}
OLD_STORM_FIELDS = {"max_dt_s", "min_dt_s", "max_retries", "max_steps", "courant_max", "bisection_iterations",
                    "root_tolerance_m", "implementation"}


def fixture(**storm):
    bed, ctx = make_bed(chain_elevation(3), depth_m=0.001)
    control = SedimentEventControl(storm=StormControl(max_dt_s=1, max_steps=10000, **storm),
                                   transport_implementation="auto")
    s = setup(bed, ctx, ksat=1e-5, control=control)
    args = (s["state0"], ctx, s["column"], s["field"], constant_rainfall(0, 10, 100), s["vegetation"], s["sediment"])
    kw = {"max_end_s": 180, "control": control, "policy": CompletionPolicy(hold_s=2), "report_every_s": 2.0}
    return args, kw


def arrays(value, path=""):
    if isinstance(value, np.ndarray):
        yield path, value
    elif dataclasses.is_dataclass(value):
        for f in dataclasses.fields(value):
            yield from arrays(getattr(value, f.name), path + "." + f.name)
    elif isinstance(value, dict):
        for key, v in value.items():
            yield from arrays(v, path + "." + key)


def storm_control_fields(path):
    manifest = json.loads((path / "checkpoint.json").read_text())

    def find(node):
        if isinstance(node, dict):
            if node.get("type") == "StormControl":
                yield node["fields"]
            for v in node.values():
                yield from find(v)
        elif isinstance(node, list):
            for v in node:
                yield from find(v)

    found = list(find(manifest["payload"]))
    assert found
    return found, manifest


def save(tmp_path, name, **storm):
    args, kw = fixture(**storm)
    paused = complete_event(*args, **kw, checkpoint_callback=lambda p: True)
    path = tmp_path / name
    save_checkpoint(path, paused, args[1], args[2], args[6], IDENTITY)
    return args, kw, paused, path


def test_default_control_encodes_exactly_the_historical_fields_and_still_loads(tmp_path):
    args, _kw, paused, path = save(tmp_path, "default")
    found, _ = storm_control_fields(path)
    assert all(set(fields) == OLD_STORM_FIELDS for fields in found)
    restored = load_checkpoint(path, args[1], args[2], args[6], IDENTITY)
    assert restored.control.storm == paused.control.storm
    assert restored.control.storm.root_solver == "bisection"


def test_newton_control_is_stored_restored_and_resumes_equivalently(tmp_path):
    storm = {"root_solver": "newton", "newton_max_iterations": 17}
    args, kw, _paused, path = save(tmp_path, "newton", **storm)
    found, _ = storm_control_fields(path)
    assert all(set(fields) == OLD_STORM_FIELDS | {"root_solver", "newton_max_iterations"} for fields in found)
    assert {f["root_solver"] for f in found} == {"newton"} and {f["newton_max_iterations"] for f in found} == {17}
    restored = load_checkpoint(path, args[1], args[2], args[6], IDENTITY)
    assert restored.control.storm.root_solver == "newton" and restored.control.storm.newton_max_iterations == 17
    assert np.any(restored.result.state.bed.water.mobile_mass_by_cell_class_kg > 0)
    full = complete_event(*args, **kw)
    resumed = complete_event(restored.result.state, *args[1:], **kw, continuation=restored)
    first, second = dict(arrays(full.dry_state)), dict(arrays(resumed.dry_state))
    assert first.keys() == second.keys()
    for name, value in first.items():
        np.testing.assert_array_equal(value, second[name], err_msg=name)
    np.testing.assert_array_equal(full.progress.result.hydrograph, resumed.progress.result.hydrograph)
    assert full.progress.result.n_accepted_steps == resumed.progress.result.n_accepted_steps


def rewrite(path, mutate):
    manifest = json.loads((path / "checkpoint.json").read_text())

    def walk(node):
        if isinstance(node, dict):
            if node.get("type") == "StormControl":
                mutate(node["fields"])
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(manifest["payload"])
    (path / "checkpoint.json").write_text(json.dumps(manifest))


@pytest.mark.parametrize("kind", ["missing_solver", "missing_cap", "unknown", "bogus_solver", "bad_cap", "huge_cap"])
def test_malformed_solver_metadata_is_refused(tmp_path, kind):
    args, _kw, _paused, path = save(tmp_path, kind, root_solver="newton")

    def mutate(fields):
        if kind == "missing_solver":
            del fields["root_solver"]  # half metadata must never silently become the bisection control
        elif kind == "missing_cap":
            del fields["newton_max_iterations"]
        elif kind == "unknown":
            fields["extra_solver_option"] = 1
        elif kind == "bogus_solver":
            fields["root_solver"] = "secant"
        elif kind == "bad_cap":
            fields["newton_max_iterations"] = 0
        else:
            fields["newton_max_iterations"] = 10**6

    rewrite(path, mutate)
    with pytest.raises((SedimentEventError, StormError)):
        load_checkpoint(path, args[1], args[2], args[6], IDENTITY)


def test_historical_control_without_either_field_still_loads_as_bisection(tmp_path):
    args, _kw, _paused, path = save(tmp_path, "historical_edit", root_solver="newton")

    def strip(fields):
        del fields["root_solver"]
        del fields["newton_max_iterations"]

    rewrite(path, strip)
    restored = load_checkpoint(path, args[1], args[2], args[6], IDENTITY)
    assert restored.control.storm.root_solver == "bisection"
    assert restored.control.storm.newton_max_iterations == 50
    found, _ = storm_control_fields(path)
    assert all(set(f) == OLD_STORM_FIELDS for f in found)
