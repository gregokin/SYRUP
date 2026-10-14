"""CW-B73: reporting-only corrections on the frozen B71 runtime (Q1 D1-D3).

* D1: the DENSE summary's sink limitation follows the run's actual graph policy (strict text unchanged).
* D2: every bounded commit record carries the ACTUAL `BedCommit.terminal_storage_change` host-int dict that
  `commit_bed` already computed (key `terminal_storage_change`); nothing is recomputed, no record exists without a
  commit, and `max_commit_log` still bounds the log.
* D3: module docstrings / one comment corrected; no code change.

Scope guards compare THIS tree with the immutable frozen B71 tree: every package file is identical except the five
assigned files; there only the module docstrings, the new `_dense_sink_limitation` helper and its one call, and ONE
added commit-record key differ (asserted by AST). The historical B71 guards stay context-specific (not rerun here as
oracles of this tree). CPU tests were run by the author; the CuPy variant is for root.
"""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("maple")

from pit_helpers import pit_inputs
from test_coupled_backends import H
from test_sediment_event import run as run_strict
from test_sediment_event import valley_case

import maple_syrup.sediment_event as se
from maple_syrup.complete_event import CompletionPolicy, complete_event
from maple_syrup.event_experiment import _dense_sink_limitation
from maple_syrup.rainfall import constant_rainfall
from maple_syrup.routing import RoutingGraphError, build_routing_graph
from maple_syrup.sediment_event import (
    SedimentEventControl,
    evolve_sediment_event,
)
from maple_syrup.storm import StormControl

TREE = Path(__file__).resolve().parents[2]
PKG = TREE / "src" / "maple_syrup"
B71 = Path("/home/okin/SYRUP/outputs/conservative_water/cw_b71_runtime_frozen/src/maple_syrup")
ASSIGNED = {"sediment_event.py", "event_experiment.py", "characteristic_transport.py", "routing.py",
            "characteristic_sparse.py"}
KEYS = {"n_terminal_before", "n_terminal_after", "n_terminal_created", "n_terminal_removed"}
STRICT_TEXT = "fixed exporting ring; new pits refused"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Scope against the frozen B71 source
# ---------------------------------------------------------------------------


def _strip_docstrings(tree):
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)) and node.body \
                and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) \
                and isinstance(node.body[0].value.value, str):
            node.body = node.body[1:]
    return tree


def _defs(path):
    return {n.name: n for n in _strip_docstrings(ast.parse(path.read_text())).body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef))}


def test_only_assigned_files_differ_from_frozen_b71():
    if not B71.is_dir():
        pytest.skip("frozen B71 tree unavailable")
    mine = {p.relative_to(PKG).as_posix(): _sha(p) for p in PKG.rglob("*.py") if "__pycache__" not in p.parts}
    frozen = {p.relative_to(B71).as_posix(): _sha(p) for p in B71.rglob("*.py") if "__pycache__" not in p.parts}
    assert set(mine) == set(frozen)
    assert {k for k in mine if mine[k] != frozen[k]} == ASSIGNED


@pytest.mark.parametrize("name", ["characteristic_transport.py", "routing.py", "characteristic_sparse.py"])
def test_docstring_only_files_have_identical_code(name):
    if not B71.is_dir():
        pytest.skip("frozen B71 tree unavailable")
    old, new = (_strip_docstrings(ast.parse((d / name).read_text())) for d in (B71, PKG))
    assert ast.dump(old) == ast.dump(new)


def test_sediment_event_differs_only_by_docstrings_and_one_commit_record_key():
    if not B71.is_dir():
        pytest.skip("frozen B71 tree unavailable")
    old, new = _defs(B71 / "sediment_event.py"), _defs(PKG / "sediment_event.py")
    assert set(old) == set(new)
    for name in set(old) - {"_commit_and_reroute"}:
        assert ast.dump(old[name]) == ast.dump(new[name]), name  # B71 helper/core/step: all identical
    # in `_commit_and_reroute` the record dict gains exactly the one key, carrying the actual BedCommit dict
    record = [n for n in ast.walk(new["_commit_and_reroute"]) if isinstance(n, ast.Dict)
              and any(isinstance(k, ast.Constant) and k.value == "terminal_storage_change" for k in n.keys)]
    assert len(record) == 1
    i = [k.value for k in record[0].keys].index("terminal_storage_change")
    assert ast.unparse(record[0].values[i]) == "dict(outcome.terminal_storage_change)"
    del record[0].keys[i], record[0].values[i]
    assert ast.dump(old["_commit_and_reroute"]) == ast.dump(new["_commit_and_reroute"])
    # module-level statements other than function/class definitions (constants, imports) are unchanged
    rest = [ast.dump(n) for n in _strip_docstrings(ast.parse((B71 / "sediment_event.py").read_text())).body
            if not isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    rest_new = [ast.dump(n) for n in _strip_docstrings(ast.parse((PKG / "sediment_event.py").read_text())).body
                if not isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    assert rest == rest_new


def test_event_experiment_differs_only_by_the_limitation_helper_and_its_call():
    if not B71.is_dir():
        pytest.skip("frozen B71 tree unavailable")
    old, new = _defs(B71 / "event_experiment.py"), _defs(PKG / "event_experiment.py")
    assert set(new) == set(old) | {"_dense_sink_limitation"}
    for name in set(old) - {"run_complete_case"}:
        assert ast.dump(old[name]) == ast.dump(new[name]), name
    calls = [n for n in ast.walk(new["run_complete_case"]) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == "_dense_sink_limitation"]
    assert len(calls) == 1 and ast.unparse(calls[0]) == "_dense_sink_limitation(graph_policy)"
    # replacing the call by the original literal restores the B71 function exactly
    for node in ast.walk(new["run_complete_case"]):
        for field, value in ast.iter_fields(node):
            if isinstance(value, list):
                for j, item in enumerate(value):
                    if item is calls[0]:
                        value[j] = ast.Constant(STRICT_TEXT)
    assert ast.dump(old["run_complete_case"]) == ast.dump(new["run_complete_case"])


# ---------------------------------------------------------------------------
# D1: the dense limitation states the policy the refreshed graph actually enforces
# ---------------------------------------------------------------------------


def _policy(pit=False, flat=False, masked=False):
    return {"allow_masked_nodata": masked, "nodata_value": -9999.0 if masked else None, "allow_pit_storage": pit,
            "allow_flat_storage": flat}


def _dem(kind):
    """A walled 3x1 interior: a strict pit (lowest cell 0), or a flat sink pair (cells 0 and 1 equal)."""
    z = np.full((5, 3), 2.0)
    z[1:4, 1] = [0.10, 0.11, 0.12] if kind == "pit" else [0.10, 0.10, 0.12]
    return z, np.zeros(z.shape, dtype=bool)


def _builds(kind, **policy):
    z, exports = _dem(kind)
    try:
        build_routing_graph(z, exports, np.ones((3, 1)), 0.5, **policy)
        return True
    except RoutingGraphError:
        return False


def test_dense_limitation_text_matches_the_actual_routing_policy():
    assert _dense_sink_limitation(_policy()) == STRICT_TEXT  # strict default: the B71 text, byte for byte
    assert _dense_sink_limitation(_policy(masked=True)) == STRICT_TEXT
    pit_only, pit_flat = _dense_sink_limitation(_policy(pit=True)), _dense_sink_limitation(_policy(pit=True, flat=True))
    assert "new pits become terminal storage" in pit_only and "new flat sinks refused" in pit_only
    assert "new pits and flat sinks become terminal storage" in pit_flat and "refused" not in pit_flat
    # the statements are those of build_routing_graph (the builder refresh_routing calls after each commit)
    assert not _builds("pit") and not _builds("flat")  # strict: a pit and a flat sink are refused
    assert _builds("pit", allow_pit_storage=True) and not _builds("flat", allow_pit_storage=True)
    assert _builds("flat", allow_pit_storage=True, allow_flat_storage=True)


# ---------------------------------------------------------------------------
# D2: the actual BedCommit counts are carried into each bounded commit record
# ---------------------------------------------------------------------------


def _spy_commit_bed(monkeypatch):
    seen = []
    real = se.commit_bed

    def spy(*a, **k):
        out = real(*a, **k)
        seen.append(None if not out.did_commit else (out.terminal_storage_change, dict(out.terminal_storage_change)))
        return out

    monkeypatch.setattr(se, "commit_bed", spy)
    return seen


def test_pit_commits_record_the_exact_bedcommit_counts_including_a_removed_terminal_cell(monkeypatch):
    inputs = pit_inputs(np, flat=True)  # two flat terminal sinks; the fed one fills and reroutes (2 -> 1)
    seen = _spy_commit_bed(monkeypatch)
    r = evolve_sediment_event(inputs["state0"], inputs["ctx"], inputs["column"], inputs["field"],
                              constant_rainfall(0.0, 40.0, 72.0), inputs["vegetation"], inputs["sediment"], 120.0,
                              SedimentEventControl(), report_every_s=40.0)
    committed = [s for s in seen if s is not None]
    assert r.n_commits == len(committed) >= 1 and len(r.commit_log) == len(committed)
    for record, (original, copy) in zip(r.commit_log, committed, strict=True):
        change = record["terminal_storage_change"]
        assert change == copy and change is not original  # carried (a plain copy), never recomputed
        assert set(change) == KEYS and all(type(v) is int for v in change.values())
    first, last = committed[0][1], committed[-1][1]
    assert first["n_terminal_before"] == 2 and last["n_terminal_after"] == int(np.sum(H(r.state.graph.pit_storage))) == 1
    assert sum(c["n_terminal_removed"] for _o, c in committed) == 1 and sum(c["n_terminal_created"] for _o, c in committed) == 0
    for _o, c in committed:
        assert c["n_terminal_after"] == c["n_terminal_before"] + c["n_terminal_created"] - c["n_terminal_removed"]


def test_strict_commits_record_all_zero_counts_and_no_record_without_a_commit(monkeypatch):
    inputs = valley_case()
    seen = _spy_commit_bed(monkeypatch)
    r = run_strict(inputs, constant_rainfall(0.0, 30.0, 72.0), 90.0, cadence=30.0)
    committed = [s for s in seen if s is not None]
    assert len(seen) > len(committed) >= 1  # trigger evaluations without a commit add no record
    assert len(r.commit_log) == r.n_commits == len(committed)
    for record in r.commit_log:
        assert record["terminal_storage_change"] == dict.fromkeys(KEYS, 0)


def test_commit_log_stays_bounded_by_max_commit_log(monkeypatch):
    inputs = pit_inputs(np, flat=True)
    seen = _spy_commit_bed(monkeypatch)
    r = evolve_sediment_event(inputs["state0"], inputs["ctx"], inputs["column"], inputs["field"],
                              constant_rainfall(0.0, 40.0, 72.0), inputs["vegetation"], inputs["sediment"], 120.0,
                              SedimentEventControl(max_commit_log=1), report_every_s=40.0)
    committed = [s for s in seen if s is not None]
    assert r.n_commits == len(committed) >= 2 and len(r.commit_log) == 1
    assert r.commit_log[0]["terminal_storage_change"] == committed[0][1]


def test_wet_pause_checkpoint_and_fresh_restart_carry_the_new_record(tmp_path):
    from test_checkpoint_backend import IDENTITY, arrays_any, assert_bitwise

    from maple_syrup.checkpoint import load_checkpoint, save_checkpoint

    schedule = constant_rainfall(0.0, 40.0, 72.0)
    policy = CompletionPolicy(hold_s=10.0, terminal_storage_water=True)
    control = SedimentEventControl(storm=StormControl(max_dt_s=1.0))
    inputs = pit_inputs(np, flat=True, pit_ksat=0.0)
    ctx, column, sediment = inputs["ctx"], inputs["column"], inputs["sediment"]

    def complete(state, **kw):
        return complete_event(state, ctx, column, inputs["field"], schedule, inputs["vegetation"], sediment,
                              max_end_s=900.0, control=control, policy=policy, report_every_s=2.0, **kw)

    paused = complete(inputs["state0"], checkpoint_callback=lambda p: p.result.n_commits >= 1 and p.result.state.t_s >= 20.0)
    assert paused.status == "wet" and paused.result.n_commits >= 1
    assert all("terminal_storage_change" in rec for rec in paused.result.commit_log)
    path = tmp_path / "wet"
    save_checkpoint(path, paused, ctx, column, sediment, IDENTITY)
    restored = load_checkpoint(path, ctx, column, sediment, IDENTITY)
    assert restored.result.commit_log == paused.result.commit_log  # the new dict round-trips exactly
    # fresh-process-equivalent resume (new state objects from the bundle) vs the uninterrupted event, both paused at
    # the same later original boundary (this ponded flat-basin case does not complete within the window)
    later = lambda p: p.result.state.t_s >= 100.0
    full = complete(inputs["state0"], checkpoint_callback=later)
    resumed = complete(restored.result.state, continuation=restored, checkpoint_callback=later)
    assert full.status == resumed.status == "wet" and full.result.state.t_s == resumed.result.state.t_s >= 100.0
    assert resumed.result.commit_log == full.result.commit_log
    # a commit made AFTER the restart point (the fed flat fills and reroutes) carries its actual removal count
    after = [rec for rec in resumed.result.commit_log if rec["t_s"] > paused.result.state.t_s]
    assert after and sum(rec["terminal_storage_change"]["n_terminal_removed"] for rec in after) == 1
    assert all(set(rec["terminal_storage_change"]) == KEYS for rec in full.result.commit_log)
    a, b = dict(arrays_any(full.result.state)), dict(arrays_any(resumed.result.state))
    assert a.keys() == b.keys()
    for name, value in a.items():
        assert_bitwise(value, b[name], name)


# ---------------------------------------------------------------------------
# Device variant (root; skipped without CuPy and a GPU)
# ---------------------------------------------------------------------------


def test_device_pit_commits_record_the_same_counts_as_the_host(gpu, monkeypatch):
    records = {}
    for name, xp in (("host", np), ("device", gpu)):
        inputs = pit_inputs(xp, flat=True)
        r = evolve_sediment_event(inputs["state0"], inputs["ctx"], inputs["column"], inputs["field"],
                                  constant_rainfall(0.0, 40.0, 72.0), inputs["vegetation"], inputs["sediment"], 120.0,
                                  SedimentEventControl(), report_every_s=40.0)
        records[name] = [rec["terminal_storage_change"] for rec in r.commit_log]
        assert all(all(type(v) is int for v in c.values()) for c in records[name])
    assert records["device"] == records["host"]
