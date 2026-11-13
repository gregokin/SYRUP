"""Phase 1.5 S2 correction01 (Root A1): the shared physical body gained an OPT-IN recharge seam. The historical whole-function AST fixture is obsolete by
design (kept unmodified in its archive); these controls replace it with opt-in-specific checks: (1) every use of the new parameter in
`_physical_trial` sits under an explicit recharge guard, (2) the actual stage CALL ORDER of the default path is the historical one (instrumented), and
the recharge path swaps exactly the column call, (3) ordinary receipts/ledgers keep their historical types and keys. The cursor-free golden trajectory
against the 587e96e sources (task-root `s2_golden_lifecycle.py`) is the behavioural identity control."""
from __future__ import annotations

import ast
import inspect

import numpy as np
from dcases import DAY, act, build, dw, pet, start, wl
from storm_cases import layers

from maple_syrup import interevent_transpiration as it
from maple_syrup import interevent_water as iw
from maple_syrup import matrix_infiltration as mi
from maple_syrup import preferential_infiltration as pi
from maple_syrup import soil_evaporation as se
from maple_syrup import subsurface as ss

GRID = np.full((4, 6), 0.2)


def _guarded_uses():
    tree = ast.parse(inspect.getsource(iw._physical_trial).lstrip())
    fn = tree.body[0]
    parents = {}
    for node in ast.walk(fn):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    bad = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Name) and node.id == "recharge_depth_m":
            cur, ok = node, False
            while cur in parents:
                cur = parents[cur]
                if isinstance(cur, (ast.If, ast.IfExp)) and any(isinstance(n, ast.Name) and n.id == "recharge_depth_m" for n in ast.walk(cur.test)):
                    ok = True
                    break
            if not ok and not isinstance(parents.get(node), ast.arg):
                bad.append(node.lineno)
    return bad


def test_every_use_of_the_recharge_parameter_in_the_shared_body_is_under_an_explicit_guard():
    assert _guarded_uses() == []  # (the signature default itself is an `ast.arg`)
    sig = inspect.signature(iw._physical_trial)
    assert sig.parameters["recharge_depth_m"].default is None  # historical callers (positional 11 args) are unchanged


def _instrument(monkeypatch, log):
    for mod, name in ((pi, "preferential_step"), (mi, "matrix_step"), (se, "stage_evaporation"), (ss, "step"), (ss, "step_with_recharge")):
        real = getattr(mod, name)
        monkeypatch.setattr(mod, name, (lambda r, n: (lambda *a, **k: (log.append(n), r(*a, **k))[1]))(real, name))


def test_default_path_stage_order_is_the_historical_one_and_recharge_swaps_only_the_column_call(monkeypatch):
    c = build(GRID, lay=layers(bottom="open"), theta=(0.1, 0.1))
    ie = wl.enter_interevent_from_boundary(c.lop, c.s0, expected_parent=wl.expect_parent(c.s0))
    log = []
    _instrument(monkeypatch, log)
    it.general_interevent_water_step(ie.binding.gop, ie.payload, pet(c, 0.0), 60.0, phase="interevent")
    assert log == ["preferential_step", "matrix_step", "stage_evaporation", "step"]
    log.clear()
    it.general_interevent_recharge_step(ie.binding.gop, ie.payload, pet(c, 0.0), 60.0, recharge_depth_m=1e-3, phase="interevent")
    assert log == ["preferential_step", "matrix_step", "stage_evaporation", "step_with_recharge"]  # same stages, ONE column call, never both


def test_ordinary_receipts_ledgers_and_state_records_keep_their_historical_shape():
    c = build(GRID, lay=layers(bottom="open"), theta=(0.1, 0.1))
    ie = wl.enter_interevent_from_boundary(c.lop, c.s0, expected_parent=wl.expect_parent(c.s0))
    cand = it.general_interevent_water_step(ie.binding.gop, ie.payload, pet(c, 0.0), 60.0, phase="interevent")
    keys = set(cand.receipt.as_dict())
    assert type(cand.receipt) is iw.InterEventStepReceipt and not any(k.startswith("recharge") for k in keys)
    assert cand.state.ledger.hydrology.rain_m3 == ie.payload.ledger.hydrology.rain_m3 == 0.0
