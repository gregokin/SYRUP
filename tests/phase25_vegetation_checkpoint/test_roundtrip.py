"""Vegetation S1: boundary and closed-day files (NumPy/Numba x Dense/published Windowed) on REAL native lifecycle states with the ACTUAL LEAF projection."""
from __future__ import annotations

import numpy as np
import pytest
from vck_builders import (
    AVAIL,
    PART,
    boundary_vns,
    closed_day_vns,
    load,
    run_day,
    save,
    storm,
)

from maple_syrup import floating_event_daily_bridge as br
from maple_syrup import floating_event_lifecycle as lc
from maple_syrup import vegetation_lifecycle as vl

MATRIX = [("numpy", False), ("numpy", True), ("numba", False), ("numba", True)]


@pytest.fixture(params=MATRIX, ids=lambda p: f"{p[0]}-{'windowed' if p[1] else 'dense'}")
def mx(request):
    if request.param[0] == "numba":
        pytest.importorskip("numba")
    return request.param


def test_boundary_file_roundtrip_and_continuation(tmp_path, leaf, mx):
    ns = boundary_vns(tmp_path / "a", leaf, *mx)
    head = ns.lin.head
    rec0 = vl.current_record(head)
    saved = save(ns, tmp_path / "ck")
    assert ns.lin.head is head and vl.current_record(head) is rec0  # a save never moves the head or the record
    r = load(ns, tmp_path / "ck", saved)
    b2 = r.head
    rec2 = vl.current_record(b2)
    assert r.kind == "boundary" and r.lineage.head is b2 and r.lineage is not ns.lin
    assert (rec2.generation, rec2.recipe_id, rec2.parent_recipe_id, rec2.binding_sha256) == (rec0.generation, rec0.recipe_id, rec0.parent_recipe_id, rec0.binding_sha256)
    assert rec2.binding.origin == "cached_derived" and rec0.binding.origin != "cached_derived"
    assert (rec2.applied_law_sha256, rec2.applied_g_sha256, rec2.applied_matrix_op_sha256) == (rec0.applied_law_sha256, rec0.applied_g_sha256, rec0.applied_matrix_op_sha256)
    a0, a2 = vl.vegetation_attribution(head), vl.vegetation_attribution(b2)
    assert {k: v for k, v in a0.items() if k != "backend"} == {k: v for k, v in a2.items() if k != "backend"}
    assert np.array_equal(b2.root.water_state.water_m, head.root.water_state.water_m) and np.array_equal(b2.root.history.wetting_m, head.root.history.wetting_m)
    assert np.array_equal(b2.root.history.cumulative_intake_m, head.root.history.cumulative_intake_m)
    assert b2.t_s == head.t_s and b2.seq == head.seq and b2.ledger == head.ledger and b2.routing[0].topology_sha256 == head.routing[0].topology_sha256
    assert b2.root.matrix_op.sha256 == head.root.matrix_op.sha256 and b2.root.pref_op.sha256 == head.root.pref_op.sha256
    n1 = storm(ns.lin, head, ns.factory)
    n2 = storm(r.lineage, b2, ns.factory)
    assert np.array_equal(n1.root.water_state.water_m, n2.root.water_state.water_m) and n1.ledger["rain_m3"] == n2.ledger["rain_m3"] and n1.t_s == n2.t_s
    assert np.array_equal(n1.root.history.wetting_m, n2.root.history.wetting_m) and vl.current_record(n1).recipe_id == vl.current_record(n2).recipe_id


def test_closed_day_file_roundtrip_preserves_the_chain_and_the_continuation(tmp_path, leaf, mx):
    ns = closed_day_vns(tmp_path / "a", leaf, *mx, days=2)
    head, closed = ns.lin.head, ns.closed
    before = (closed.execution_digest, closed.digest, closed.parent_digest, closed.revision, closed.request_sequence, closed.cursor, closed.ledger, closed.frame_identity())
    saved = save(ns, tmp_path / "ck", closed)
    assert ns.lin.head is head
    r = load(ns, tmp_path / "ck", saved, partition=PART)
    c2 = r.closed
    assert (c2.execution_digest, c2.digest, c2.parent_digest, c2.revision, c2.request_sequence, c2.cursor, c2.ledger, c2.frame_identity()) == before
    assert isinstance(r.lineage.head, lc.DailyHold) and c2.ctx.hold is r.lineage.head and np.array_equal(c2.water.water_m, closed.water.water_m)
    assert r.lineage.head.vegetation.recipe_id == head.vegetation.recipe_id
    b1 = br.hand_back(ns.lin, closed)
    b2 = br.hand_back(r.lineage, c2)
    assert b1.ledger == b2.ledger and b1.t_s == b2.t_s and b1.cursor == b2.cursor and b1.seq == b2.seq
    s1 = br.open_day(ns.lin, b1, partition=PART, available_bytes_fn=AVAIL)
    s2 = br.open_day(r.lineage, b2, partition=PART, available_bytes_fn=AVAIL)
    d1, d2 = run_day(s1, 2).state, run_day(s2, 2).state
    assert d1.execution_digest != s1.execution_digest and (d1.execution_digest, d1.digest) == (d2.execution_digest, d2.digest) and np.array_equal(d1.water.water_m, d2.water.water_m)
