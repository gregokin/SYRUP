"""WH1: a wind-refreshed Boundary round-trips through the UNCHANGED native lifecycle checkpoint and resumes into another wind and a day without reinitialising the bed, soil, properties, cursor, ledgers, J/I or the clock. Authored, NOT executed by its
author. No codec, profile or provenance restriction was changed; a fresh-process (separate interpreter) restart is NOT covered here."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import wind_builders as wb
from lcp_builders import IDENT, PART, run_day

from maple_syrup import floating_event_daily_bridge as br
from maple_syrup import floating_event_lifecycle as lc
from maple_syrup import floating_lifecycle_checkpoint as lcp
from maple_syrup import floating_wind_handoff as wh

AVAIL = wb.AVAIL
CELL = (1, 1)


def _save(ns, path):
    return lcp.save_floating_lifecycle_checkpoint(path, ns.lin, ns.factory, IDENT, original_topology=ns.original, available_bytes_fn=AVAIL)


def _load(ns, path, saved):
    return lcp.load_floating_lifecycle_checkpoint(path, ns.factory, IDENT, saved.manifest_sha256, original_topology=ns.original, available_bytes_fn=AVAIL)


def test_the_initial_wind_boundary_is_saveable_and_resumes(tmp_path):
    ns = wb.started(depth=0.003)
    head = ns.lin.head
    saved = _save(ns, tmp_path / "ck0")
    assert ns.lin.head is head  # a save never moves the head
    r = _load(ns, tmp_path / "ck0", saved)
    b2 = r.head
    assert r.kind == "boundary" and r.lineage.head is b2 and b2.t_s == 0.0 and np.array_equal(b2.root.water_state.water_m, head.root.water_state.water_m) and b2.ledger == head.ledger
    assert np.array_equal(b2.surface_depth_m, head.surface_depth_m) and float(np.asarray(b2.surface_depth_m).max()) > 0.0  # the initial ponded water survives the file


def test_a_refreshed_wind_boundary_roundtrips_and_the_restored_lineage_resumes(tmp_path):
    ns = wb.started()
    child = wb.child_move(ns.b.bed, ns.ctx, CELL, (1, 2), klass=0)  # ACTUAL native sediment exchange + ledger: a bed the native checkpoint validators and ledger reconciliation accept (no direct class overwrite)
    old, nb, cand = wb.wind(ns, child, end_s=75.0)
    assert cand.receipt()["properties"]["root_replaced"] is True
    head = ns.lin.head
    saved = _save(ns, tmp_path / "ck")
    assert ns.lin.head is head
    r = _load(ns, tmp_path / "ck", saved)
    b2 = r.head
    assert r.kind == "boundary" and r.lineage.head is b2 and r.lineage is not ns.lin
    assert b2.t_s == nb.t_s == 75.0 and b2.seq == nb.seq and b2.ledger == nb.ledger and b2.cursor == nb.cursor and b2.provenance == nb.provenance and b2.role == "boundary"
    assert b2.root.generation == nb.root.generation and b2.root.column.sha256 == nb.root.column.sha256 and b2.root.matrix_op.sha256 == nb.root.matrix_op.sha256  # the refreshed property generation is reproduced
    assert np.array_equal(b2.root.water_state.water_m, nb.root.water_state.water_m) and b2.root.water_state.t_s == nb.root.water_state.t_s
    assert np.array_equal(b2.root.history.wetting_m, nb.root.history.wetting_m) and np.array_equal(b2.root.history.cumulative_intake_m, nb.root.history.cumulative_intake_m)
    assert b2.routing[0].topology_sha256 == nb.routing[0].topology_sha256 and np.array_equal(b2.surface_depth_m, nb.surface_depth_m)
    # the context is rebuilt from the verified restored boundary and the original static case + the expected record: nothing is regenerated from the initial terrain
    wctx2 = wh.restore_wind_context(b2, ns.ctx, ns.wctx.record())
    assert wctx2.record() == ns.wctx.record()
    ns2 = SimpleNamespace(lin=r.lineage, b=b2, wctx=wctx2)
    _o1, n1, c1 = wb.wind(ns, wb.child_same(ns.b.bed), end_s=125.0, occurrence_id="o2", start_s=75.0)
    _o2, n2, c2 = wb.wind(ns2, wb.child_same(ns2.b.bed), end_s=125.0, occurrence_id="o2", start_s=75.0)
    assert c1.receipt()["properties"] == c2.receipt()["properties"] and n1.ledger == n2.ledger and n1.t_s == n2.t_s == 125.0
    assert np.array_equal(n1.root.water_state.water_m, n2.root.water_state.water_m) and np.array_equal(n1.root.history.wetting_m, n2.root.history.wetting_m)
    # a restored lineage starts with no last-occurrence record (process-local, bounded): the schedule cursor owns cross-process identity
    s1 = br.open_day(ns.lin, n1, partition=PART, run_digest=wb.RUN, available_bytes_fn=AVAIL)
    s2 = br.open_day(r.lineage, n2, partition=PART, run_digest=wb.RUN, available_bytes_fn=AVAIL)
    d1, d2 = run_day(s1, 0).state, run_day(s2, 0).state
    assert d1.water.t_s == d2.water.t_s == 125.0 + 86400.0 and np.array_equal(d1.water.water_m, d2.water.water_m)
    b3, b4 = br.hand_back(ns.lin, d1), br.hand_back(r.lineage, d2)
    assert b3.cursor == b4.cursor and b3.cursor.ordinal == 1 and b3.ledger == b4.ledger


def test_a_wind_boundary_with_a_pending_recession_is_not_saveable(tmp_path, monkeypatch):
    from maple_syrup import native_floating_soil as nfs

    ns = wb.started(bottom="closed", saturation=0.999999)
    real = nfs._batch_properties
    monkeypatch.setattr(nfs, "_batch_properties", lambda batch, cfg, K, pb: (lambda f, s, k: (f * 0.5, s * 0.5, k))(*real(batch, cfg, K, pb)))  # DECLARED perturbation: the capacity shrinks and wakes the flow
    _o, nb, _c = wb.wind(ns, wb.child_move(ns.b.bed, ns.ctx, CELL, (1, 2), klass=0), end_s=40.0)
    monkeypatch.undo()
    assert nb.role == "pending_recession"
    import pytest

    with pytest.raises(lcp.LifecycleCheckpointError):
        _save(ns, tmp_path / "ck")  # a pending recession is not carried by the file: the existing restriction is unchanged
    assert isinstance(ns.lin.head, lc.Boundary)
