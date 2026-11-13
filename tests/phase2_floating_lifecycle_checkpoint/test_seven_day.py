"""S1: a REAL seven-day N=3 run with intervening native events + property refreshes: pause/resume before/after the due days, lagging replay (existing frames verified, never overwritten), duplicate restore, chained re-save of a
restored head; every restored continuation equals the uninterrupted run (digests, water, ledger, sealed frame bytes)."""
from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from bridge_builders import build, complete
from maple_syrup import floating_daily as fd
from maple_syrup import floating_daily_output as fo
from maple_syrup import floating_event_daily_bridge as br
from maple_syrup import floating_event_lifecycle as lc
from maple_syrup import native_floating_soil as nfs
from maple_syrup import recharge as rc
from maple_syrup import soil_moisture_output as g0
from maple_syrup.rainfall import constant_rainfall

from lcp_builders import AVAIL, PART, RUN, load, save

LIM = g0.MoistureOutputLimits(max_snapshots=10, max_total_bytes=10**7, max_manifest_bytes=8192, chunk_bytes=4096, host_memory_budget_bytes=10**7, host_reserve_bytes=0, disk_reserve_bytes=0)
BIG = 10**12
EVENTS = (1, 4)  # a real native storm + selected refresh follows these days (indices)


def big():
    return BIG


def changed_boundary(tmp, mp, backend, windowed):
    """A boundary whose first event's refresh CHANGED the properties (declared provider perturbation), with the genuine generation-0 topology."""
    owner, bed, factory = build(tmp, backend, windowed, mobile_kg=0.1)
    real = nfs._batch_properties
    def scaled(b_, c_, k_, p_):
        f, sa, k = real(b_, c_, k_, p_)
        return f * 0.9, sa * 0.9, k

    mp.setattr(nfs, "_batch_properties", scaled)
    lin, occ, st = lc.open_floating_event(owner, bed, factory, available_bytes_fn=AVAIL)
    original = occ.event._initial_topology
    h = complete(occ.event, st)
    b = lc.close_occurrence(lin, occ, occ.event.completion_capability(h, h.state), h.state, available_bytes_fn=AVAIL)
    mp.undo()
    return SimpleNamespace(lin=lin, b=b, factory=factory, original=original)


def event_cycle(lin, boundary, factory):
    sched = constant_rainfall(boundary.t_s, boundary.t_s + 30.0, 10.0)
    occ, s = lc.begin_occurrence(lin, boundary, lambda vert, topology: factory(vert, topology, sched), available_bytes_fn=AVAIL)
    h = complete(occ.event, s)
    return lc.close_occurrence(lin, occ, occ.event.completion_capability(h, h.state), h.state, available_bytes_fn=AVAIL)


def one_day(lin, boundary, ordinal, out, st=None):
    st = br.open_day(lin, boundary, partition=PART, run_digest=RUN, available_bytes_fn=AVAIL) if st is None else st
    n = st.ctx.n_cells
    r = br.day_action(st, plan=rc.plan_daily_rain(0.0, 0.0, 5.0, run_digest=RUN, ordinal=ordinal), pet_terrain_m_s=np.full(n, 3e-8), cover=np.full(n, 0.3), available_bytes_fn=AVAIL)
    t = fd.request_floating_withdrawal(r.state, expected_parent=fd.expect_parent(r.state), request_m=np.zeros((st.ctx.n_layers, n)), expected_sequence=0, available_bytes_fn=AVAIL)
    return fd.close_floating_day(t.state, expected_parent=fd.expect_parent(t.state), output=out, host_available_bytes_fn=big, disk_available_bytes_fn=big)


def tree(d: Path) -> dict:
    return {str(p.relative_to(d)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.rglob("*")) if p.is_file()}


class Drive:
    """One process-like segment: runs days on `lin` from `boundary`/`st`, records the receipts, saves after the requested closed days."""

    def __init__(self, ns, out, tmp, partition=PART):
        self.ns, self.out, self.tmp = ns, out, tmp
        self.rows, self.saved = [], {}

    def days(self, lin, boundary, first, last, st=None, save_after=()):
        for day in range(first, last + 1):
            res = one_day(lin, boundary, day, self.out, st=st if day == first else None)
            closed = res.state
            self.rows.append((day, res.notice.action, res.notice.due, closed.execution_digest, closed.digest, hashlib.sha256(np.asarray(closed.water.water_m).tobytes()).hexdigest()))
            if day in save_after:
                self.saved[day] = save(SimpleNamespace(lin=lin, factory=self.ns.factory, original=self.ns.original), self.tmp / f"ck{day}", closed)
            boundary = br.hand_back(lin, closed)
            if day in EVENTS:
                boundary = event_cycle(lin, boundary, self.ns.factory)
        return lin, boundary

    def restore(self, day):
        sv = self.saved[day]
        r = load(self.ns, sv.path, sv, partition=PART)
        self.out = fo.FloatingMoistureOutput.resume(self.out_dir, r.closed, expected_parent=fd.expect_parent(r.closed), limits=LIM, host_available_bytes_fn=big)
        return r


def mk(tmp_path, mp, windowed):
    ns = changed_boundary(tmp_path / "a", mp, "numpy", windowed)
    outdir = tmp_path / "out"
    outdir.mkdir()
    origin = br.open_day(ns.lin, ns.b, partition=PART, run_digest=RUN, available_bytes_fn=AVAIL)
    out = fo.FloatingMoistureOutput.create(outdir, origin, expected_parent=fd.expect_parent(origin), every_n_days=3, limits=LIM, host_available_bytes_fn=big, disk_available_bytes_fn=big)
    d = Drive(ns, out, tmp_path)
    d.out_dir = outdir
    return d, origin


@pytest.fixture(params=[False, True], ids=["dense", "windowed"])
def windowed(request):
    return request.param


@pytest.fixture
def reference(tmp_path_factory, windowed):
    tmp = tmp_path_factory.mktemp("ref")
    mp = pytest.MonkeyPatch()
    try:
        d, origin = mk(tmp, mp, windowed)
        lin, b = d.days(d.ns.lin, d.ns.b, 0, 6, st=origin)
    finally:
        mp.undo()
    return SimpleNamespace(rows=d.rows, tree=tree(d.out_dir), lin=lin, b=b, ledger=dict(b.ledger), water=np.asarray(b.root.water_state.water_m).copy(), cursor=b.cursor, windowed=windowed)


def check(ref, d, lin, b):
    assert d.rows == ref.rows
    assert tree(d.out_dir) == ref.tree  # the sealed frame bytes are exactly the uninterrupted run's
    assert b.ledger == ref.ledger and b.cursor == ref.cursor and np.array_equal(np.asarray(b.root.water_state.water_m), ref.water)


def test_the_uninterrupted_reference_has_due_frames_only_on_days_3_and_6(reference):
    assert [(r[0], r[1]) for r in reference.rows] == [(0, "not_due"), (1, "not_due"), (2, "saved"), (3, "not_due"), (4, "not_due"), (5, "saved"), (6, "not_due")]


def test_pause_before_the_due_days_then_chained_restores_equal_the_uninterrupted_run(tmp_path, reference, windowed):
    mp = pytest.MonkeyPatch()
    try:
        d, origin = mk(tmp_path, mp, windowed)
        lin, b = d.days(d.ns.lin, d.ns.b, 0, 1, st=origin, save_after=(1,))  # closed day 2: BEFORE the due day 3
        r = d.restore(1)
        # continue exactly as the uninterrupted process: the closed day 2 is handed back, the event + refresh follow, then days 3..5 (due 3 written here), save before the due day 6
        boundary = br.hand_back(r.lineage, r.closed)
        boundary = event_cycle(r.lineage, boundary, d.ns.factory)
        lin, b = d.days(r.lineage, boundary, 2, 4, save_after=(4,))
        r2 = d.restore(4)  # the restored head of a RESTORED lineage is saveable and restorable again
        boundary = br.hand_back(r2.lineage, r2.closed)
        boundary = event_cycle(r2.lineage, boundary, d.ns.factory)
        lin, b = d.days(r2.lineage, boundary, 5, 6)
    finally:
        mp.undo()
    check(reference, d, lin, b)


def test_pause_after_the_due_days_and_a_duplicate_restore(tmp_path, reference, windowed):
    mp = pytest.MonkeyPatch()
    try:
        d, origin = mk(tmp_path, mp, windowed)
        lin, b = d.days(d.ns.lin, d.ns.b, 0, 2, st=origin, save_after=(2,))  # AFTER the due day 3 was written
        dup = load(d.ns, d.saved[2].path, d.saved[2], partition=PART)  # a duplicate restore is an independent lineage of the same file
        r = d.restore(2)
        assert dup.closed.execution_digest == r.closed.execution_digest and dup.lineage is not r.lineage
        lin, b = d.days(r.lineage, br.hand_back(r.lineage, r.closed), 3, 5, save_after=(5,))
        r2 = d.restore(5)  # AFTER the due day 6
        lin, b = d.days(r2.lineage, br.hand_back(r2.lineage, r2.closed), 6, 6)
    finally:
        mp.undo()
    check(reference, d, lin, b)


def test_a_lagging_checkpoint_replays_existing_frames_without_overwriting_them(tmp_path, reference, windowed):
    mp = pytest.MonkeyPatch()
    try:
        d, origin = mk(tmp_path, mp, windowed)
        d.days(d.ns.lin, d.ns.b, 0, 5, st=origin, save_after=(2,))  # the writer reaches day 6 (frame 6 sealed); the lifecycle file is from the closed day 3
        sealed = tree(d.out_dir)
        n_rows = len(d.rows)
        r = d.restore(2)  # a LAGGING restart: the output record is ahead of the restored state
        boundary = br.hand_back(r.lineage, r.closed)
        d.days(r.lineage, boundary, 3, 5)  # replay: the existing frame 6 is VERIFIED against the replayed bytes
        replay = d.rows[n_rows:]
        assert [x[1] for x in replay] == ["not_due", "not_due", "verified_existing"]
        assert tree(d.out_dir) == sealed  # nothing overwritten
        assert [x[3:5] for x in replay] == [x[3:5] for x in d.rows[3:6]]  # the replayed execution/state digests equal the first pass
    finally:
        mp.undo()
