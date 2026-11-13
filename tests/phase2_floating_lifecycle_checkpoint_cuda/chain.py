"""Backend-neutral drivers shared by the GPU definitions and a HOST control: the seven-day N=3 chain with an intervening real native event + property refresh, paused/restored through a lifecycle file. The device test passes the
CUDA outer and a device `Side`; the host control passes the CPU outer and the matched CPU twin, which validates the sequence logic (days, notices, frames, restart) without a device."""
from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from maple_syrup import floating_daily as fd
from maple_syrup import floating_daily_output as fo
from maple_syrup import floating_event_daily_bridge as br
from maple_syrup import floating_event_lifecycle as lc
from maple_syrup import native_floating_soil as nfs
from maple_syrup import recharge as rc
from maple_syrup import soil_evaporation as se
from maple_syrup import soil_moisture_output as g0
from maple_syrup.rainfall import constant_rainfall

BIG = 1 << 40
RUN = "ab" * 32
PART = se.EvaporationPartition(1.25, 0.5, 1.0, 0.125, 0.75)
LIM = g0.MoistureOutputLimits(max_snapshots=10, max_total_bytes=10**8, max_manifest_bytes=8192, chunk_bytes=1 << 16, host_memory_budget_bytes=10**8, host_reserve_bytes=0, disk_reserve_bytes=0)
EVENTS = (2,)  # a real native event + selected property refresh follows this closed day (index)


def tree(d: Path) -> dict:
    return {str(p.relative_to(d)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.rglob("*")) if p.is_file()}


def refresh_event(side, lin, boundary, mp, host):
    """A REAL 30 s / 10 mm/h native event, genuinely completed, closed through the selected refresh with the provider values scaled by 0.9 AFTER completion and BEFORE the close (declared control)."""
    sched = constant_rainfall(boundary.t_s, boundary.t_s + 30.0, 10.0)
    occ, s = lc.begin_occurrence(lin, boundary, lambda v, topology=None: side.factory(v, topology, sched), available_bytes_fn=side.rdr, **side.hk)
    h = side.complete(occ, s)
    real = nfs._batch_properties

    def scaled(b_, c_, k_, p_):
        f, sa, k = real(b_, c_, k_, p_)
        return f * 0.9, sa * 0.9, k

    mp.setattr(nfs, "_batch_properties", scaled)
    try:
        return lc.close_occurrence(lin, occ, occ.event.completion_capability(h, h.state), h.state, available_bytes_fn=side.rdr, **side.hk)
    finally:
        mp.setattr(nfs, "_batch_properties", real)


def one_day(side, lin, boundary, ordinal, out, st=None):
    st = br.open_day(lin, boundary, partition=PART, run_digest=RUN, available_bytes_fn=side.rdr, **side.hk) if st is None else st
    n = st.ctx.n_cells
    pet, cov = side.arrays(n)
    kw = {"host_available_bytes_fn": side.hk["host_available_bytes_fn"]} if side.hk else {}
    r = br.day_action(st, plan=rc.plan_daily_rain(0.0, 0.0, 5.0, run_digest=RUN, ordinal=ordinal), pet_terrain_m_s=pet, cover=cov, available_bytes_fn=side.rdr, **kw)
    t = fd.request_floating_withdrawal(r.state, expected_parent=fd.expect_parent(r.state), request_m=side.xp.zeros((st.ctx.n_layers, n)), expected_sequence=0, available_bytes_fn=side.rdr, **side.hk)
    return fd.close_floating_day(t.state, expected_parent=fd.expect_parent(t.state), output=out, available_bytes_fn=side.rdr, host_available_bytes_fn=lambda: BIG, disk_available_bytes_fn=lambda: BIG)


def hand(side, lin, closed):
    return br.hand_back(lin, closed, available_bytes_fn=side.rdr, host_available_bytes_fn=side.hk["host_available_bytes_fn"]) if side.hk else br.hand_back(lin, closed)


def run(side, outdir, mp, host, *, save, load, pause_after=None):
    """Days 0..6 on `side`; `pause_after` (closed-day index) saves a lifecycle file, RESTORES it into a fresh lineage and continues from the restored closed day. Returns the receipts, the sealed frame tree and the final water."""
    outdir.mkdir(parents=True, exist_ok=True)
    boundary = side.boundary()
    lin = side.lin
    out = None
    rows = []
    restored = None
    for day in range(7):
        if out is None:  # the first day is opened here so the writer can be created from the origin day state (as the CPU suite does)
            st0 = br.open_day(lin, boundary, partition=PART, run_digest=RUN, available_bytes_fn=side.rdr, **side.hk)
            out = fo.FloatingMoistureOutput.create(outdir, st0, expected_parent=fd.expect_parent(st0), every_n_days=3, limits=LIM, host_available_bytes_fn=lambda: BIG, disk_available_bytes_fn=lambda: BIG)
            res = one_day(side, lin, boundary, day, out, st=st0)
        else:
            res = one_day(side, lin, boundary, day, out)
        closed = res.state
        rows.append((day, res.notice.action, res.notice.due, closed.execution_digest, closed.digest, hashlib.sha256(host(closed.water.water_m).tobytes()).hexdigest()))
        if pause_after == day:
            sv = save(lin, closed)
            restored = load(sv)
            lin, closed = restored.lineage, restored.closed
            assert closed.execution_digest == rows[-1][3] and closed.digest == rows[-1][4]  # the restored closed day IS the saved one
            out = fo.FloatingMoistureOutput.resume(outdir, closed, expected_parent=fd.expect_parent(closed), limits=LIM, host_available_bytes_fn=lambda: BIG)
        boundary = hand(side, lin, closed)
        if day in EVENTS:
            boundary = refresh_event(side, lin, boundary, mp, host)
    return SimpleNamespace(rows=rows, tree=tree(outdir), water=host(boundary.root.water_state.water_m).copy(), ledger=dict(boundary.ledger), restored=restored)


def check_equal(ref, got):
    assert [(r[0], r[1], r[2]) for r in ref.rows] == [(0, "not_due", False), (1, "not_due", False), (2, "saved", True), (3, "not_due", False), (4, "not_due", False), (5, "saved", True), (6, "not_due", False)]
    assert got.rows == ref.rows  # receipts, execution/state digests and water hashes equal the uninterrupted run's
    assert got.tree == ref.tree  # the sealed frame bytes are exactly the uninterrupted run's
    assert got.ledger == ref.ledger and np.array_equal(got.water, ref.water)
