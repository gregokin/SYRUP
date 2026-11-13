# ruff: noqa: RUF059
"""Evolving-model output: the every-N writer continues across accepted native events and property refreshes of the bridge profile (versioned evolving profile; the historical fixed profile stays strict)."""
from __future__ import annotations

import numpy as np
import pytest
from bridge_builders import AVAIL, RUN, complete, dry_boundary, dry_boundary_changed

from maple_syrup import floating_daily as fd
from maple_syrup import floating_daily_output as fo
from maple_syrup import floating_event_daily_bridge as br
from maple_syrup import floating_event_lifecycle as lc
from maple_syrup import native_floating_soil as nfs
from maple_syrup import recharge as rc
from maple_syrup import soil_evaporation as se
from maple_syrup import soil_moisture_output as g0
from maple_syrup.rainfall import constant_rainfall

PART = se.EvaporationPartition(1.25, 0.5, 1.0, 0.125, 0.75)
LIM = g0.MoistureOutputLimits(max_snapshots=10, max_total_bytes=10**7, max_manifest_bytes=8192, chunk_bytes=4096, host_memory_budget_bytes=10**7, host_reserve_bytes=0, disk_reserve_bytes=0)
BIG = 10**12


def big():
    return BIG


def one_day(lin, boundary, ordinal, out, partition=PART, st=None):
    st = br.open_day(lin, boundary, partition=partition, run_digest=RUN, available_bytes_fn=AVAIL) if st is None else st
    n = st.ctx.n_cells
    r = br.day_action(st, plan=rc.plan_daily_rain(0.0, 0.0, 5.0, run_digest=RUN, ordinal=ordinal), pet_terrain_m_s=np.full(n, 3e-8), cover=np.full(n, 0.3), available_bytes_fn=AVAIL)
    t = fd.request_floating_withdrawal(r.state, expected_parent=fd.expect_parent(r.state), request_m=np.zeros((st.ctx.n_layers, n)), expected_sequence=0, available_bytes_fn=AVAIL)
    res = fd.close_floating_day(t.state, expected_parent=fd.expect_parent(t.state), output=out, host_available_bytes_fn=big, disk_available_bytes_fn=big)
    return st, res


def event_cycle(lin, boundary, factory, monkeypatch=None, perturb=False):
    """A REAL native storm at the boundary's clock + the selected property refresh (optionally with a declared provider perturbation that changes the properties) -> the next boundary."""
    sched = constant_rainfall(boundary.t_s, boundary.t_s + 30.0, 10.0)
    occ, s = lc.begin_occurrence(lin, boundary, lambda vert, topology: factory(vert, topology, sched), available_bytes_fn=AVAIL)
    h = complete(occ.event, s)
    if perturb:
        real = nfs._batch_properties
        monkeypatch.setattr(nfs, "_batch_properties", lambda b_, c_, k_, p_: (lambda f, sa, k: (f * 0.9, sa * 0.9, k))(*real(b_, c_, k_, p_)))
    b = lc.close_occurrence(lin, occ, occ.event.completion_capability(h, h.state), h.state, available_bytes_fn=AVAIL)
    return b


def test_n3_seven_days_with_intervening_native_events_and_changed_and_noop_refreshes(tmp_path, monkeypatch):
    lin, b, factory, orig_sha = dry_boundary_changed(tmp_path / "a", monkeypatch)  # the first event's refresh CHANGED the properties
    (tmp_path / "out").mkdir()
    origin = br.open_day(lin, b, partition=PART, run_digest=RUN, available_bytes_fn=AVAIL)
    out = fo.FloatingMoistureOutput.create(tmp_path / "out", origin, expected_parent=fd.expect_parent(origin), every_n_days=3, limits=LIM, host_available_bytes_fn=big, disk_available_bytes_fn=big)
    notices, gens, columns, boundary = [], [], [], b
    for day in range(7):
        st, res = one_day(lin, boundary, day, out, st=origin if day == 0 else None)
        gens.append(st.ctx.property_generation)
        columns.append(st.ctx.root_column_sha256)
        notices.append(res.notice)
        boundary = br.hand_back(lin, res.state)
        if day in (1, 4):
            boundary = event_cycle(lin, boundary, factory)  # a REAL native event + NO-OP refresh (identical properties, new property generation)
    assert [n.due for n in notices] == [False, False, True, False, False, True, False] and [n.action for n in notices if n.due] == ["saved", "saved"]  # only the closed days 3 and 6, none for the storms
    assert [n.day for n in notices] == [1, 2, 3, 4, 5, 6, 7]
    assert gens[0] < gens[2] < gens[5] and columns[0] != orig_sha and columns[0] == columns[2] == columns[5]  # generations advance across the events; the first (changed) refresh changed the column, the later no-op refreshes did not


def test_a_changed_case_is_still_refused_and_the_old_and_new_profiles_do_not_mix(tmp_path):
    lin, b, factory = dry_boundary(tmp_path / "a", "numpy", False)
    (tmp_path / "out").mkdir()
    origin = br.open_day(lin, b, partition=PART, run_digest=RUN, available_bytes_fn=AVAIL)
    out = fo.FloatingMoistureOutput.create(tmp_path / "out", origin, expected_parent=fd.expect_parent(origin), every_n_days=1, limits=LIM, host_available_bytes_fn=big, disk_available_bytes_fn=big)
    st, res = one_day(lin, b, 0, out, st=origin)
    assert res.notice.action == "saved"
    boundary = br.hand_back(lin, res.state)
    # (1) another evaporation partition = another case (a separate lineage opened with it): refused by the writer, the cadence untouched
    other = se.EvaporationPartition(1.3, 0.5, 1.0, 0.125, 0.75)
    lin_o, b_o, _f = dry_boundary(tmp_path / "other", "numpy", False)
    st_o = br.open_day(lin_o, b_o, partition=other, run_digest=RUN, available_bytes_fn=AVAIL)
    n = st_o.ctx.n_cells
    r_o = br.day_action(st_o, plan=rc.plan_daily_rain(0.0, 0.0, 5.0, run_digest=RUN, ordinal=0), pet_terrain_m_s=np.full(n, 3e-8), cover=np.full(n, 0.3), available_bytes_fn=AVAIL)
    t_o = fd.request_floating_withdrawal(r_o.state, expected_parent=fd.expect_parent(r_o.state), request_m=np.zeros((st_o.ctx.n_layers, n)), expected_sequence=0, available_bytes_fn=AVAIL)
    with pytest.raises(g0.MoistureOutputError):
        fd.close_floating_day(t_o.state, expected_parent=fd.expect_parent(t_o.state), output=out, host_available_bytes_fn=big, disk_available_bytes_fn=big)
    # (2) the historical fixed (generation-0) profile does not mix with the bridge profile in either direction
    import builders as bld
    from dbuild import lineage as old_lineage

    old_state, *_ = old_lineage()
    (tmp_path / "out_old").mkdir()
    old_out = fo.FloatingMoistureOutput.create(tmp_path / "out_old", old_state, expected_parent=fd.expect_parent(old_state), every_n_days=1, limits=LIM, host_available_bytes_fn=big, disk_available_bytes_fn=big)
    with pytest.raises(g0.MoistureOutputError):
        fo.FloatingMoistureOutput.resume(tmp_path / "out_old", origin, expected_parent=fd.expect_parent(origin), limits=LIM, host_available_bytes_fn=big)  # bridge state against an old-profile record
    with pytest.raises(g0.MoistureOutputError):
        fo.FloatingMoistureOutput.resume(tmp_path / "out", old_state, expected_parent=fd.expect_parent(old_state), limits=LIM, host_available_bytes_fn=big)  # old state against a bridge record
    assert bld is not None and old_out is not None
    # (3) the legitimate day with the SAME case still works after the refused one (retry: cadence untouched)
    st2, res2 = one_day(lin, boundary, 1, out)
    assert res2.notice.day == 2 and res2.notice.action == "saved"


def test_resume_cadence_continues_from_the_accepted_closed_state(tmp_path):
    lin, b, factory = dry_boundary(tmp_path / "a", "numpy", False)
    (tmp_path / "out").mkdir()
    origin = br.open_day(lin, b, partition=PART, run_digest=RUN, available_bytes_fn=AVAIL)
    out = fo.FloatingMoistureOutput.create(tmp_path / "out", origin, expected_parent=fd.expect_parent(origin), every_n_days=1, limits=LIM, host_available_bytes_fn=big, disk_available_bytes_fn=big)
    st, res = one_day(lin, b, 0, out, st=origin)
    closed1 = res.state
    boundary = br.hand_back(lin, closed1)
    resumed = fo.FloatingMoistureOutput.resume(tmp_path / "out", closed1, expected_parent=fd.expect_parent(closed1), limits=LIM, host_available_bytes_fn=big)  # a fresh writer from the ACCEPTED closed state
    assert resumed.last_day == 1 and resumed.every_n_days == 1
    st2, res2 = one_day(lin, boundary, 1, resumed)
    assert res2.notice.day == 2 and res2.notice.action == "saved"
