# ruff: noqa: RUF059
"""Correction03: the pending storm needs the PRODUCER's proof of the forcing HISTORY (every accepted step integrated the native schedule's rate on a piece-confined interval), not a total. Root's same-total wrong-history
repro, off-piece steps, manually stepped but conforming steps, unproven (restored/foreign) chains, the zero-rain R0 carry, and the native-direct equivalence of the capacity fallback."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from bridge_builders import AVAIL, RUN, complete, dry_boundary

from maple_syrup import floating_event_daily_bridge as br
from maple_syrup import floating_event_lifecycle as lc
from maple_syrup import recharge as rc
from maple_syrup import soil_evaporation as se
from maple_syrup.rainfall import constant_rainfall

PART = se.EvaporationPartition(1.25, 0.5, 1.0, 0.125, 0.75)
PET, COV = 3e-8, 0.3


def pending_boundary(tmp, **kw):
    """A dry genuine boundary, then the day's plan P = 0.01 mm at 3600 mm/h (T = 0.005 mm): the FULL native storm, pending on the cursor with its bound forcing (10 s)."""
    lin, b, factory = dry_boundary(tmp, "numpy", False, **kw)
    st = br.open_day(lin, b, partition=PART, run_digest=RUN, available_bytes_fn=AVAIL)
    n = st.ctx.n_cells
    pl = rc.plan_daily_rain(0.01, 3600.0, 0.005, run_digest=RUN, ordinal=0)
    b3 = br.begin_storm_from_day(lin, st, pl, pet_terrain_m_s=np.full(n, PET), cover=np.full(n, COV), available_bytes_fn=AVAIL)
    return lin, b3, factory, pl, n


def begin_matched(lin, b3, factory):
    f = b3.pending_forcing
    occ, s = lc.begin_occurrence(lin, b3, lambda vert, topology: factory(vert, topology, br.storm_schedule(f)), available_bytes_fn=AVAIL)
    return occ, s, f


def close(lin, occ, h):
    return lc.close_occurrence(lin, occ, occ.event.completion_capability(h, h.state), h.state, available_bytes_fn=AVAIL)


def test_canonical_native_evolve_control_passes_and_records_the_match(tmp_path):
    lin, b3, factory, pl, n = pending_boundary(tmp_path / "a")
    occ, s, f = begin_matched(lin, b3, factory)
    assert occ.event.forcing_conformity(s) is True  # nothing integrated yet
    h = complete(occ.event, s)
    assert occ.event.forcing_conformity(h.state) is True  # the receipt survives evolve's copies/caps/snaps, the terminal settlement and the recession
    b4 = close(lin, occ, h)
    assert b4.storm_match == f.digest and lin.head is b4


def test_same_total_wrong_history_is_refused_with_head_and_capability_unchanged_and_the_canonical_control_still_passes(tmp_path):
    """Root's reproducer: the declared constant 3600 mm/h forcing, but the occurrence is manually stepped at DOUBLE the rate for the first half and ZERO for the second half: the same 3e-5 m3 total."""
    lin, b3, factory, pl, n = pending_boundary(tmp_path / "a")
    occ, s, f = begin_matched(lin, b3, factory)
    rate = occ.event.schedule.rate_after_m_per_s(float(s.t_s))
    half = (f.end_s - f.t0_s) / 2.0
    s1 = occ.event.step(s, 2.0 * rate, half).state  # manual public step: twice the native rate
    s2 = occ.event.step(s1, 0.0, half).state  # then zero: same integral
    assert occ.event.forcing_conformity(s2) is False
    h = complete(occ.event, s2)  # a later CORRECT native evolve + genuine completion cannot make the chain canonical
    assert occ.event.forcing_conformity(h.state) is False
    cap = occ.event.completion_capability(h, h.state)
    with pytest.raises(lc.LifecycleError, match="producer proof of its forcing history"):
        lc.close_occurrence(lin, occ, cap, h.state, available_bytes_fn=AVAIL)
    assert lin.head is occ and h not in occ.event._completions_consumed  # head unchanged, capability unconsumed (retryable)
    # the canonical native control (same plan/forcing, own lineage) passes
    lin2, b3c, factory2, pl2, n2 = pending_boundary(tmp_path / "b")
    occ2, s_c, f2 = begin_matched(lin2, b3c, factory2)
    assert close(lin2, occ2, complete(occ2.event, s_c)).storm_match == f2.digest


def test_off_piece_manual_step_is_refused_but_a_manual_step_that_matches_the_schedule_per_step_is_valid(tmp_path):
    lin, b3, factory, pl, n = pending_boundary(tmp_path / "a")
    occ, s, f = begin_matched(lin, b3, factory)
    rate = occ.event.schedule.rate_after_m_per_s(float(s.t_s))
    dur = f.end_s - f.t0_s
    s_a = occ.event.step(s, rate, 0.8 * dur).state  # a conforming step up to 20% of the duration before the rainfall end
    assert occ.event.forcing_conformity(s_a) is True
    s_off = occ.event.step(s_a, rate, 0.4 * dur).state  # the right rate but the interval crosses the rainfall END (rain integrated past the piece)
    assert occ.event.forcing_conformity(s_off) is False
    # manually stepped, but each step IS the native rate inside ONE piece, snapped to the edge: valid per step
    lin2, b3v, factory2, pl2, n2 = pending_boundary(tmp_path / "b")
    occ2, s2, f2 = begin_matched(lin2, b3v, factory2)
    half = (f2.end_s - f2.t0_s) / 2.0
    t = float(s2.t_s)
    r1 = occ2.event.schedule.rate_after_m_per_s(t)
    st1 = occ2.event.step(s2, r1, half).state
    t1 = float(st1.t_s)
    r2 = occ2.event.schedule.rate_after_m_per_s(t1)
    st2 = occ2.event.step(st1, r2, f2.end_s - t1, t_end_s=f2.end_s).state
    assert occ2.event.forcing_conformity(st2) is True
    assert close(lin2, occ2, complete(occ2.event, st2)).storm_match == f2.digest


def test_an_unproven_chain_has_no_proof_from_the_total_alone(tmp_path):
    lin, b3, factory, pl, n = pending_boundary(tmp_path / "a")
    occ, s, f = begin_matched(lin, b3, factory)
    mid = occ.event.evolve(s, float(f.end_s) + 1.0).state
    assert occ.event.forcing_conformity(mid) is True
    foreign = replace(mid)  # a state NOT produced by the producer's step chain (as a restored state would be)
    assert occ.event.forcing_conformity(foreign) is None
    h = complete(occ.event, foreign)
    assert occ.event.forcing_conformity(h.state) is None
    with pytest.raises(lc.LifecycleError, match="unproven"):
        close(lin, occ, h)
    assert lin.head is occ


def test_zero_rain_recertification_after_a_matched_storm_carries_and_a_manual_positive_step_in_it_is_refused(tmp_path):
    lin, b3, factory, pl, n = pending_boundary(tmp_path / "a")
    occ, s, f = begin_matched(lin, b3, factory)
    b4 = close(lin, occ, complete(occ.event, s))
    assert b4.storm_match == f.digest and b4.cursor.pending is not None
    zero = constant_rainfall(b4.t_s, b4.t_s + 5.0, 0.0)
    occ_r, s_r = lc.begin_occurrence(lin, b4, lambda vert, topology: factory(vert, topology, zero), available_bytes_fn=AVAIL)  # legitimate rain-0 recertification
    h_r = complete(occ_r.event, s_r)
    assert occ_r.event.forcing_conformity(h_r.state) is True
    b5 = close(lin, occ_r, h_r)
    assert b5.storm_match == f.digest and lin.head is b5  # the match carries through the R0 occurrence
    # a manual POSITIVE-rate step inside a zero-schedule occurrence is non-conforming and refused
    occ_x, s_x = lc.begin_occurrence(lin, b5, lambda vert, topology: factory(vert, topology, zero), available_bytes_fn=AVAIL)
    bad = occ_x.event.step(s_x, 1e-6, 1.0).state
    assert occ_x.event.forcing_conformity(bad) is False
    h_x = complete(occ_x.event, bad)
    with pytest.raises(lc.LifecycleError, match="producer proof"):
        close(lin, occ_x, h_x)
    assert lin.head is occ_x


def test_closed_capacity_fallback_is_native_direct_equivalent_for_the_same_original_state_and_forcing(tmp_path):
    kw = {"bottom": "closed", "saturation": 0.999999, "rain_mm_h": 0.0, "end_s": 2.0}
    lin, b, factory = dry_boundary(tmp_path / "a", "numpy", False, **kw)
    st = br.open_day(lin, b, partition=PART, run_digest=RUN, available_bytes_fn=AVAIL)
    n = st.ctx.n_cells
    pl = rc.plan_daily_rain(0.01, 3600.0, 5.0, run_digest=RUN, ordinal=0)  # P = 0.01 mm <= T: a recharge plan whose pulse cannot fit the near-saturated CLOSED profile
    assert pl.selection == "recharge"
    res = br.day_action(st, plan=pl, pet_terrain_m_s=np.full(n, PET), cover=np.full(n, COV), available_bytes_fn=AVAIL)
    assert res.outcome == "requires_full_storm" and res.branch == "storm_fallback" and res.capacity_failure
    w_parent = np.array(st.water.water_m, copy=True)
    b3 = br.begin_storm_from_day(lin, st, pl, pet_terrain_m_s=np.full(n, PET), cover=np.full(n, COV), available_bytes_fn=AVAIL)
    f = b3.pending_forcing
    assert f.p_mm == 0.01 and np.array_equal(np.asarray(b3.root.water_state.water_m), w_parent)  # FULL P at the UNCHANGED parent
    occ, s, _f = begin_matched(lin, b3, factory)
    end = float(f.end_s) + 30.0
    a = occ.event.evolve(s, end).state  # the genuine native storm over its rainfall and 30 s of recession (a closed saturated plane film never quiets: the equivalence is compared at the same native instant)
    assert occ.event.forcing_conformity(a) is True
    # native-direct control: an independent lineage with the SAME original state runs the SAME forcing through the ordinary lifecycle (no daily plumbing)
    lin2, b2, factory2 = dry_boundary(tmp_path / "b", "numpy", False, **kw)
    occ2, s2 = lc.begin_occurrence(lin2, b2, lambda vert, topology: factory2(vert, topology, br.storm_schedule(f)), available_bytes_fn=AVAIL)
    d = occ2.event.evolve(s2, end).state
    pa, pd = occ.vertical.current_payload(a.hydraulic.soil_water_m), occ2.vertical.current_payload(d.hydraulic.soil_water_m)
    assert np.array_equal(np.asarray(pa.state.water_m), np.asarray(pd.state.water_m)) and np.array_equal(np.asarray(pa.history.wetting_m), np.asarray(pd.history.wetting_m))
    assert np.array_equal(np.asarray(a.hydraulic.depth_m), np.asarray(d.hydraulic.depth_m)) and float(a.t_s) == float(d.t_s) and float(f.p_mm) == 0.01
