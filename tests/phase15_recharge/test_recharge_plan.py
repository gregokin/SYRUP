"""Phase 1.5 S1: rainfall descriptor validation, branch selection, full-storm forcing for EVERY positive P and the bounded occurrence cursor."""
from __future__ import annotations

import dataclasses
import math
import sys

import numpy as np
import pytest

from maple_syrup.rainfall import constant_rainfall
from maple_syrup.rainfall import RainfallSchedule
from maple_syrup.recharge import BoundForcing, DailyCursor, PendingStorm, RechargeError, plan_daily_rain, verify_forcing, verify_plan

RUN = "ab" * 32
GOOD = [(0.0, 0.0, 5.0, "no_rain"), (0.0, 10.0, 5.0, "no_rain"), (2.0, 10.0, 5.0, "recharge"), (5.0, 10.0, 5.0, "recharge"),
        (5.000001, 10.0, 5.0, "storm"), (30.0, 3.0, 5.0, "storm"), (1e-3, 20.0, 0.0, "storm"), (0.0, 0.0, 0.0, "no_rain")]


@pytest.mark.parametrize("p,i,t,selection", GOOD)
def test_selection_table_including_equality_and_no_defaults(p, i, t, selection):
    plan = plan_daily_rain(p, i, t)
    assert plan.selection == selection
    assert plan.depth_m == p / 1000.0


def test_equality_attempts_recharge_and_zero_threshold_sends_all_positive_rain_to_the_storm():
    assert plan_daily_rain(5.0, 7.0, 5.0).selection == "recharge"
    assert plan_daily_rain(1e-9, 7.0, 0.0).selection == "storm"
    assert plan_daily_rain(0.0, 0.0, 0.0).selection == "no_rain"


@pytest.mark.parametrize("p,i,t", [(float("nan"), 1.0, 1.0), (1.0, float("inf"), 1.0), (1.0, 1.0, float("nan")), (-1.0, 1.0, 1.0), (1.0, 1.0, -1.0),
                                   (1.0, 0.0, 1.0), (1.0, -3.0, 1.0), (True, 1.0, 1.0), ("1", 1.0, 1.0), (1.0, None, 1.0), (np.array([1.0]), 1.0, 1.0)])
def test_invalid_descriptors_are_refused_before_selection(p, i, t):
    with pytest.raises(RechargeError):
        plan_daily_rain(p, i, t)


def test_positive_rain_with_underflowing_or_overflowing_duration_or_depth_is_refused_not_dropped():
    with pytest.raises(RechargeError):
        plan_daily_rain(5e-324, 1e308, 1.0)  # duration underflows to 0
    with pytest.raises(RechargeError):
        plan_daily_rain(1e-300, 1e300, 1.0)  # 3600 P / I underflows to a denormal/zero
    with pytest.raises(RechargeError):
        plan_daily_rain(1e308, 1e-300, 1e309 if False else 1.0)  # overflow
    ok = plan_daily_rain(1e-12, 1e3, 1.0)  # tiny but representable rain is kept
    assert ok.duration_s > 0.0 and ok.selection == "recharge"


@pytest.mark.parametrize("p,i,t", [(2.0, 10.0, 5.0), (30.0, 3.0, 5.0), (5.0, 10.0, 5.0)])
def test_full_forcing_exists_for_every_positive_p_including_the_recharge_selection(p, i, t):
    plan = plan_daily_rain(p, i, t)
    assert plan.duration_s == pytest.approx(3600.0 * p / i, rel=1e-15)
    f = plan.bind_forcing(86400.0)
    ref = constant_rainfall(0.0, plan.duration_s, i)  # the actual rainfall API; LOCAL edges [0, D] (the native event adds its origin)
    assert list(f.schedule.edges_s) == list(ref.edges_s) == [0.0, plan.duration_s] and list(f.schedule.intensity_mm_per_h) == list(ref.intensity_mm_per_h)
    assert f.schedule.depth_m(0.0, plan.duration_s) * 1000.0 == pytest.approx(p, rel=1e-12)  # FULL P is preserved
    assert f.p_mm == p and f.t0_s == 86400.0 and f.end_s == 86400.0 + plan.duration_s and len(f.digest) == 64  # physical start/end are explicit absolutes


def test_no_rain_has_no_forcing_and_clock_rounding_never_loses_rain():
    assert plan_daily_rain(0.0, 0.0, 5.0).bind_forcing(0.0) is None
    plan = plan_daily_rain(1e-6, 100.0, 5.0)  # a 3.6e-5 s storm
    with pytest.raises(RechargeError):
        plan.bind_forcing(1e18)  # t0 + duration is not representable at this clock
    with pytest.raises(RechargeError):
        plan.bind_forcing(1e12)  # the MAPLE FP64 clock bound at this clock (1.8e-3 s) exceeds the whole 3.6e-5 s storm: unresolvable, refused
    assert 1e11 + plan.duration_s > 1e11  # the end IS representable here, so it is the declared clock bound (not the end == t0 test) that refuses
    with pytest.raises(RechargeError, match="declared MAPLE FP64 operation bound"):
        plan.bind_forcing(1e11)
    ok = plan.bind_forcing(1e9)  # resolvable: accepted, with the REALIZED native amount and the declared bound recorded, never hidden
    assert abs(ok.amount_mm - 1e-6) <= ok.amount_bound_mm and ok.amount_bound_mm > 0.0
    with pytest.raises(RechargeError):
        plan.bind_forcing(float("nan"))
    with pytest.raises(RechargeError):
        plan.bind_forcing(-1.0)
    assert plan.bind_forcing(0.0).end_s == plan.duration_s


def test_plan_digest_binds_the_occurrence_and_pairs_run_with_ordinal():
    a, b = plan_daily_rain(2.0, 10.0, 5.0, run_digest=RUN, ordinal=3), plan_daily_rain(2.0, 10.0, 5.0, run_digest=RUN, ordinal=4)
    assert a.digest != b.digest and a.digest == plan_daily_rain(2.0, 10.0, 5.0, run_digest=RUN, ordinal=3).digest
    for kw in ({"run_digest": RUN}, {"ordinal": 1}, {"run_digest": "x", "ordinal": 1}, {"run_digest": RUN, "ordinal": -1}, {"run_digest": RUN, "ordinal": True}):
        with pytest.raises(RechargeError):
            plan_daily_rain(2.0, 10.0, 5.0, **kw)


def test_cursor_consumes_every_branch_once_and_refuses_gaps_repeats_and_other_runs():
    cur = DailyCursor(RUN)
    parent = "cd" * 32
    plans = [plan_daily_rain(p, 10.0, 5.0, run_digest=RUN, ordinal=k) for k, p in enumerate([0.0, 2.0, 5.0])]
    for plan, branch in zip(plans, ("no_rain", "recharge", "recharge")):
        cur = cur.consume(plan, branch, parent)
    assert cur.ordinal == 3 and cur.last_branch == "recharge" and cur.pending is None
    with pytest.raises(RechargeError):
        cur.consume(plans[2], "recharge", parent)  # repeat
    with pytest.raises(RechargeError):
        cur.consume(plan_daily_rain(1.0, 10.0, 5.0, run_digest=RUN, ordinal=5), "recharge", parent)  # gap
    with pytest.raises(RechargeError):
        cur.consume(plan_daily_rain(1.0, 10.0, 5.0, run_digest="ee" * 32, ordinal=3), "recharge", parent)  # other run
    with pytest.raises(RechargeError):
        cur.consume(plan_daily_rain(1.0, 10.0, 5.0, run_digest=RUN, ordinal=3), "bogus", parent)


def test_cursor_storm_and_fallback_pending_state():
    cur = DailyCursor(RUN)
    storm = plan_daily_rain(30.0, 3.0, 5.0, run_digest=RUN, ordinal=0)
    f = storm.bind_forcing(0.0)
    pend = cur.begin_storm(storm, f, fallback=False, t0_s=0.0)
    assert pend.pending.branch == "storm" and pend.ordinal == 0  # not consumed until genuine completion
    with pytest.raises(RechargeError):
        pend.check(storm)  # another occurrence while pending
    other = plan_daily_rain(30.0, 3.0, 5.0, run_digest=RUN, ordinal=0, label="other")
    with pytest.raises(RechargeError):
        pend.consume(other, "storm", "cd" * 32)  # not the pending plan
    done = pend.consume(storm, "storm", "cd" * 32)
    assert done.ordinal == 1 and done.pending is None
    rech = plan_daily_rain(2.0, 10.0, 5.0, run_digest=RUN, ordinal=1)
    fb = done.begin_storm(rech, rech.bind_forcing(10.0), fallback=True, t0_s=10.0)
    assert fb.pending.branch == "storm_fallback"
    assert fb.consume(rech, "storm_fallback", "cd" * 32).ordinal == 2
    with pytest.raises(RechargeError):
        done.begin_storm(rech, rech.bind_forcing(10.0), fallback=False, t0_s=10.0)  # a recharge selection starts a storm only as a fallback
    with pytest.raises(RechargeError):
        done.begin_storm(rech, storm.bind_forcing(0.0), fallback=True, t0_s=0.0)  # forcing substitution
    with pytest.raises(RechargeError):
        done.begin_storm(rech, rech.bind_forcing(10.0), fallback=True, t0_s=11.0)  # not the actual physical start
    for bad in (1, 0, None, "x", np.bool_(True)):
        with pytest.raises(RechargeError):
            done.begin_storm(rech, rech.bind_forcing(10.0), fallback=bad, t0_s=10.0)  # fallback must be a real bool
    with pytest.raises(RechargeError):
        done.begin_storm(plan_daily_rain(0.0, 0.0, 5.0, run_digest=RUN, ordinal=1), None, fallback=False, t0_s=0.0)


PARENT = "cd" * 32


def _forged(plan, **kw):
    """A valid-looking record built directly from the frozen dataclass with altered fields (the digest kept unless replaced)."""
    return dataclasses.replace(plan, **kw)


@pytest.mark.parametrize("kw", [{"depth_m": 0.5}, {"selection": "storm"}, {"p_mm": 3.0}, {"i_mm_h": 20.0}, {"t_mm": 9.0}, {"reason": "x"},
                                {"duration_s": 1.0}, {"digest": "ee" * 32}, {"label": "other"}, {"ordinal": 1}, {"run_digest": "ee" * 32}])
def test_forged_plan_fields_are_refused_by_every_consuming_boundary(kw):
    plan = plan_daily_rain(2.0, 10.0, 5.0, run_digest=RUN, ordinal=0)
    bad = _forged(plan, **kw)
    cur = DailyCursor(RUN)
    with pytest.raises(RechargeError):
        verify_plan(bad)
    with pytest.raises(RechargeError):
        cur.check(bad)
    with pytest.raises(RechargeError):
        cur.consume(bad, "recharge", PARENT)
    with pytest.raises(RechargeError):
        bad.bind_forcing(0.0)
    with pytest.raises(RechargeError):
        cur.begin_storm(bad, plan.bind_forcing(0.0), fallback=True, t0_s=0.0)


def test_forged_plan_with_recomputed_looking_digest_for_other_values_is_still_refused():
    real = plan_daily_rain(2.0, 10.0, 5.0, run_digest=RUN, ordinal=0)
    other = plan_daily_rain(6.0, 10.0, 5.0, run_digest=RUN, ordinal=0)  # a genuine STORM plan for another P (T = 5), with ITS digest
    mixed = _forged(real, p_mm=other.p_mm, depth_m=other.depth_m, duration_s=other.duration_s, digest=other.digest)  # keeps the 'recharge' selection
    verify_plan(other)
    with pytest.raises(RechargeError):
        verify_plan(mixed)  # valid-looking digest/depth/duration of the other P but the wrong selection: not canonical


def test_plan_types_and_float_types_are_exact():
    plan = plan_daily_rain(2.0, 10.0, 5.0, run_digest=RUN, ordinal=0)
    with pytest.raises(RechargeError):
        verify_plan(_forged(plan, p_mm=np.float64(2.0)))
    with pytest.raises(RechargeError):
        verify_plan(object())


def _scaled(forcing, factor, **kw):
    sch = forcing.schedule
    new = RainfallSchedule(edges_s=sch.edges_s, intensity_mm_per_h=sch.intensity_mm_per_h * factor, provenance=sch.provenance)
    return dataclasses.replace(forcing, schedule=new, **kw)


def test_forged_forcing_is_refused_with_the_old_plan_and_forcing_digests_kept():
    storm = plan_daily_rain(30.0, 3.0, 5.0, run_digest=RUN, ordinal=0)
    f = storm.bind_forcing(0.0)
    verify_forcing(storm, f)
    cur = DailyCursor(RUN)
    for bad in (_scaled(f, 2.0), _scaled(f, 0.5), dataclasses.replace(f, p_mm=31.0), dataclasses.replace(f, intensity_mm_h=4.0),
                dataclasses.replace(f, t0_s=5.0), dataclasses.replace(f, end_s=f.end_s + 1.0), dataclasses.replace(f, digest="ee" * 32),
                dataclasses.replace(f, plan_digest="ee" * 32), dataclasses.replace(f, amount_mm=f.amount_mm * 2.0),
                dataclasses.replace(f, schedule=None), "not a forcing"):
        with pytest.raises(RechargeError):
            verify_forcing(storm, bad)
        with pytest.raises(RechargeError):
            cur.begin_storm(storm, bad, fallback=False, t0_s=0.0)
    # the schedule digest is also recomputed from the actual arrays: a schedule replaced with a DIFFERENT digest pair is refused too
    with pytest.raises(RechargeError):
        verify_forcing(storm, _scaled(f, 2.0, digest=storm.bind_forcing(0.0).digest))


def test_forcing_scalar_fields_and_native_amount_are_recorded_and_within_the_declared_bound():
    f = plan_daily_rain(2.3, 7.1, 5.0).bind_forcing(86400.0)
    assert abs(f.amount_mm - 2.3) <= f.amount_bound_mm and 0.0 < f.amount_bound_mm < 2.3e-9


def test_consume_requires_the_matching_branch_and_pending_state():
    cur = DailyCursor(RUN)
    rech = plan_daily_rain(2.0, 10.0, 5.0, run_digest=RUN, ordinal=0)
    storm = plan_daily_rain(30.0, 3.0, 5.0, run_digest=RUN, ordinal=0)
    none = plan_daily_rain(0.0, 0.0, 5.0, run_digest=RUN, ordinal=0)
    for plan, ok, bad in ((rech, "recharge", ("no_rain", "storm", "storm_fallback")), (none, "no_rain", ("recharge", "storm", "storm_fallback")),
                          (storm, None, ("no_rain", "recharge", "storm", "storm_fallback"))):
        for b in bad:
            with pytest.raises(RechargeError):
                cur.consume(plan, b, PARENT)  # wrong branch / a storm branch with nothing pending
        if ok:
            assert cur.consume(plan, ok, PARENT).ordinal == 1
    pend = cur.begin_storm(storm, storm.bind_forcing(0.0), fallback=False, t0_s=0.0)
    for b in ("no_rain", "recharge", "storm_fallback"):
        with pytest.raises(RechargeError):
            pend.consume(storm, b, PARENT)  # wrong branch while a storm is pending
    with pytest.raises(RechargeError):
        pend.consume(none, "no_rain", PARENT)  # a non-storm occurrence while pending
    with pytest.raises(RechargeError):
        pend.consume(plan_daily_rain(30.0, 3.0, 5.0, run_digest="ee" * 32, ordinal=0), "storm", PARENT)  # wrong run
    fb = cur.begin_storm(rech, rech.bind_forcing(0.0), fallback=True, t0_s=0.0)
    with pytest.raises(RechargeError):
        fb.consume(rech, "recharge", PARENT)  # a fallback occurrence cannot be consumed as the recharge it failed
    with pytest.raises(RechargeError):
        fb.consume(rech, "storm", PARENT)  # nor as a plain storm
    assert fb.consume(rech, "storm_fallback", PARENT).last_branch == "storm_fallback"
    with pytest.raises(RechargeError):
        pend.consume(storm, "storm", "not-a-digest")


def test_cursor_field_coherence_is_enforced_at_construction():
    plan_d = "ab" * 32
    with pytest.raises(RechargeError):
        DailyCursor(RUN, 3)  # consumed occurrences must record their last occurrence
    with pytest.raises(RechargeError):
        DailyCursor(RUN, 0, None, plan_d, "recharge", PARENT)  # ordinal 0 records none
    with pytest.raises(RechargeError):
        DailyCursor(RUN, 1, None, plan_d, None, PARENT)  # partial last-field group
    with pytest.raises(RechargeError):
        DailyCursor(RUN, 1, None, plan_d, "recharge", None)
    with pytest.raises(RechargeError):
        DailyCursor(RUN, 1, PendingStorm(2, plan_d, "storm", plan_d), plan_d, "recharge", PARENT)  # pending ordinal != next ordinal
    with pytest.raises(RechargeError):
        DailyCursor(RUN, 0, PendingStorm(3, plan_d, "storm", plan_d))
    assert DailyCursor(RUN, 1, PendingStorm(1, plan_d, "storm", plan_d), plan_d, "recharge", PARENT).pending.ordinal == 1


def test_cursor_size_is_bounded_over_many_occurrences():
    cur = DailyCursor(RUN)
    sizes = set()
    for k in range(10_000):
        plan = plan_daily_rain(float(k % 3), 10.0, 5.0, run_digest=RUN, ordinal=k)
        cur = cur.consume(plan, "no_rain" if k % 3 == 0 else "recharge", "cd" * 32)
        if k % 2500 == 0:
            sizes.add(len(repr(cur)))
    assert cur.ordinal == 10_000 and max(sizes) - min(sizes) <= 8  # scalars only: no growing set/history
    assert len(cur.digest()) == 64
