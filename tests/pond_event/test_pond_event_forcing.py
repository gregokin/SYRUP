"""Forcing split, bounded adaptive retry and incompleteness of the pond event. Nothing here was run by its author."""
import numpy as np
import pytest
from pond_helpers import (
    DX,
    arrays_of,
    assert_unchanged,
    basin,
    east_outlets,
    host,
    initial,
    make_bed,
    make_event,
    make_topology,
    plane,
    wet_plane,
)

from maple_syrup.experimental_storm import ExperimentalControl
from maple_syrup.pond_event import PondEventControl
from maple_syrup.rainfall import RainfallProvenance, RainfallSchedule


def knotted():
    return RainfallSchedule(edges_s=[0.0, 3.3, 7.1, 9.0], intensity_mm_per_h=[60.0, 0.0, 120.0],
                            provenance=RainfallProvenance(kind="constant"))


def test_rain_knots_split_exactly_and_the_ledger_is_the_exact_integral(cpu):
    bed, ctx = make_bed(plane(), depth=0.0)
    topo = make_topology(bed, east_outlets(3, 4), drop=0.02)
    event = make_event(bed, ctx, topo, schedule=knotted(), max_dt_s=0.5)
    state = initial(event, bed)
    result = event.evolve(state, 12.0)
    assert result.reached_target and result.stop_reason is None and result.state.t_s == 12.0
    water = event.audit(result.state)["water"]
    expected = 12 * DX * DX * knotted().depth_m(0.0, 12.0)  # exact piecewise integral
    # dt = 0.5 does not divide 3.3 / 7.1 / 9.0: a step straddling a knot would be off by far more than round-off
    assert water["rain_m3"] == pytest.approx(expected, rel=1e-12)
    assert water["closes"]
    # the first call to a knot lands exactly on it (snapped time), never past it
    mid = event.evolve(state, 3.3)
    assert mid.state.t_s == 3.3 and event.audit(mid.state)["water"]["rain_m3"] == pytest.approx(
        12 * DX * DX * knotted().depth_m(0.0, 3.3), rel=1e-12)


def test_recession_after_the_last_knot_applies_no_rain(cpu):
    bed, ctx = make_bed(plane(), depth=0.002)
    topo = make_topology(bed, east_outlets(3, 4), drop=0.02)
    event = make_event(bed, ctx, topo, schedule=knotted(), max_dt_s=0.5)
    state = event.evolve(initial(event, bed), 9.0).state
    rain_at_end = event.audit(state)["water"]["rain_m3"]
    later = event.evolve(state, 14.0).state
    audit = event.audit(later)
    assert audit["water"]["rain_m3"] == rain_at_end and audit["water"]["closes"]
    assert audit["water"]["export_m3"] >= event.audit(state)["water"]["export_m3"]


def deep_basin(max_dt, **kw):
    bed, ctx = make_bed(basin(), depth=np.maximum(0.225 - basin(), 0.0))
    topo = make_topology(bed, [(1, 4, 0, 1)])
    ctl = PondEventControl(time=ExperimentalControl(max_dt_s=max_dt, min_dt_s=1.0 / 1024.0, **kw))
    event = make_event(bed, ctx, topo, control=ctl)
    return event, initial(event, bed)


def test_hydraulic_rejection_halves_dt_boundedly_and_the_cap_is_carried():
    event, state = deep_basin(8.0, max_retries=12)
    result = event.evolve(state, 8.0)
    assert result.n_rejected_attempts > 0 and result.rejections[0]["reason"]
    assert result.reached_target and result.state.t_s == 8.0
    assert result.max_accepted_dt_s < 8.0  # shrunk to what survived the CFL
    cap = result.state.hydraulic.next_dt_cap_s
    assert cap is not None and 0.0 < cap <= 8.0
    audit = event.audit(result.state)
    assert audit["water"]["closes"] and audit["water"]["export_m3"] > 0.0  # the 5 mm over the crest overflowed


def test_retry_bound_stops_incomplete_and_never_continues_silently():
    event, state = deep_basin(8.0, max_retries=1)  # the native validator requires >= 1
    before = arrays_of(state)
    result = event.evolve(state, 8.0)
    assert not result.reached_target and result.stop_reason == "max_retries"
    assert result.n_rejected_attempts == 2  # 8 s and 4 s are both rejected by the CFL, the bound stops the call
    assert result.n_accepted_steps == 0 and result.state.t_s == 0.0 and result.elapsed_s == 0.0
    assert_unchanged(before, state)
    capped = deep_basin(0.2, max_retries=2, max_steps=3)
    stopped = capped[0].evolve(capped[1], 8.0)
    assert stopped.stop_reason == "max_steps" and stopped.n_accepted_steps == 3 and not stopped.reached_target
    assert stopped.state.t_s < 8.0 and stopped.state.t_s == pytest.approx(0.6)


def test_transport_rejection_is_retried_then_reported_not_hidden(cpu):
    bed, ctx = make_bed(plane(), depth=0.003)
    topo = make_topology(bed, east_outlets(3, 4), drop=0.02)
    from maple_syrup.rainfall import constant_rainfall

    ctl = PondEventControl(time=ExperimentalControl(max_dt_s=0.2, min_dt_s=1.0 / 1024.0, max_retries=3),
                           sediment_courant_max=1e-6, max_transport_substeps=1)
    event = make_event(bed, ctx, topo, control=ctl, schedule=constant_rainfall(0.0, 100.0, 100.0))
    state = initial(event, bed)
    before = arrays_of(state)
    result = event.evolve(state, 5.0)
    # the diffuse law gives the flowing shallow layer a positive sediment velocity, so n_substeps > 1 is required
    assert result.n_rejected_attempts == 4 and result.stop_reason == "max_retries"
    assert "substeps" in result.rejections[0]["reason"]
    assert not result.reached_target and result.state.t_s == 0.0
    assert_unchanged(before, state)


def test_end_time_must_advance(cpu):
    event, state = wet_plane(cpu)
    with pytest.raises(Exception, match="end_s"):
        event.evolve(state, 0.0)
    assert host(state.hydraulic.depth_m).shape == (3, 4)
