"""Logic tests of the stage-1 importer on synthetic series (no data files): QA masks, END matching, rain-free runs, exact profile integration, complete-day air aggregation, hashes. Synthetic values are LOGIC fixtures only."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarks.subsurface_validation import neon_inputs as ni

T0 = ni.parse_utc("2025-04-01T00:00:00Z")


def series(values, *, qf=None, sci=None, vmin=None, vmax=None, numpts=None, name="s", t0=T0):
    n = len(values)
    start = t0 + ni.HALF_HOUR_S * np.arange(n, dtype=np.int64)
    qf = np.zeros(n, dtype=np.int64) if qf is None else np.asarray(qf, dtype=np.int64)
    sci_ok = np.ones(n, dtype=bool) if sci is None else np.asarray(sci, dtype=bool)
    v = np.asarray(values, dtype=float)
    passing = np.isfinite(v) & (qf == 0) & sci_ok
    return ni.Series(name, start, start + ni.HALF_HOUR_S, v, None if vmin is None else np.asarray(vmin, dtype=float), None if vmax is None else np.asarray(vmax, dtype=float),
                     None if numpts is None else np.asarray(numpts, dtype=float), None, qf, sci_ok, passing)


# ---------------------------------------------------------------- reading and QA (tiny CSV)
def _csv(path, rows):
    head = "startDateTime,endDateTime,VSWCMean,VSWCExpUncert,VSWCFinalQF,VSWCFinalQFSciRvw\n"
    path.write_text(head + "".join(rows))


def test_read_swc_qa_flags_are_kept_as_supplied(tmp_path):
    p = tmp_path / "x.csv"
    _csv(p, ['"2025-04-01T00:00:00Z","2025-04-01T00:30:00Z",0.05,0.2135,0,\n', '"2025-04-01T00:30:00Z","2025-04-01T01:00:00Z",0.06,0.2135,1,\n',
             '"2025-04-01T01:00:00Z","2025-04-01T01:30:00Z",0.07,0.2135,0,1\n', '"2025-04-01T01:30:00Z","2025-04-01T02:00:00Z",,0.2135,0,\n',
             '"2025-04-01T02:00:00Z","2025-04-01T02:30:00Z",0.08,0.2135,0,0\n'])
    s = ni.read_swc(p, "swc")
    assert list(s.passing) == [True, False, False, False, True]          # FinalQF 1, science review 1 and a missing value all fail
    assert s.value[1] == 0.06 and s.final_qf[1] == 1                    # the failing value is retained, not filled or relabelled
    assert s.end_s[0] == s.start_s[0] + 1800


def test_read_refuses_a_gap_a_bad_interval_and_missing_columns(tmp_path):
    p = tmp_path / "gap.csv"
    _csv(p, ['"2025-04-01T00:00:00Z","2025-04-01T00:30:00Z",0.05,0.2,0,\n', '"2025-04-01T01:00:00Z","2025-04-01T01:30:00Z",0.06,0.2,0,\n'])
    with pytest.raises(ni.InputError, match="gap-free"):
        ni.read_swc(p, "swc")
    p.write_text("startDateTime,endDateTime,other\n")
    with pytest.raises(ni.InputError, match="missing columns"):
        ni.read_swc(p, "swc")
    with pytest.raises(ni.InputError, match="UTC time stamp"):
        ni.parse_utc("2025-04-01 00:00:00")


def test_hash_pins_refuse_a_changed_file(tmp_path):
    p = tmp_path / "f.bin"
    p.write_bytes(b"abc")
    sha, n = ni.sha256_file(p)
    ni.verify_file(p, sha, n)
    p.write_bytes(b"abd")
    with pytest.raises(ni.InputError, match="differ from the pin"):
        ni.verify_file(p, sha, n)
    with pytest.raises(ni.InputError, match="missing"):
        ni.verify_file(tmp_path / "nope", sha, n)


def test_depth_join_exactly_one_valid_record_and_positive_depth(tmp_path):
    p = tmp_path / "d"
    p.write_text("domainID,siteID,horizontalPosition.HOR,verticalPosition.VER,sensorDepth,startDateTime,endDateTime\nD14,JORN,004,501,-0.058,2010-01-01T00:00:00Z,NA\n"
                 "D14,JORN,004,502,-0.158,2010-01-01T00:00:00Z,2020-01-01T00:00:00Z\nD14,JORN,004,502,-0.159,2019-01-01T00:00:00Z,NA\n")
    sha, _ = ni.sha256_file(p)
    tab = ni.load_depth_table(p, sha)
    assert ni.sensor_depth_m(tab, "JORN", "004", "501", T0) == pytest.approx(0.058)
    with pytest.raises(ni.InputError, match="exactly one"):
        ni.sensor_depth_m(tab, "JORN", "004", "502", ni.parse_utc("2019-06-01T00:00:00Z"))     # two overlapping records
    with pytest.raises(ni.InputError, match="exactly one"):
        ni.sensor_depth_m(tab, "JORN", "004", "999", T0)
    with pytest.raises(ni.InputError, match="differs from the pin"):
        ni.load_depth_table(p, "0" * 64)


# ---------------------------------------------------------------- joint availability, END matching, rain-free runs
def test_joint_passing_and_end_matching_have_no_nearest_time_or_fill():
    a = series([0.1, 0.2, 0.3, 0.4], qf=[0, 0, 1, 0], name="a")
    b = series([0.5, 0.6, 0.7, 0.8], qf=[0, 1, 0, 0], name="b")
    joint = ni.joint_passing_end_times(a, b)
    assert list(joint) == [a.end_s[0], a.end_s[3]]
    targets = np.array([a.end_s[0], a.end_s[1] + 600, a.end_s[3]])          # the middle target is 10 min off the grid: no match, no nearest substitute
    m = ni.match_end_observations({"a": a, "b": b}, targets)
    assert np.isnan(m["a"][1]) and m["a"][0] == 0.1 and m["a"][2] == 0.4
    assert list(m["joint"]) == [True, False, True]
    m2 = ni.match_end_observations({"a": a, "b": b}, np.array([a.end_s[1], a.end_s[2]]))       # each has one level failing QA at that END
    assert list(m2["joint"]) == [False, False] and np.isfinite(m2["a"][0]) and np.isnan(m2["b"][0])


def test_grids_must_match():
    a, b = series([1, 2, 3]), series([1, 2, 3], t0=T0 + 1800)
    with pytest.raises(ni.InputError, match="not on the grid"):
        ni.joint_passing_end_times(a, b)


def test_dry_runs_rules_exclusion_censoring_and_the_rain_free_suffix():
    rain = np.zeros(48 * 12)
    rain[100] = 0.2           # one tip
    qf = np.zeros(len(rain), dtype=np.int64)
    qf[400] = 1               # failing gauge record breaks the run
    r = series(rain, qf=qf, name="rain")
    runs = ni.find_dry_runs(r, min_hours=48, exclude_hours=24)
    assert [(x.start_s - T0) // 1800 for x in runs] == [0, 101, 401]
    assert runs[0].run_hours == 50.0 and runs[0].censored_at_start and not runs[0].censored_at_end
    assert runs[0].exclusion_end_s == T0 + 86400 and runs[0].evaluable_hours == 26.0
    assert runs[2].censored_at_end
    assert ni.find_dry_runs(r, min_hours=500, exclude_hours=1) == []
    anchor = T0 + 1800 * 101                       # START of the run after the tip
    assert ni.max_rainfree_days(r, anchor) == 6    # 299 rain-free half-hours (101..399) = 6 whole days; the failing gauge record 400 ends the suffix
    with pytest.raises(ni.InputError):
        ni.max_rainfree_days(r, T0 + 7)


def test_initial_state_candidates_keep_first_joint_and_boundary_coincident_distinct():
    base = ni.parse_utc("2025-04-02T00:00:00Z")
    ends = np.array([base + 3600, base + 7200, base + 86400 + 25200, base + 2 * 86400 + 25200], dtype=np.int64)          # 01:00, 02:00, then 07:00 UTC on later days
    sel = ni.select_initial_state(ends, base, boundary_second_of_day=25200)
    assert sel["first_joint_at_or_after_exclusion_end_s"] == base + 3600
    assert sel["first_boundary_coincident_joint_s"] == base + 86400 + 25200
    assert ni.select_initial_state(ends, base + 10 * 86400)["first_joint_at_or_after_exclusion_end_s"] is None
    edge = ni.select_initial_state(ends, int(ends[0]))                      # an END equal to the exclusion end lies inside the excluded period: strictly after is required
    assert edge["first_joint_at_or_after_exclusion_end_s"] == int(ends[1])
    with pytest.raises(ni.InputError):
        ni.select_initial_state(ends, base, boundary_second_of_day=100)
    d = ni.daily_end_times(base + 3600, 3)
    assert list(np.diff(d)) == [86400] * 3 and d[0] == base + 3600


# ---------------------------------------------------------------- exact profile integration
def _numeric_integral(depths, theta, a, b, n=200001):
    z = np.linspace(a, b, n)
    th = np.interp(z, depths, theta)         # np.interp holds the end values constant outside: the same extension rule
    return float(np.trapezoid(th, z))


def test_profile_integration_conserves_the_analytical_integral_and_uses_actual_depths():
    depths = np.array([0.058, 0.158, 0.258])
    theta = np.array([0.04, 0.07, 0.06])
    edges = [0.0, 0.1, 0.2, 0.3]
    r = ni.integrate_initial_profile(depths, theta, edges)
    for k in range(3):
        assert r["layer_water_m"][k] == pytest.approx(_numeric_integral(depths, theta, edges[k], edges[k + 1]), rel=1e-9)
    analytic_total = 0.04 * 0.058 + 0.5 * (0.04 + 0.07) * 0.1 + 0.5 * (0.07 + 0.06) * 0.1 + 0.06 * (0.3 - 0.258)
    assert r["total_water_m"] == pytest.approx(analytic_total, rel=1e-14) and r["layer_water_m"].sum() == pytest.approx(analytic_total, rel=1e-14)
    assert r["layer_theta"][0] != pytest.approx(theta[0])               # the point value is NOT the layer mean by fiat
    assert r["layer_theta"] == pytest.approx(r["layer_water_m"] / 0.1)


def test_profile_integration_constant_profile_and_edge_cases():
    r = ni.integrate_initial_profile([0.058, 0.158, 0.258], [0.05, 0.05, 0.05], [0.0, 0.1, 0.2, 0.3])
    assert r["layer_theta"] == pytest.approx([0.05, 0.05, 0.05], abs=1e-15)
    one = ni.integrate_initial_profile([0.1], [0.07], [0.0, 0.05, 0.3])          # a single sensor: constant everywhere
    assert one["layer_theta"] == pytest.approx([0.07, 0.07])
    with pytest.raises(ni.InputError):
        ni.integrate_initial_profile([0.1, 0.1], [0.1, 0.1], [0.0, 0.3])
    with pytest.raises(ni.InputError, match="never clipped"):
        ni.integrate_initial_profile([0.058, 0.158], [0.05, 1.2], [0.0, 0.3])
    with pytest.raises(ni.InputError):
        ni.integrate_initial_profile([0.058, 0.158], [0.05, np.nan], [0.0, 0.3])
    with pytest.raises(ni.InputError):
        ni.integrate_initial_profile([0.058], [0.05], [0.1, 0.3])


# ---------------------------------------------------------------- complete-day air aggregation
def _air(n_days=3, *, drop=None, bad=None):
    n = 48 * n_days
    k = np.arange(n)
    mean = 15.0 + 8.0 * np.sin(2 * np.pi * (k % 48) / 48)
    vmin, vmax = mean - 0.7, mean + 0.9
    qf = np.zeros(n, dtype=np.int64)
    for i in bad or ():
        qf[i] = 1
    s = series(mean, qf=qf, vmin=vmin, vmax=vmax, numpts=np.full(n, 1800.0), name="air")
    if drop is not None:
        keep = np.ones(n, dtype=bool)
        keep[drop] = False
        idx = np.where(keep)[0]
        s = ni.Series("air", s.start_s[idx], s.end_s[idx], s.value[idx], s.vmin[idx], s.vmax[idx], s.numpts[idx], None, s.final_qf[idx], s.sci_ok[idx], s.passing[idx])
    return s


def test_complete_day_requires_all_48_and_uses_actual_min_and_max_not_means():
    s = _air()
    days = ni.aggregate_daily_air(s, T0 + 86400 * np.arange(3))
    assert all(d.complete for d in days)
    d0 = days[0]
    assert d0.tmin_c == pytest.approx(float(s.vmin[:48].min())) and d0.tmax_c == pytest.approx(float(s.vmax[:48].max()))
    assert d0.tmin_c < float(s.value[:48].min()) and d0.tmax_c > float(s.value[:48].max())              # extrema of the 30-minute extrema, not of the means
    assert d0.tmean_c == pytest.approx(float(s.value[:48].mean()))


def test_incomplete_or_failing_days_are_reported_not_filled():
    miss = ni.aggregate_daily_air(_air(drop=[10]), T0 + 86400 * np.arange(3))
    assert [d.complete for d in miss] == [False, True, True] and "47 of 48" in miss[0].reason and miss[0].tmin_c is None
    bad = ni.aggregate_daily_air(_air(bad=[60]), T0 + 86400 * np.arange(3))
    assert [d.complete for d in bad] == [True, False, True] and "1 of 48 records fail QA" in bad[1].reason
    shifted = ni.aggregate_daily_air(_air(), T0 + 1800 + 86400 * np.arange(3))                               # a window running past the file end is incomplete
    assert [d.complete for d in shifted] == [True, True, False]
    with pytest.raises(ni.InputError):
        ni.aggregate_daily_air(series([1.0] * 48), [T0])


def test_pet_uses_the_actual_weather_api_and_refuses_an_incomplete_day():
    pytest.importorskip("maple_syrup.weather_pet")
    from maple_syrup import weather_pet as wp

    day = ni.aggregate_daily_air(_air(), [T0])[0]
    pet = ni.hargreaves_daily_pet_mm(day, 32.59)
    rec = wp.DailyWeather(ni.day_date(T0), day.tmin_c, day.tmean_c, day.tmax_c)
    assert pet == wp.potential_demand(rec, 32.59, method="ecohyd_hargreaves").pet_mm_day and pet > 0.0
    assert ni.day_date(T0, 0).isoformat() == "2025-04-01" and ni.day_date(T0).isoformat() == "2025-04-01"      # midpoint 12:00 UTC = 05:00 at UTC-7
    assert ni.day_date(T0 + 12 * 3600, -7).isoformat() == "2025-04-01" and ni.day_date(T0 + 12 * 3600, 0).isoformat() == "2025-04-02"       # the fixed offset moves the date rule only when the midpoint crosses local midnight
    bad = ni.aggregate_daily_air(_air(drop=[3]), [T0])[0]
    with pytest.raises(ni.InputError, match="incomplete"):
        ni.hargreaves_daily_pet_mm(bad, 32.59)


def test_daily_median_is_descriptive_and_ignores_failing_records():
    s = series([0.1] * 48 + [0.2] * 48, qf=[1] + [0] * 95)
    med = ni.daily_median_descriptive(s, [T0, T0 + 86400])
    assert med[0] == pytest.approx(0.1) and med[1] == pytest.approx(0.2)


# ---------------------------------------------------------------- correction 01: strict flags, finite geometry, valid air weights
def test_fractional_or_non_numeric_flags_are_refused_never_read_as_zero(tmp_path):
    head = "startDateTime,endDateTime,VSWCMean,VSWCExpUncert,VSWCFinalQF,VSWCFinalQFSciRvw\n"
    p = tmp_path / "x.csv"
    for qf, sci in (("0.5", ""), ("0", "0.5"), ("abc", ""), ("nan", ""), ("0", "x")):
        p.write_text(head + f'"2025-04-01T00:00:00Z","2025-04-01T00:30:00Z",0.05,0.2,{qf},{sci}\n')
        with pytest.raises(ni.InputError, match="flag"):
            ni.read_swc(p, "swc")
    p.write_text(head + '"2025-04-01T00:00:00Z","2025-04-01T00:30:00Z",0.05,0.2,0.0,\n"2025-04-01T00:30:00Z","2025-04-01T01:00:00Z",0.05,0.2,,\n"2025-04-01T01:00:00Z","2025-04-01T01:30:00Z",0.05,0.2,2,\n')
    s = ni.read_swc(p, "swc")
    assert list(s.final_qf) == [0, -1, 2] and list(s.passing) == [True, False, False]                  # "0.0" is the exact integer 0; blank stays -1 (not 0); 2 is kept and fails


def test_profile_integration_rejects_non_finite_geometry():
    for depths, edges in (([np.nan, 0.158], [0.0, 0.3]), ([0.058, np.inf], [0.0, 0.3]), ([0.058, 0.158], [0.0, np.nan]), ([0.058, 0.158], [0.0, np.inf])):
        with pytest.raises(ni.InputError, match="finite"):
            ni.integrate_initial_profile(depths, [0.05, 0.06], edges)


def _air_with(**fields):
    s = _air(n_days=1)
    arrays = {"value": s.value.copy(), "vmin": s.vmin.copy(), "vmax": s.vmax.copy(), "numpts": s.numpts.copy()}
    for k, (idx, v) in fields.items():
        arrays[k][idx] = v
    return ni.Series("air", s.start_s, s.end_s, arrays["value"], arrays["vmin"], arrays["vmax"], arrays["numpts"], None, s.final_qf, s.sci_ok, s.passing)


@pytest.mark.parametrize("field,value,reason", [("numpts", 0.0, "point count"), ("numpts", -5.0, "point count"), ("numpts", float("nan"), "point count"), ("numpts", float("inf"), "point count"),
                                                ("vmin", 99.0, "minimum <= mean"), ("vmax", -99.0, "minimum <= mean")])
def test_a_48_pass_day_with_invalid_weights_or_ordering_is_marked_incomplete_explicitly(field, value, reason):
    d = ni.aggregate_daily_air(_air_with(**{field: (7, value)}), [T0])[0]
    assert not d.complete and reason in d.reason and d.tmean_c is None and d.n_passing == 48


# ---------------------------------------------------------------- correction 02: the science flag is blank-or-zero ONLY
@pytest.mark.parametrize("sci,ok", [("", True), ("0", True), ("0.0", True), ("1", False), ("-1", False), ("-1.0", False), ("2", False)])
def test_science_review_flag_passes_only_when_blank_or_zero_and_minus_one_never_passes(tmp_path, sci, ok):
    p = tmp_path / "x.csv"
    p.write_text("startDateTime,endDateTime,VSWCMean,VSWCExpUncert,VSWCFinalQF,VSWCFinalQFSciRvw\n" + f'"2025-04-01T00:00:00Z","2025-04-01T00:30:00Z",0.05,0.2,0,{sci}\n')
    s = ni.read_swc(p, "swc")
    assert bool(s.passing[0]) is ok and bool(s.sci_ok[0]) is ok
    p.write_text("startDateTime,endDateTime,VSWCMean,VSWCExpUncert,VSWCFinalQF,VSWCFinalQFSciRvw\n" + f'"2025-04-01T00:00:00Z","2025-04-01T00:30:00Z",0.05,0.2,{sci or 0},\n')
    assert bool(ni.read_swc(p, "swc").passing[0]) is (sci in ("", "0", "0.0"))                       # a literal -1 FINAL flag never passes either (it is not 0)
