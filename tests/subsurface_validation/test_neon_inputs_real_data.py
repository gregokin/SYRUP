"""Importer checks on the ACTUAL pinned RELEASE-2026 files (read-only). They re-derive the facts recorded in `phase3_observation_inventory.json` independently of the inventory generator and verify the newly retrieved air-temperature and
MOAB megapit manifests. Data-gated: skipped only when the pinned data tree is absent (the reason is printed)."""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarks.subsurface_validation import neon_inputs as ni
from benchmarks.subsurface_validation import paths as P

pytestmark = pytest.mark.skipif(not (P.RAW_ROOT.is_dir() and P.DEPTH_TABLE.is_file()), reason=f"the pinned data tree is absent ({P.RAW_ROOT})")
INV = json.loads(P.INVENTORY.read_text())


def _file(sec, role):
    return P.RAW_ROOT / next(e["path"] for e in INV["selected_files"][sec] if e["role"] == role)


@pytest.fixture(scope="module")
def sites():
    out = {}
    for key, sec, plot in (("JORN_plot4_2025-04", "JORN", 4), ("MOAB_plot2_2025-05", "MOAB", 2)):
        lev = {v: ni.read_swc(_file(sec, f"soil_moisture_plot{plot}_vertical{v}"), v) for v in ("501", "502", "503")}
        out[key] = {"levels": lev, "rain": ni.read_precip(_file(sec, "tipping_bucket_precip")), "sec": sec}
    return out


def test_all_selected_raw_files_match_their_pins():
    assert ni.verify_inventory_files(INV, P.RAW_ROOT) == 21


def test_corrected_depth_join_equals_the_inventory_and_not_the_shifted_sensor_file():
    tab = ni.load_depth_table(P.DEPTH_TABLE, INV["corrected_depth_table"]["sha256"])
    t = ni.parse_utc("2025-04-15T00:00:00Z")
    assert [ni.sensor_depth_m(tab, "JORN", "004", v, t) for v in ("501", "502", "503")] == pytest.approx([0.058, 0.158, 0.258])
    assert [ni.sensor_depth_m(tab, "MOAB", "002", v, ni.parse_utc("2025-05-15T00:00:00Z")) for v in ("501", "502", "503")] == pytest.approx([0.055, 0.155, 0.255])
    assert INV["sensor_depths_cm"]["JORN_plot4"] == pytest.approx([5.8, 15.8, 25.8]) and INV["sensor_depths_cm"]["MOAB_plot2"] == pytest.approx([5.5, 15.5, 25.5])


@pytest.mark.parametrize("key", ["JORN_plot4_2025-04", "MOAB_plot2_2025-05"])
def test_qa_and_joint_counts_equal_the_inventory(sites, key):
    d, c = sites[key], INV["candidates"][key]
    for v, s in d["levels"].items():
        assert int(s.passing.sum()) == c["per_level"][v]["passing"] and len(s) == c["per_level"][v]["rows"]
    assert len(ni.joint_passing_end_times(*d["levels"].values())) == c["joint_simultaneous_three_level_passing"]
    assert round(float(np.sum(d["rain"].value[d["rain"].passing])), 3) == c["rain"]["primary_tipping"]["total_passing_mm"]


@pytest.mark.parametrize("key", ["JORN_plot4_2025-04", "MOAB_plot2_2025-05"])
def test_dry_runs_equal_the_inventory_candidates(sites, key):
    runs = ni.find_dry_runs(sites[key]["rain"], min_hours=120, exclude_hours=72)
    inv = INV["candidates"][key]["candidate_dry_runs_declared_rule_primary_gauge"]
    assert [(ni.format_utc(r.start_s), ni.format_utc(r.last_start_s), r.run_hours, r.evaluable_hours) for r in runs] == [(x["run_start_utc"], x["run_last_start_utc"], x["run_hours"], x["evaluable_hours"]) for x in inv]
    assert [(r.censored_at_start, r.censored_at_end) for r in runs] == [(x["censored_at_month_start"], x["censored_at_month_end"]) for x in inv]


def test_jorn_run_a_anchor_candidates_and_rainfree_suffix(sites):
    d = sites["JORN_plot4_2025-04"]
    run = ni.find_dry_runs(d["rain"], min_hours=120, exclude_hours=72)[0]
    joint = ni.joint_passing_end_times(*d["levels"].values())
    sel = ni.select_initial_state(joint, run.exclusion_end_s, boundary_second_of_day=7 * 3600)
    assert ni.format_utc(sel["first_joint_at_or_after_exclusion_end_s"]) == "2025-04-08T22:00:00Z"            # the 21:00 END (= the exclusion end) lies inside the excluded 72 h
    assert ni.format_utc(sel["first_boundary_coincident_joint_s"]) == "2025-04-12T07:00:00Z"            # (b) is NOT the literally first joint measurement
    assert ni.max_rainfree_days(d["rain"], sel["first_joint_at_or_after_exclusion_end_s"]) == 13
    assert sel["first_boundary_coincident_joint_s"] in {int(x) for x in joint}


def test_initial_profile_of_the_first_joint_state_conserves_water(sites):
    d = sites["JORN_plot4_2025-04"]
    t = ni.parse_utc("2025-04-08T22:00:00Z")
    obs = ni.match_end_observations(d["levels"], np.array([t]))
    assert bool(obs["joint"][0])
    theta = np.array([obs[v][0] for v in ("501", "502", "503")])
    r = ni.integrate_initial_profile([0.058, 0.158, 0.258], theta, [0.0, 0.1, 0.2, 0.3])
    assert r["layer_water_m"].sum() == pytest.approx(r["total_water_m"], rel=1e-14) and np.all(r["layer_theta"] > 0.0)


def _air_file(root, site_dir, month):
    return glob.glob(str(root / "DP1.00002.001" / month / "*000.020.030*"))[0]


@pytest.mark.skipif(not (P.AIR_JORN / "manifest.json").is_file(), reason="the Root air-temperature retrieval is absent")
@pytest.mark.parametrize("root,month", [("AIR_JORN", "2025-04"), ("AIR_MOAB", "2025-05")])
def test_air_retrieval_manifests_match_the_files_and_the_declared_level_is_the_nearest_to_2_m(root, month):
    r = getattr(P, root)
    assert ni.verify_manifest_files(r / "manifest.json", r) == 12
    import csv
    with open(glob.glob(str(r / "DP1.00002.001" / month / "*sensor_positions*"))[0]) as fh:
        pos = list(csv.DictReader(fh))
    z = {x["HOR.VER"]: float(x["zOffset"]) for x in pos}
    assert min(z, key=lambda k: abs(z[k] - 2.0)) == "000.020"                                  # the a-priori height rule selects level 2 (1.51 m JORN, 1.95 m MOAB)


@pytest.mark.skipif(not (P.AIR_JORN / "manifest.json").is_file(), reason="the Root air-temperature retrieval is absent")
def test_air_series_qa_counts_and_days_incomplete_in_the_jorn_a_window(sites):
    air = ni.read_air_single(_air_file(P.AIR_JORN, "JORN", "2025-04"), expected_rows=1440)
    assert int(air.passing.sum()) == 1435
    start = ni.parse_utc("2025-04-08T22:00:00Z")
    days = ni.aggregate_daily_air(air, ni.daily_end_times(start, 13)[:-1])
    assert sum(d.complete for d in days) == 12 and [d.reason for d in days if not d.complete] == ["1 of 48 records fail QA"]
    assert all(d.tmin_c < d.tmean_c < d.tmax_c for d in days if d.complete)
