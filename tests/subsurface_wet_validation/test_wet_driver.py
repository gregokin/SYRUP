"""Engineering tests of the JORN 160-day field-column driver. Inputs are small SYNTHETIC files in the pinned formats; every physics call is the REAL native SYRUP/MAPLE code (no mocks of physics). They qualify composition, clocks,
ledgers, refusals, targets and restart of the DRIVER; they are not field validation and use no real data, GPU or network. They need the Root runtime (MAPLE importable; source pins are those of the importing environment)."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import synthetic as S

from benchmarks.subsurface_validation import drydown_driver as dd
from benchmarks.subsurface_validation import neon_inputs as ni
from benchmarks.subsurface_wet_validation import config as C
from benchmarks.subsurface_wet_validation import driver as D
from benchmarks.subsurface_wet_validation import inputs as I

LOW = {50: 0.5, 51: 0.5, 53: 2.0}                    # day 1: rows 2,3,5 of the day (1 and 4 mm/h) with a zero gap row: far below K = 38.2 mm/h
HIGH = {100: 20.0, 101: 25.0, 102: 20.0}              # day 2: 40-50 mm/h > K: onset, ponding, capacity-limited runoff


def run(tmp_path, **kw):
    cfg = S.build(tmp_path, **kw)
    c, sha = C.load_config(S.write_config(tmp_path, cfg))
    return c, sha, D.execute(c, sha)


# ------------------------------------------------------------------ config refusals
@pytest.mark.parametrize("path,value", [("schema", "x"), ("matrix.suction_m", None), ("matrix.conductivity_m_s", 1e-5), ("storm.max_dt_s", 600), ("controls.backend", "cuda"), ("physics.transpiration_factor", 1),
                                        ("physics.bottom", "closed"), ("layers.edges_m", [0.0, 0.1, 0.1, 0.3]), ("forcing.files.rainfall_30min.path", "relative.csv"), ("initial.end_utc", "2024-01-18T07:30:00Z"),
                                        ("forcing.expected.rain_rows", 5), ("extra", 1), ("swc.levels.501", [])])
def test_config_refusals(tmp_path, path, value):
    cfg = S.mutated(S.build(tmp_path), path, value)
    with pytest.raises(C.ConfigError):
        C.validate_config(cfg)


def test_config_missing_key_names_it(tmp_path):
    with pytest.raises(C.ConfigError, match="suction_provenance|missing"):
        C.validate_config(S.mutated(S.build(tmp_path), "matrix.suction_provenance", KeyError))


# ------------------------------------------------------------------ staged refusals before any native field step
def _no_native(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("a native field object was built before the refusal")
    monkeypatch.setattr(D.Run, "__init__", boom)


def test_pin_hash_refusal_precedes_native(tmp_path, monkeypatch):
    cfg = S.build(tmp_path)
    c, sha = C.load_config(S.write_config(tmp_path, cfg))
    Path(cfg["forcing"]["files"]["rainfall_30min"]["path"]).write_text("tampered")
    _no_native(monkeypatch)
    with pytest.raises(ni.InputError, match="pin"):
        D.execute(c, sha)


def test_source_pair_refusal_precedes_native(tmp_path, monkeypatch):
    cfg = S.build(tmp_path)
    cfg["expected_sources"]["maple_syrup"]["package_sha256"] = "0" * 64
    c, sha = C.load_config(S.write_config(tmp_path, cfg))
    _no_native(monkeypatch)
    with pytest.raises(D.ControlFailure, match="not the declared"):
        D.execute(c, sha)


@pytest.mark.parametrize("edit", ["rain_qf", "rain_total", "air_date", "swc_ic", "depth", "rain_negative"])
def test_forcing_and_mask_refusals_precede_native(tmp_path, monkeypatch, edit):
    cfg = S.build(tmp_path, rain_mm=LOW)
    rp = Path(cfg["forcing"]["files"]["rainfall_30min"]["path"])
    ap = Path(cfg["forcing"]["files"]["daily_air"]["path"])
    if edit == "rain_qf":
        rp.write_text(rp.read_text().replace(",0,\n", ",1,\n", 1))
    elif edit == "rain_negative":
        rp.write_text(rp.read_text().replace("0.000", "-0.001", 1))
    elif edit == "rain_total":
        cfg["forcing"]["expected"]["total_rain_mm"] = "9.9"
    elif edit == "air_date":
        ap.write_text(ap.read_text().replace("2024-01-19", "2024-01-20"))
    elif edit == "swc_ic":
        cfg["initial"]["expected_theta"] = [0.0709, 0.0921, 0.0659]            # the START-labelled screen values are never the IC
    elif edit == "depth":
        cfg["initial"]["expected_depths_m"] = [0.06, 0.16, 0.26]
    for f in (rp, ap):                                                          # re-pin edited files so ONLY the validation (not the hash) refuses
        h, n = ni.sha256_file(f)
        for e in cfg["forcing"]["files"].values():
            if e["path"] == str(f):
                e["sha256"], e["bytes"] = h, n
    c, sha = C.load_config(S.write_config(tmp_path, cfg))
    _no_native(monkeypatch)
    with pytest.raises(ni.InputError):
        D.execute(c, sha)


def test_plan_only_builds_no_native_object(tmp_path, monkeypatch, capsys):
    cfg = S.build(tmp_path, rain_mm=LOW)
    _no_native(monkeypatch)
    assert D.main(["--config", str(S.write_config(tmp_path, cfg)), "--out", str(tmp_path / "o")]) == 0
    plan = json.loads((tmp_path / "o" / "plan.json").read_text())
    assert plan["n_wet_days"] == 1 and plan["profile"] == C.PROFILE and plan["matrix"]["peak_rain_exceeds_K"] is False
    assert not (tmp_path / "o" / "result.json").exists()


# ------------------------------------------------------------------ physics composition (real native code)
def test_zero_rain_reproduces_daily_column(tmp_path):
    c, _sha, (res, _, _) = run(tmp_path)
    from maple_syrup import subsurface as ss

    prep = I.prepare_inputs(c)
    col = ss.prepare_column(dd._layers({"layers": c["layers"]}, {"evaporation_factor": 1.0, "diffusion_coefficient": c["physics"]["diffusion_coefficient"], "bottom": "open"}), np.array([True]), 1.0, backend="numpy")
    st = ss.column_state(col, prep["initial"]["layer_water_m"].reshape(-1, 1).copy(), 0.0)
    for k, row in enumerate(res["rows"]):
        st = ss.step(col, st, 86400.0, added_water_m=np.zeros((3, 1)), potential_evaporation_m_s=np.array([prep["pet_mm_day"][k] / 1000.0 / 86400.0])).state
        assert row["water_m"] == np.asarray(st.water_m)[:, 0].tolist()          # bit-identical to the old native daily law
        assert row["physical_t_s"] == 86400.0 * (k + 1) and row["added_storm_s"] == 0.0 and row["n_storm_steps"] == 0
    assert res["ledger"]["n_matrix_steps"] == 0 and res["added_rain_hours"] == 0.0


def test_low_rate_all_intake_and_clocks(tmp_path):
    _c, _sha, (res, _, _) = run(tmp_path, rain_mm=LOW)
    r1 = res["rows"][1]
    assert r1["storm_rows"] == 4 and r1["first_wet_row"] == 2 and r1["last_wet_row"] == 5 and r1["n_storm_steps"] == 4     # zero gap row kept inside the schedule
    assert r1["runoff_pool_m3"] == 0.0 and r1["n_ponded_steps"] == 0 and r1["n_onsets"] == 0 and r1["room_rejected_m3"] == 0.0
    assert r1["intake_m3"] == pytest.approx(r1["rain_m3"], rel=1e-15) and r1["rain_m3"] == pytest.approx(3.0e-3, rel=1e-12)
    assert r1["added_storm_s"] == 7200.0 and r1["physical_t_s"] == 86400.0 * 2 + 7200.0                  # storm time is ADDITIONAL physical time, exact
    assert res["physical_end_t_s"] == 86400.0 * 4 + 7200.0 and res["nominal_elapsed_s"] == 86400.0 * 4 and res["added_rain_hours"] == 2.0
    assert res["rows"][1]["nominal_close_utc"] == "2024-01-20T07:00:00Z"                              # nominal calendar is separate from the physical clock
    cc = res["cumulative_closure"]
    assert abs(cc["soil_residual_m3"]) <= cc["soil_bound_m3"] and abs(cc["rain_residual_m3"]) <= cc["rain_bound_m3"]


def test_et_off_during_rain_and_once_per_day(tmp_path, monkeypatch):
    from maple_syrup import subsurface as ss

    calls, real = [], ss.step

    def spy(column, state, dt_s, **kw):
        calls.append((float(dt_s), float(np.asarray(kw["potential_evaporation_m_s"])[0]), float(np.asarray(kw["added_water_m"]).sum())))
        return real(column, state, dt_s, **kw)

    monkeypatch.setattr(ss, "step", spy)
    init = D.Run.initialise
    monkeypatch.setattr(D.Run, "initialise", lambda self: (calls.clear(), init(self))[1])      # the disposable warm-up control steps precede initialise: not accepted steps
    run(tmp_path, rain_mm=LOW | HIGH)
    daily = [x for x in calls if x[0] == 86400.0]
    storm = [x for x in calls if x[0] != 86400.0]
    assert len(daily) == 4 and all(x[1] > 0.0 for x in daily)                  # exactly one interevent step per nominal day, with PET (E on)
    assert storm and all(x[1] == 0.0 for x in storm)                          # no evaporation during any rain sub-step
    assert all(x[2] == 0.0 for x in calls)                                    # the matrix intake is credited once by the matrix stage, never added again in the soil step


def test_dt900_control_halves_steps_same_clock(tmp_path):
    a = run(tmp_path / "a" if (tmp_path / "a").mkdir() is None else tmp_path, rain_mm=LOW)[2][0]
    (tmp_path / "b").mkdir()
    b = run(tmp_path / "b", rain_mm=LOW, max_dt_s=900)[2][0]
    assert b["rows"][1]["n_storm_steps"] == 2 * a["rows"][1]["n_storm_steps"]
    assert a["physical_end_t_s"] == b["physical_end_t_s"] and b["rows"][1]["added_storm_s"] == 7200.0
    assert b["rows"][1]["intake_m3"] == pytest.approx(a["rows"][1]["intake_m3"], rel=1e-12)


def test_capacity_limited_rain_exports_runoff_and_conserves(tmp_path):
    _c, _sha, (res, _, _) = run(tmp_path, rain_mm=HIGH)
    r = res["rows"][2]
    assert r["n_ponded_steps"] > 0 and r["runoff_pool_m3"] > 0.0 and r["postpond_intake_m3"] > 0.0
    assert r["rain_m3"] == pytest.approx(r["intake_m3"] + r["runoff_pool_m3"], rel=1e-12) and abs(r["closure_residual_m3"]) <= r["closure_bound_m3"]
    assert res["matrix_regime"]["n_ponded_steps"] > 0 and res["plan"]["matrix"]["peak_rain_exceeds_K"] is True
    assert res["plan"]["matrix"]["disclosure"].startswith("psi and g are UNMEASURED")


def test_wetting_history_retained_across_dry_boundaries(tmp_path):
    _c, _sha, (res, _, _) = run(tmp_path, rain_mm=HIGH, n_days=5)
    J = [r["wetting_J_m"] for r in res["rows"]]
    assert J[0] == 0.0 and J[2] > 0.0 and J[3] == J[2] == J[4]                 # not reset by the dry days that follow, never recovered
    assert [r["cumulative_matrix_intake_m"] for r in res["rows"]][-1] == pytest.approx(res["ledger"]["intake_m3"], rel=1e-12)


def test_pet_is_the_weather_pet_api_without_canopy_factor(tmp_path):
    from maple_syrup import weather_pet as wp

    c, _sha, (res, _, _) = run(tmp_path)
    prep = I.prepare_inputs(c)
    for k, d in enumerate(prep["air"]):
        rec = wp.DailyWeather(d["date"], d["tmin_c"], d["tmean_c"], d["tmax_c"], provenance="observed")
        assert prep["pet_mm_day"][k] == wp.potential_demand(rec, 32.59052, method="ecohyd_hargreaves").pet_mm_day == res["rows"][k]["pet_mm_day"]


# ------------------------------------------------------------------ initial state and targets
def test_initial_uses_end_labelled_records_and_end_matched_targets(tmp_path):
    c, _sha, (res, _, _) = run(tmp_path)
    prep = I.prepare_inputs(c)
    assert prep["theta0_points"].tolist() == list(S.IC_THETA) and prep["t0_s"] == S.T0
    assert prep["initial"]["layer_theta"][0] == pytest.approx(prep["initial"]["layer_water_m"][0] / 0.1)
    levels = I.read_levels(c)
    v, ok = I.lookup(levels, "501", S.T0 + 86400)                              # the record ENDING at the first forecast close
    assert ok and prep["obs"][0, 0] == v and res["rows"][0]["obs_theta"][0] == v
    assert I.lookup(levels, "501", S.T0 + 86400 + 1)[1] is False
    assert "labelling error" in prep["support_note"]


def test_missing_and_failed_swc_targets_are_null_and_do_not_truncate(tmp_path):
    t1, t2 = S.T0 + 2 * 86400, S.T0 + 3 * 86400
    clean = run(tmp_path / "x" if (tmp_path / "x").mkdir() is None else tmp_path, rain_mm=LOW)[2][0]
    (tmp_path / "y").mkdir()
    _c, _sha, (res, _, _) = run(tmp_path / "y", rain_mm=LOW, swc_fail={("501", t1)}, swc_missing={("502", t2)})
    assert res["days_done"] == 4 and len(res["rows"]) == 4                     # all days run despite bad/missing sensor targets
    assert res["rows"][1]["obs_theta"][0] is None and res["rows"][2]["obs_theta"][1] is None and res["rows"][2]["obs_theta"][0] is not None
    assert res["scores"]["raw"]["501"]["n"] == 3 and res["scores"]["raw"]["502"]["n"] == 3 and res["scores"]["raw"]["503"]["n"] == 4
    assert [r["water_m"] for r in res["rows"]] == [r["water_m"] for r in clean["rows"]]   # observations never drive, stop or reset the model


def test_initial_record_must_pass_qc(tmp_path):
    cfg = S.build(tmp_path, swc_fail={("502", S.T0)})
    c, _ = C.load_config(S.write_config(tmp_path, cfg))
    with pytest.raises(ni.InputError, match="no QC-passing record ENDING"):
        I.prepare_inputs(c)


def test_two_swc_files_merge_and_duplicate_end_refused(tmp_path):
    cfg = S.build(tmp_path, split_swc=True)
    c, _ = C.load_config(S.write_config(tmp_path, cfg))
    assert len(c["swc"]["levels"]["501"]) == 2 and np.isfinite(I.prepare_inputs(c)["obs"]).all()
    dup = copy.deepcopy(cfg)
    dup["swc"]["levels"]["501"].append(dup["swc"]["levels"]["501"][0])
    c2, _ = C.load_config(S.write_config(tmp_path, dup, "dup.json"))
    with pytest.raises(ni.InputError, match="duplicate END"):
        I.read_levels(c2)


def test_scoring_raw_and_fixed_offset_aligned():
    prep = {"obs": np.array([[0.10, 0.20, 0.30], [0.11, np.nan, 0.28]]), "probe_layers": [0, 1, 2], "theta0_points": np.array([0.09, 0.21, 0.29]), "initial": {"layer_theta": np.array([0.08, 0.22, 0.30])}}
    pred = np.array([[0.09, 0.21, 0.31], [0.10, 0.20, 0.30]])
    s = I.score_run(prep, pred)
    assert s["raw"]["501"]["bias"] == pytest.approx(((0.09 - 0.10) + (0.10 - 0.11)) / 2) and s["raw"]["502"]["n"] == 1
    # offset aligned: (pred - m0) - (obs - o0): layer 0 -> (0.01-0.01), (0.02-0.02) = 0
    assert s["offset_aligned"]["501"]["rmse"] == pytest.approx(0.0, abs=1e-15) and s["persistence_offset_aligned"]["501"]["rmse"] == pytest.approx(np.sqrt((0.01**2 + 0.02**2) / 2))
    assert s["initial_offset_model_minus_point"] == pytest.approx([-0.01, 0.01, 0.01])


# ------------------------------------------------------------------ restart (driver-level) and immutable outputs
def test_midpoint_restart_identical_and_guarded(tmp_path):
    cfg = S.build(tmp_path, rain_mm=LOW | HIGH, n_days=5)
    c, sha = C.load_config(S.write_config(tmp_path, cfg))
    full, _, _ = D.execute(c, sha)
    part, state, _ = D.execute(c, sha, stop_after_day=3)
    assert part["complete"] is False and part["scores"] is None and state["next_day"] == 3 and "NOT a production checkpoint" in state["label"]
    state = json.loads(json.dumps(state))                                          # through the on-disk JSON form
    done, _, _ = D.execute(c, sha, resume_state=state)
    for key in ("rows", "ledger", "physical_end_t_s", "added_rain_hours", "cumulative_closure", "scores"):
        assert done[key] == full[key], key
    # refusals
    with pytest.raises(D.ControlFailure, match="unmodified"):
        D.execute(c, sha, resume_state={**state, "t_s": state["t_s"] + 1.0})
    other = copy.deepcopy(cfg)
    other["name"] = "another"
    c2, sha2 = C.load_config(S.write_config(tmp_path, other, "other.json"))
    with pytest.raises(D.ControlFailure, match="different config"):
        D.execute(c2, sha2, resume_state=state)
    dup = copy.deepcopy(state)
    dup["rows"][1]["day_index"] = 0
    dup["state_sha256"] = __import__("hashlib").sha256(json.dumps({k: v for k, v in dup.items() if k != "state_sha256"}, sort_keys=True, allow_nan=False).encode()).hexdigest()
    with pytest.raises(D.ControlFailure, match="ordinals"):
        D.execute(c, sha, resume_state=dup)
    with pytest.raises(D.ControlFailure, match="beyond the resumed day"):
        D.execute(c, sha, stop_after_day=3, resume_state=state)


def test_changed_input_after_export_is_refused_on_resume(tmp_path):
    cfg = S.build(tmp_path, rain_mm=LOW, n_days=4)
    c, sha = C.load_config(S.write_config(tmp_path, cfg))
    _, state, _ = D.execute(c, sha, stop_after_day=2)
    Path(cfg["forcing"]["files"]["daily_air"]["path"]).write_text("changed")
    with pytest.raises(ni.InputError, match="pin"):
        D.execute(c, sha, resume_state=state)


def test_immutable_output_and_deterministic_retry(tmp_path):
    cfg = S.build(tmp_path, rain_mm=LOW)
    p = S.write_config(tmp_path, cfg)
    out = tmp_path / "out"
    assert D.main(["--config", str(p), "--out", str(out), "--execute"]) == 0
    first = (out / "result.json").read_bytes()
    assert D.main(["--config", str(p), "--out", str(out), "--execute"]) == 0      # identical retry accepted (physical results are deterministic)
    assert (out / "result.json").read_bytes() == first and (out / "timing.json").exists()
    assert D.write_immutable(out / "result.json", first.decode()) == "identical_retry"
    with pytest.raises(D.ControlFailure, match="immutable"):
        D.write_immutable(out / "result.json", "different")
    t = json.loads((out / "timing.json").read_text())
    assert {"source_checks_and_input_preparation_s", "operator_prepare_s", "jit_warmup_control_s", "integration_s", "output_write_s", "backend"} <= set(t)


def test_numba_matches_numpy_within_inherited_contract(tmp_path):
    pytest.importorskip("numba")
    (tmp_path / "np").mkdir()
    (tmp_path / "nb").mkdir()
    a = run(tmp_path / "np", rain_mm=LOW | HIGH, n_days=4)[2][0]
    b = run(tmp_path / "nb", rain_mm=LOW | HIGH, n_days=4, backend="numba")[2][0]
    for ra, rb in zip(a["rows"], b["rows"]):
        assert np.allclose(ra["water_m"], rb["water_m"], rtol=2e-12, atol=1e-14) and ra["n_storm_steps"] == rb["n_storm_steps"]
    assert b["backend"] == "numba" and a["backend"] == "numpy"


# ================================================================== correction01
def _edit_pinned(cfg: dict, role: str, fn) -> None:
    """Rewrite a pinned forcing file with `fn(text)` and re-pin it, so ONLY content validation (not the hash) can refuse."""
    e = cfg["forcing"]["files"][role]
    p = Path(e["path"])
    p.write_text(fn(p.read_text()))
    e["sha256"], e["bytes"] = ni.sha256_file(p)


def test_daily_air_weight_is_sample_count_weight_not_elapsed_seconds(tmp_path):
    cfg = S.build(tmp_path, low_weight={48 + 3: 1726.0, 48 + 4: 1700.0})           # day 1: weight 86326 - 74 ... all 48 half hours still represented
    c, _ = C.load_config(S.write_config(tmp_path, cfg))
    air = I.load_daily_air(c)
    assert air[1]["weight"] == 86400.0 - 74.0 - 100.0 + 0.0 and air[0]["weight"] == 86400.0
    assert I.prepare_inputs(c)["n_days"] == 4


def test_daily_air_weight_86326_exact(tmp_path):
    cfg = S.build(tmp_path, low_weight={50: 1726.0})
    c, _ = C.load_config(S.write_config(tmp_path, cfg))
    assert I.load_daily_air(c)[1]["weight"] == 86326.0


@pytest.mark.parametrize("edit", ["zero_weight", "nan_weight", "weight_off", "mean_off", "tmax_off", "frac_count", "neg_count", "count_sum", "interp_count"])
def test_daily_air_refusals(tmp_path, edit):
    cfg = S.build(tmp_path, interpolated_days=(1,))
    lines = {"zero_weight": ("86400.0", "0.0"), "nan_weight": ("86400.0", "nan"), "weight_off": ("86400.0", "86399.0"), "frac_count": (",47,1,", ",47.5,0.5,"), "neg_count": (",47,1,", ",49,-1,"), "count_sum": (",47,1,", ",47,2,"),
             "interp_count": (",reconstructed_complete", ",observed_complete")}
    if edit in lines:
        old, new = lines[edit]
        _edit_pinned(cfg, "daily_air", lambda t: t.replace(old, new, 1))
    elif edit == "mean_off":
        _edit_pinned(cfg, "daily_air", _perturb_mean)
    else:
        _edit_pinned(cfg, "daily_air", _bump_tmax)
    c, _ = C.load_config(S.write_config(tmp_path, cfg))
    with pytest.raises(ni.InputError):
        I.load_daily_air(c)


def _perturb_mean(text: str) -> str:
    rows = text.split("\n")
    f = rows[2].split(",")
    f[3] = repr(float(f[3]) + 0.01)
    rows[2] = ",".join(f)
    return "\n".join(rows)


def _bump_tmax(text: str) -> str:
    rows = text.split("\n")
    f = rows[2].split(",")
    f[2] = repr(float(f[2]) + 0.5)
    rows[2] = ",".join(f)
    return "\n".join(rows)


def test_daily_air_bound_to_derived_rows_and_receipt(tmp_path):
    cfg = S.build(tmp_path, interpolated_days=(1,))
    _edit_pinned(cfg, "receipt", lambda t: t.replace('"rows": 4', '"rows": 5'))
    c, _ = C.load_config(S.write_config(tmp_path, cfg))
    with pytest.raises(ni.InputError, match="receipt"):
        I.load_daily_air(c)
    cfg2 = S.build(tmp_path / "d", interpolated_days=(1,)) if (tmp_path / "d").mkdir() is None else None
    _edit_pinned(cfg2, "derived_30min", lambda t: t.replace("false,true,1800.0", "false,true,1799.0", 1))      # one weight changed: the daily weight no longer folds
    c2, _ = C.load_config(S.write_config(tmp_path / "d", cfg2))
    with pytest.raises(ni.InputError, match="aggregation"):
        I.load_daily_air(c2)


def test_no_artificial_scale_floor_in_closure():
    assert "1e-300" not in Path(D.__file__).read_text()


def test_pooled_scores_equal_weight_unequal_counts_and_empty_depth():
    obs = np.array([[0.10, np.nan, 0.30], [0.12, np.nan, np.nan], [0.14, np.nan, 0.31]])
    prep = {"obs": obs, "probe_layers": [0, 1, 2], "theta0_points": np.array([0.09, 0.21, 0.29]), "initial": {"layer_theta": np.array([0.08, 0.22, 0.30])}}
    pred = np.array([[0.11, 0.2, 0.28], [0.10, 0.2, 0.3], [0.17, 0.2, 0.34]])
    s = I.score_run(prep, pred)
    e = np.array([0.11 - 0.10, 0.10 - 0.12, 0.17 - 0.14, 0.28 - 0.30, 0.34 - 0.31])                            # 3 depth-1 + 2 depth-3 pairs, depth 2 EMPTY
    assert s["overall_n_raw"] == 5 and s["overall_rmse_raw"] == pytest.approx(np.sqrt(np.mean(e**2)), rel=1e-14)
    assert s["pooled_raw"]["bias"] == pytest.approx(e.mean()) and s["equal_depth_rmse_raw"] is None and s["raw"]["502"]["n"] == 0
    eo = np.array([(0.11 - 0.08) - (0.10 - 0.09), (0.10 - 0.08) - (0.12 - 0.09), (0.17 - 0.08) - (0.14 - 0.09), (0.28 - 0.30) - (0.30 - 0.29), (0.34 - 0.30) - (0.31 - 0.29)])
    assert s["overall_rmse_offset_aligned"] == pytest.approx(np.sqrt(np.mean(eo**2)), rel=1e-12)
    ep = np.array([0.10 - 0.09, 0.12 - 0.09, 0.14 - 0.09, 0.30 - 0.29, 0.31 - 0.29])
    assert s["overall_persistence_rmse_raw"] == pytest.approx(np.sqrt(np.mean(ep**2)), rel=1e-12)
    assert s["overall_persistence_rmse_offset_aligned"] == pytest.approx(np.sqrt(np.mean(ep**2)), rel=1e-12)
    assert s["overall_n_offset_aligned"] == 5
    full = I.score_run({**prep, "obs": np.where(np.isnan(obs), 0.2, obs)}, pred)                                # all depths present: the labelled equal-depth metric exists and differs from the pooled one
    assert full["equal_depth_rmse_raw"] is not None and full["overall_rmse_raw"] != full["equal_depth_rmse_raw"]


def _swc_file(cfg: dict, level: str = "501", index: int = 0) -> Path:
    return Path(cfg["swc"]["levels"][level][index]["path"])


def _repin_swc(cfg: dict, level: str, fn) -> None:
    e = cfg["swc"]["levels"][level][0]
    p = Path(e["path"])
    p.write_text(fn(p.read_text()))
    e["sha256"], e["bytes"] = ni.sha256_file(p)


def test_actual_removed_swc_row_leaves_forecast_unchanged_and_target_null(tmp_path):
    clean_dir, cut_dir = tmp_path / "clean", tmp_path / "cut"
    clean_dir.mkdir()
    cut_dir.mkdir()
    clean = run(clean_dir, rain_mm=LOW)[2][0]
    cfg = S.build(cut_dir, rain_mm=LOW)
    close = ni.format_utc(S.T0 + 2 * 86400)                                                                       # the record ENDING at the day-1 close
    before = _swc_file(cfg).read_text().count("\n")
    _repin_swc(cfg, "501", lambda t: "\n".join(line for line in t.split("\n") if f",{close}," not in line))
    assert _swc_file(cfg).read_text().count("\n") == before - 1                                                   # a genuine missing row in the MIDDLE of the file
    c, sha = C.load_config(S.write_config(cut_dir, cfg))
    res = D.execute(c, sha)[0]
    assert res["rows"][1]["obs_theta"][0] is None and res["rows"][0]["obs_theta"][0] is not None and res["scores"]["raw"]["501"]["n"] == 3
    assert [r["water_m"] for r in res["rows"]] == [r["water_m"] for r in clean["rows"]] and res["rows"][-1]["wetting_J_m"] == clean["rows"][-1]["wetting_J_m"]


@pytest.mark.parametrize("bad", ["not_a_number", "qf_fraction", "qf_text", "sci_fraction", "empty_value"])
def test_invalid_numeric_or_flag_in_observation_is_a_failed_target(tmp_path, bad):
    cfg = S.build(tmp_path)
    close = ni.format_utc(S.T0 + 2 * 86400)

    def edit(text: str) -> str:
        out = []
        for line in text.split("\n"):
            f = line.split(",")
            if len(f) > 5 and f[1] == close:
                if bad == "not_a_number":
                    f[2] = "abc"
                elif bad == "empty_value":
                    f[2] = ""
                elif bad == "qf_fraction":
                    f[3] = "0.5"
                elif bad == "qf_text":
                    f[3] = "x"
                else:
                    f[4] = "0.5"
            out.append(",".join(f))
        return "\n".join(out)

    _repin_swc(cfg, "502", edit)
    c, _ = C.load_config(S.write_config(tmp_path, cfg))
    prep = I.prepare_inputs(c)
    assert np.isnan(prep["obs"][1, 1]) and np.isfinite(prep["obs"][1, 0]) and prep["n_days"] == 4


@pytest.mark.parametrize("edit", ["duplicate", "unsorted", "duration", "columns"])
def test_observation_file_structure_stays_strict(tmp_path, edit):
    cfg = S.build(tmp_path)

    def fn(text: str) -> str:
        lines = text.split("\n")
        if edit == "duplicate":
            lines.insert(4, lines[3])
        elif edit == "unsorted":
            lines[3], lines[4] = lines[4], lines[3]
        elif edit == "duration":
            f = lines[3].split(",")
            f[0] = ni.format_utc(ni.parse_utc(f[1]) - 3600)
            lines[3] = ",".join(f)
        else:
            lines[0] = lines[0].replace("VSWCMean", "other")
        return "\n".join(lines)

    _repin_swc(cfg, "503", fn)
    c, _ = C.load_config(S.write_config(tmp_path, cfg))
    with pytest.raises(ni.InputError):
        I.prepare_inputs(c)


def test_initial_still_needs_all_three_passing_with_in_file_gap(tmp_path):
    cfg = S.build(tmp_path, swc_missing={("503", S.T0)})
    c, _ = C.load_config(S.write_config(tmp_path, cfg))
    with pytest.raises(ni.InputError, match="no QC-passing record ENDING"):
        I.prepare_inputs(c)


# ------------------------------------------------------------------ staged checkpoint import
def _reseal(st: dict) -> dict:
    st = {k: v for k, v in st.items() if k != "state_sha256"}
    st["state_sha256"] = __import__("hashlib").sha256(json.dumps(st, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return st


def _snapshot(run) -> tuple:
    return (json.dumps(run.rows, sort_keys=True), dict(run.ledger), run.storage0, np.asarray(run.state.water_m).tolist(), float(run.state.t_s), np.asarray(run.history.wetting_m).tolist(), np.asarray(run.history.cumulative_intake_m).tolist())


@pytest.mark.parametrize("corrupt", ["last_water", "clock", "storage0", "ledger_rain", "ledger_time", "ledger_steps", "wetting", "row_pet", "n_terms"])
def test_failed_import_leaves_run_unchanged_even_with_recomputed_seal(tmp_path, corrupt):
    cfg = S.build(tmp_path, rain_mm=LOW | HIGH, n_days=5)
    c, sha = C.load_config(S.write_config(tmp_path, cfg))
    _, state, _ = D.execute(c, sha, stop_after_day=3)
    state = json.loads(json.dumps(state))
    prep = I.prepare_inputs(c)
    run_ = D.Run(c, prep)
    run_.initialise()
    run_.run_days(0, 2)                                                                                           # an ACCEPTED, different in-memory run
    src = D.check_sources(c)
    before = _snapshot(run_)
    bad = json.loads(json.dumps(state))
    if corrupt == "last_water":
        bad["water_m"][0] += 1e-9
    elif corrupt == "clock":
        bad["t_s"] += 900.0
    elif corrupt == "storage0":
        bad["storage0_m3"] += 1e-9
    elif corrupt == "ledger_rain":
        bad["ledger"]["rain_m3"] += 1e-6
    elif corrupt == "ledger_time":
        bad["ledger"]["storm_time_s"] += 1800.0
    elif corrupt == "ledger_steps":
        bad["ledger"]["n_soil_steps"] += 1.0
    elif corrupt == "wetting":
        bad["wetting_m"] += 1e-9
    elif corrupt == "row_pet":
        bad["rows"][0]["pet_mm_day"] += 0.1
    else:
        bad["ledger"]["n_terms"] = float("nan")
    with pytest.raises((D.ControlFailure, ValueError)):
        run_.import_state(_reseal(bad), sha, src)
    assert _snapshot(run_) == before                                                                              # nothing was adopted
    run_.import_state(state, sha, src)                                                                            # the genuine state is accepted and continues exactly
    run_.run_days(3, 5)
    assert run_.rows == D.execute(c, sha)[0]["rows"]


def test_resume_timing_counters_are_labelled(tmp_path):
    cfg = S.build(tmp_path, rain_mm=LOW | HIGH, n_days=5)
    c, sha = C.load_config(S.write_config(tmp_path, cfg))
    _, state, t1 = D.execute(c, sha, stop_after_day=3)
    _, _, t2 = D.execute(c, sha, resume_state=json.loads(json.dumps(state)))
    assert t1["n_days_integrated_this_process"] == 3 and t2["n_days_integrated_this_process"] == 2 and t2["n_days_cumulative"] == 5
    assert t2["n_soil_steps_cumulative"] == t1["n_soil_steps_cumulative"] + t2["n_soil_steps_this_process"] and "excludes interpreter start" in t2["notes"]
    assert "imports_and_input_prep_s" not in t2


# ------------------------------------------------------------------ ACTUAL case files, PLAN-only data import (no field integration)
CASES = Path(__file__).resolve().parents[3] / "case_numpy.json"


@pytest.mark.skipif(not CASES.exists(), reason="Root-owned actual case files are not present")
@pytest.mark.parametrize("name", ["case_numpy.json", "case_numba.json", "case_numba_dt900.json"])
def test_actual_case_plan_only_import(name, monkeypatch):
    _no_native(monkeypatch)                                                                                       # plan-only never builds a native field object
    cfg, sha = C.load_config(CASES.parent / name)
    plan = D.plan_report(cfg, sha, I.prepare_inputs(cfg))
    assert plan["n_days"] == 160 and plan["n_wet_days"] == 11 and plan["rain"]["n_wet_rows"] == 26 and plan["rain"]["total_mm_decimal"] == "25.088"
    assert plan["planned_added_rain_s"] == 99000.0 and plan["planned_added_rain_hours"] == 27.5
    assert plan["nominal_elapsed_s"] == 13824000.0 and plan["planned_physical_end_s"] == 13923000.0
    assert plan["initial"]["point_theta"] == pytest.approx([0.0711, 0.0921, 0.0658]) and plan["initial"]["total_water_m"] == pytest.approx(0.0229424, abs=5e-8)
    assert plan["initial"]["layer_water_m"] == pytest.approx([0.00729522, 0.008624814, 0.007022366], rel=1e-8)
    assert plan["targets"]["valid_per_level"] == [56, 147, 79] and sum(plan["targets"]["valid_per_level"]) == 282
    assert plan["n_interpolated_air_half_hours"] == 36 and plan["matrix"]["peak_rain_exceeds_K"] is False
