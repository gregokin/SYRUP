"""Numerics, scoring and API tests of the evaporation-stress calibration helper. The synthetic recovery uses a toy forward function (REVIEW-ONLY mathematics); no native column is run and no model result is faked here."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarks.subsurface_stress_calibration import calibrate as C
from benchmarks.subsurface_stress_calibration import scoring as SC
from benchmarks.subsurface_stress_calibration import search as SE


# ------------------------------------------------------------------------------------------------------------- scoring
def test_residual_is_change_minus_change_and_equals_aligned_model():
    m, m0, o, o0 = np.array([0.20, 0.18, 0.15]), 0.22, np.array([0.30, 0.27, 0.24]), 0.32
    s = SC.sensor_score(m, m0, o, o0)
    expected = (m - m0) - (o - o0)
    assert np.allclose(s["residual"], expected, atol=1e-15)
    assert np.allclose(s["aligned_model"] - o, expected, atol=1e-15) and math.isclose(s["offset"], 0.10)
    assert math.isclose(s["rmse"], math.sqrt(np.mean(expected**2)))
    assert math.isclose(s["rmse_unshifted"], math.sqrt(np.mean((m - o) ** 2)))        # raw RMSE reported beside it
    assert math.isclose(s["zero_change_rmse"], math.sqrt(np.mean((o0 - o) ** 2)))


def test_constant_initial_bias_does_not_change_score():
    m, o = np.array([0.20, 0.18, 0.15, 0.14]), np.array([0.31, 0.26, 0.25, 0.21])
    a = SC.sensor_score(m, 0.22, o, 0.33)
    for bias in (-0.07, 0.0, 0.123):
        b = SC.sensor_score(m + bias, 0.22 + bias, o, 0.33)
        assert math.isclose(a["sse"], b["sse"], abs_tol=1e-15) and np.allclose(a["residual"], b["residual"], atol=1e-15)


def test_inputs_not_modified():
    m, o = np.array([0.2, 0.1]), np.array([0.3, np.nan])
    m_c, o_c = m.copy(), o.copy()
    SC.sensor_score(m, 0.25, o, 0.35)
    assert m.tobytes() == m_c.tobytes() and o.tobytes() == o_c.tobytes()


def test_missing_observations_counted_out_and_t0_not_in_series():
    o = np.array([0.30, np.nan, 0.28, np.nan, np.nan])
    s = SC.sensor_score(np.linspace(0.2, 0.1, 5), 0.22, o, 0.32)
    assert s["n"] == 2 and s["n_missing"] == 3 and np.isnan(s["residual"][[1, 3, 4]]).all()
    # t0 would add a forced zero residual: the series has length n (t >= 1) and the initial pair only fixes the offset
    full = SC.sensor_score(np.array([0.2]), 0.22, np.array([0.30]), 0.32)
    assert full["n"] == 1


def test_all_invalid_refused_and_bad_inputs():
    s = SC.sensor_score(np.array([0.2, 0.1]), 0.22, np.array([np.nan, np.nan]), 0.3)
    assert s["n"] == 0 and s["rmse"] is None
    with pytest.raises(SC.ScoreError, match="every observation is invalid"):
        SC.pooled_objective({"w": SC.window_score(np.array([[0.2, 0.1]]), [0.22], np.array([[np.nan, np.nan]]), [0.3])})
    with pytest.raises(SC.ScoreError):
        SC.sensor_score(np.array([0.2, np.nan]), 0.22, np.array([0.3, 0.2]), 0.3)          # non-finite model
    with pytest.raises(SC.ScoreError):
        SC.sensor_score(np.array([0.2]), 0.22, np.array([0.3, 0.2]), 0.3)                    # length mismatch
    with pytest.raises(SC.ScoreError):
        SC.sensor_score(np.array([0.2]), np.nan, np.array([0.3]), 0.3)                       # non-finite initial value
    with pytest.raises(SC.ScoreError):
        SC.sensor_score(np.array([0.2]), 0.2, np.array([0.3]), True)


def test_separate_offset_per_window_and_sensor():
    obs = np.array([[0.30, 0.28], [0.20, 0.19]])
    w = SC.window_score(np.array([[0.10, 0.08], [0.05, 0.04]]), [0.12, 0.06], obs, [0.32, 0.21])
    assert math.isclose(w["sensors"][0]["offset"], 0.20) and math.isclose(w["sensors"][1]["offset"], 0.15)
    assert w["sensors"][0]["offset"] != w["sensors"][1]["offset"]
    assert w["sse"] == pytest.approx(0.0, abs=1e-30)             # changes match exactly despite different offsets


def test_pooled_counts_not_mean_of_window_rmse():
    a = SC.window_score(np.array([[0.10] * 9]), [0.10], np.array([[0.20] * 9]), [0.20 - 0.0])      # zero change both -> residual 0
    b = SC.window_score(np.array([[0.30]]), [0.10], np.array([[0.10]]), [0.10])                    # one point, residual 0.2
    pooled = SC.pooled_objective({"a": a, "b": b})
    assert pooled["n"] == 10 and math.isclose(pooled["mse"], 0.04 / 10)
    assert not math.isclose(pooled["rmse"], (a["sse"] + math.sqrt(0.04)) / 2)                        # NOT the mean of window RMSEs
    assert pooled["rmse"] == pytest.approx(math.sqrt(0.004))


def test_breakdown_counts_and_bias():
    obs = np.array([[0.3, 0.2], [0.2, np.nan]])
    w1 = SC.window_score(np.array([[0.2, 0.2], [0.1, 0.1]]), [0.25, 0.15], obs, [0.35, 0.25])
    w2 = SC.window_score(np.array([[0.2, 0.2], [0.1, 0.1]]), [0.25, 0.15], obs, [0.35, 0.25])
    bd = SC.breakdown({"w1": w1, "w2": w2}, {"w1": "JORN", "w2": "MOAB"}, ["a", "b"])
    assert bd["window"]["w1"]["n"] == 3 and bd["site"]["JORN"]["n"] == 3 and bd["depth"]["a"]["n"] == 4 and bd["depth"]["b"]["n"] == 2
    assert bd["window_depth"]["w1/b"]["n"] == 1
    assert bd["site"]["JORN"]["bias_signed"] == pytest.approx(np.mean(np.concatenate([w1["sensors"][0]["residual"], w1["sensors"][1]["residual"]])[[0, 1, 2]]))


# --------------------------------------------------------------------------------------------------------------- search
def test_grid_includes_endpoints_and_supported_range():
    g = SE.dense_grid()
    assert g[0] == 0.010001 and g[-1] == 1.0 and g.size == SE.N_GRID and np.all(np.diff(g) > 0)


def test_local_minimum_runs_plateaus_and_ends():
    assert SE.local_minimum_runs([3, 1, 2, 0.5, 0.5, 2, 1, 4]) == [(1, 1), (3, 4), (6, 6)]
    assert SE.local_minimum_runs([1, 2, 3]) == [(0, 0)] and SE.local_minimum_runs([3, 2, 1]) == [(2, 2)]
    assert SE.local_minimum_runs([5, 5, 5]) == [(0, 2)]


def test_cache_runs_each_stress_once():
    calls = []
    f = SE.CachedObjective(lambda x: calls.append(x) or (x - 0.4) ** 2)
    f(0.3), f(0.3), f(0.5)
    assert calls == [0.3, 0.5] and f.order == [0.3, 0.5]
    with pytest.raises(SE.SearchError):
        SE.CachedObjective(lambda x: float("nan"))(0.2)


def test_synthetic_known_stress_recovery_with_initial_bias():
    """REVIEW-ONLY toy forward model: theta(t) decays with a stress-dependent rate; the observation is the toy at the true stress plus an arbitrary initial bias. The pooled change score recovers the stress."""
    t = np.arange(1, 9)

    def toy(s, bias=0.0):
        rate = 0.02 * (1.0 + 4.0 * s)
        return 0.30 + bias - np.cumsum(np.full(t.size, rate))

    true, bias = 0.37, 0.081
    obs0, obs = 0.33, toy(true)[:] + (0.33 - 0.30)
    def objective(s):
        w = SC.window_score(toy(s, bias)[None, :], [0.30 + bias], obs[None, :], [obs0])
        return SC.pooled_objective({"w": w})["mse"]
    res = SE.search(objective)
    assert res["selected"]["stress"] == pytest.approx(true, abs=1e-6) and res["selected"]["objective"] < 1e-18
    assert not res["boundary_solution"] and res["n_local_minima"] == 1


def test_multiple_minima_and_global_choice():
    f = lambda x: min((x - 0.2) ** 2 + 0.05, (x - 0.7) ** 2 + 0.01)
    res = SE.search(f)
    assert res["multiple_minima"] and res["n_local_minima"] == 2
    assert res["selected"]["stress"] == pytest.approx(0.7, abs=1e-6) and res["selected"]["objective"] == pytest.approx(0.01)
    other = next(m for m in res["minima"] if m is not res["selected"])
    assert other["stress"] == pytest.approx(0.2, abs=1e-6)


def test_boundary_and_flat_reporting():
    res = SE.search(lambda x: x)                                   # monotone: optimum at the lower bound
    assert res["boundary_solution"] and res["selected"]["at_lower_bound"] and res["selected"]["stress"] == SE.STRESS_MIN
    res = SE.search(lambda x: -x)
    assert res["selected"]["at_upper_bound"] and res["selected"]["stress"] == 1.0
    flat = SE.search(lambda x: 1.0)
    assert flat["selected"]["flat_plateau"] and flat["selected"]["n_grid_points_in_plateau"] == SE.N_GRID


def test_golden_fallback_matches_scipy(monkeypatch):
    f = lambda x: (x - 0.31) ** 2
    assert SE.golden(f, 0.0, 1.0) == pytest.approx(0.31, abs=1e-8)
    monkeypatch.setitem(sys.modules, "scipy.optimize", None)
    x, how = SE.refine_bracket(f, 0.0, 1.0)
    assert how == "golden_section" and x == pytest.approx(0.31, abs=1e-8)


def test_near_optimal_intervals_descriptive():
    x = np.linspace(0.0, 1.0, 11)
    out = SE.near_optimal_intervals(x, 1.0 + (x - 0.5) ** 2 * 10, fractions=(0.1,))
    assert out["intervals"]["0.1"] == [pytest.approx([0.4, 0.6])] and "not a confidence" in out["descriptive_only"]
    with pytest.raises(SE.SearchError):
        SE.near_optimal_intervals([0.2, 0.1], [1.0, 2.0])


# ------------------------------------------------------------------------------------------------------- config / CLI
def test_fits_frozen_and_configs_valid_without_raw_data():
    assert C.FITS == {"main": ("harmonic", 0.05), "secondary": ("geometric", 0.05), "sensitivity_d0": ("harmonic", 0.0)}
    cfgs = C.build_configs("harmonic", 0.05, stress=0.25)
    assert set(cfgs) == set(C.WINDOWS)
    for c in cfgs.values():
        assert c["layers"]["theta_evaporation_stress"] == 0.25 and len(c["scenarios"]) == 1
        s = c["scenarios"][0]
        assert (s["evaporation_factor"], s["diffusion_coefficient"], s["bottom"], s["potential_et"]) == (1.0, 0.05, "open", "weather")
        assert c["pins"]["expected_sources"] == C.EXPECTED_SOURCES and c["provenance"]["stress_calibration"]["T"] == 0
    site = C.build_configs("harmonic", 0.05, stress=None)
    assert site["JORN4_A"]["layers"]["theta_evaporation_stress"] == site["JORN4_A"]["provenance"]["stress_calibration"]["site_theta1500"]


def test_with_stress_changes_only_stress():
    from benchmarks.subsurface_validation import drydown_driver as dd

    c = dd.validate_config(C.build_configs("geometric", 0.05, 0.3)["MOAB2_C"])
    c2 = C.with_stress(c, 0.6)
    assert c2["layers"]["theta_evaporation_stress"] == 0.6 and c["layers"]["theta_evaporation_stress"] == 0.3
    assert all(np.array_equal(c["layers"][k], c2["layers"][k]) for k in ("edges_m", "theta_field_capacity", "interface_conductivity_m_s"))


def test_plan_writes_configs_and_refuses_nonempty(tmp_path):
    out = tmp_path / "o"
    proto = C.plan(out)
    assert len(proto["config_sha256"]) == 3 * 2 * 3 and (out / "protocol.json").exists()
    assert json.loads((out / "configs" / "main_baseline_0.18_JORN4_A.json").read_text())["layers"]["theta_evaporation_stress"] == 0.18
    with pytest.raises(C.CalibrationError, match="not empty"):
        C.plan(out)
    assert C.main(["--out", str(out)]) == 2
    assert C.main(["--out", str(tmp_path / "x"), "--verify-cuda"]) == 2            # needs --execute


def test_window_score_from_run_uses_probe_layer_initial_mean_and_point_initial():
    """Adapter check with a synthetic prepared window/run (no column): model initial = integrated layer mean of the probe's layer, observation initial = raw point value."""
    cfg = {"layers": {"edges_m": np.array([0.0, 0.1, 0.2, 0.3])}}
    prep = {"sensor_depths_m": np.array([0.05, 0.15, 0.25]), "theta0_points": np.array([0.30, 0.25, 0.20]), "initial": {"layer_theta": np.array([0.28, 0.24, 0.22])},
            "obs": {"501": np.array([0.29, np.nan]), "502": np.array([0.24, 0.23]), "503": np.array([0.19, 0.18])}}
    lt = np.array([[0.27, 0.235, 0.215], [0.26, 0.23, 0.21]])
    w = C.window_score_from_run(cfg, prep, {"layer_theta": lt})
    s0 = w["sensors"][0]
    assert s0["n"] == 1 and math.isclose(s0["offset"], 0.30 - 0.28)
    assert np.allclose(s0["residual"][0], (0.27 - 0.28) - (0.29 - 0.30))
    assert math.isclose(w["sensors"][2]["offset"], 0.20 - 0.22)


# --------------------------------------------------------------------------------------------- correction 01 regressions
def test_breakdown_aggregates_by_sensor_id_and_keeps_actual_depths():
    obs = np.array([[0.3, 0.2], [0.2, 0.1], [0.1, 0.05]])
    ws = SC.window_score(np.array([[0.2, 0.2], [0.1, 0.1], [0.05, 0.05]]), [0.25, 0.15, 0.08], obs, [0.35, 0.25, 0.12])
    cfgs = {"JORN4_A": {"site": {"site": "JORN"}}, "MOAB2_C": {"site": {"site": "MOAB"}}}
    preps = {"JORN4_A": {"sensor_depths_m": np.array([0.058, 0.158, 0.258]), "obs": dict(zip(("501", "502", "503"), obs))},
             "MOAB2_C": {"sensor_depths_m": np.array([0.055, 0.155, 0.255]), "obs": dict(zip(("501", "502", "503"), obs))}}
    rec = C.score_record(cfgs, preps, {"JORN4_A": ws, "MOAB2_C": ws})
    assert set(rec["breakdown"]["depth"]) == {"501", "502", "503"}
    assert "MOAB2_C/501" in rec["breakdown"]["window_depth"]
    assert rec["windows"]["MOAB2_C"]["depth_labels"] == ["501@0.055m", "502@0.155m", "503@0.255m"]
    assert rec["windows"]["JORN4_A"]["depth_labels"][0] == "501@0.058m"
    assert not any("0.058" in lab for lab in rec["windows"]["MOAB2_C"]["depth_labels"])
    assert rec["windows"]["MOAB2_C"]["sensors"][0]["depth_m"] == 0.055


def test_new_config_provenance_is_truthful():
    shared = C.build_configs("harmonic", 0.05, stress=0.25)
    site = C.build_configs("harmonic", 0.05, stress=None)
    for w in C.WINDOWS:
        p = shared[w]["provenance"]
        assert p["role"] == "calibration_all_windows" and p["historical_template_role"] in ("train", "heldout")
        assert "all three windows" in p["evaporation_stress"] and "theta1500 (no-refit" not in p["evaporation_stress"]
        assert p["stress_calibration"]["stress_source"] == "calibrated shared constant"
        q = site[w]["provenance"]
        assert q["role"] == "calibration_all_windows" and "site-specific" in q["evaporation_stress"] and q["stress_calibration"]["stress_source"].startswith("site-specific")
        assert "heldout" not in shared[w]["name"] and "train" not in shared[w]["name"]


def test_selection_is_minimum_over_every_evaluated_point(monkeypatch):
    def bad_refine(fn, a, b, xatol=1e-10):
        fn(0.5025)                                                  # an intermediate evaluation (inside the narrow well) that beats the returned candidate
        return a, "scipy_bounded"
    monkeypatch.setattr(SE, "refine_bracket", bad_refine)
    f = lambda x: (x - 0.5) ** 2 - (1.0 if abs(x - 0.5025) < 0.0005 else 0.0)
    res = SE.search(f)
    cache_best = min(res["profile"]["objective"])
    assert res["selected"]["objective"] == cache_best
    assert res["selected"]["stress"] == 0.5025 and res["selected"]["objective"] < min(m["objective"] for m in res["minima"])
    xs = res["profile"]["stress"]
    assert res["selected"]["stress"] == xs[res["profile"]["objective"].index(cache_best)]
    assert {"at_lower_bound", "at_upper_bound", "on_grid", "flat_plateau"} <= set(res["selected"])


def test_selection_tie_goes_to_smaller_stress():
    res = SE.search(lambda x: 1.0)
    assert res["selected"]["stress"] == SE.STRESS_MIN


def test_failed_scipy_refinement_is_not_reported_successful(monkeypatch):
    class Bad:
        success, x, fun = False, 0.9, float("nan")

    import scipy.optimize as so

    monkeypatch.setattr(so, "minimize_scalar", lambda *a, **k: Bad())
    x, how = SE.refine_bracket(lambda x: (x - 0.31) ** 2, 0.0, 1.0)
    assert how == "golden_section_after_scipy_failure" and x == pytest.approx(0.31, abs=1e-8)


def test_run_record_serializes_cumulative_budget_without_model_run():
    from maple_syrup import subsurface as ss

    cum = ss.CumulativeBudget(n_steps=2, elapsed_s=172800.0, added_m3=0.0, surface_return_m3=0.0, drainage_m3=1e-4, evaporation_m3=2e-3, evaporation_unmet_m3=3e-5)
    run = {"trajectory": {"t_s": np.array([1.0, 2.0])}, "layer_theta": np.zeros((2, 3)), "initial_storage_m": 0.1, "final_storage_m": 0.09, "cumulative": cum, "ledger": {"residual_m3": 0.0}}
    rec = C.run_record(run)
    assert rec["cumulative_budget"]["drainage_m3"] == 1e-4 and rec["cumulative_budget"]["evaporation_m3"] == 2e-3 and rec["cumulative_budget"]["n_steps"] == 2
    json.dumps(rec, allow_nan=False)
