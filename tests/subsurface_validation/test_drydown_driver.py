"""Driver tests. The property values here are TEST FIXTURES that exercise the code paths (config strictness, the actual column, ledgers, controls, scoring); they are not field values and no result of this file is a finding."""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarks.subsurface_validation import drydown_driver as dd
from benchmarks.subsurface_validation import neon_inputs as ni
from benchmarks.subsurface_validation import paths as P

INV_SHA = hashlib.sha256(P.INVENTORY.read_bytes()).hexdigest()
AIR_SHA = hashlib.sha256((P.AIR_JORN / "manifest.json").read_bytes()).hexdigest() if (P.AIR_JORN / "manifest.json").is_file() else "0" * 64
FIXTURE_CFG = {
    "schema": dd.SCHEMA, "name": "TEST FIXTURE (not field values)", "provenance": {"properties": "test fixture"},
    "site": {"key": "JORN_plot4_2025-04", "raw_section": "JORN", "site": "JORN", "hor": "004", "plot": 4, "latitude_deg": 32.59052, "latitude_source": "test fixture"},
    "air": {"root": "AIR_JORN", "month": "2025-04", "level": "000.020"},
    "window": {"run_start_utc": "2025-04-05T21:00:00Z", "min_dry_hours": 120, "exclusion_hours": 72, "anchor": "first_joint", "boundary_second_of_day": 25200, "max_days": 3, "calendar_utc_offset_hours": -7},
    "layers": {"edges_m": [0.0, 0.1, 0.2, 0.3], "theta_residual": [0.02, 0.02, 0.02], "theta_field_capacity": [0.12, 0.12, 0.12], "theta_saturation": [0.40, 0.40, 0.40], "theta_evaporation_stress": 0.08,
               "interface_conductivity_m_s": [1e-6, 1e-6]},
    "scenarios": [{"name": "zeroET_d0", "diffusion_coefficient": 0.0, "evaporation_factor": 1.0, "potential_et": "zero", "bottom": "open"},
                  {"name": "wx_d0.05", "diffusion_coefficient": 0.05, "evaporation_factor": 1.0, "potential_et": "weather", "bottom": "closed"}],
    "controls": {"backends": ["numpy"], "numba_rtol": 2e-12, "numba_atol": 1e-14, "restart_after_day": 2},
    "pins": {"depth_table_sha256": "0" * 64, "inventory_sha256": INV_SHA, "inventory_candidate_key": "JORN_plot4_2025-04", "air_manifest_sha256": AIR_SHA,
             "expected_sources": {"maple_syrup": {"package_sha256": "37d89f23dbf60626c85f85ea5dd6d1734d3bc7a4aa61d8f2fdfc29f237bcfbd0", "files": 180}, "maple": {"package_sha256": "fbc504bbcd1ec0c02e6a96089134a06187532455ef7ba5e6f146801c8313c02e", "files": 412}}},
}


def cfg_with(**edits):
    c = copy.deepcopy(FIXTURE_CFG)
    for path, v in edits.items():
        d = c
        keys = path.split("__")
        for k in keys[:-1]:
            d = d[int(k)] if isinstance(d, list) else d[k]
        if isinstance(d, list):
            d[int(keys[-1])] = v
        else:
            d[keys[-1]] = v
    return c


# ---------------------------------------------------------------- config strictness
def test_a_null_property_is_refused_by_name():
    for path in ("layers__theta_residual", "layers__theta_evaporation_stress", "layers__interface_conductivity_m_s", "layers__theta_field_capacity", "layers__theta_saturation"):
        with pytest.raises(dd.ConfigError, match="no physical property or choice has a default"):
            dd.validate_config(cfg_with(**{path: None}))
    with pytest.raises(dd.ConfigError, match="diffusion_coefficient"):
        dd.validate_config(cfg_with(scenarios__0__diffusion_coefficient=None))
    with pytest.raises(dd.ConfigError, match="latitude"):
        dd.validate_config(cfg_with(site__latitude_deg=None))


def test_unknown_missing_wrong_or_bool_keys_and_values_are_refused():
    c = copy.deepcopy(FIXTURE_CFG)
    c["extra"] = 1
    with pytest.raises(dd.ConfigError, match="keys must be exactly"):
        dd.validate_config(c)
    c = copy.deepcopy(FIXTURE_CFG)
    c["execution_approved_by_root"] = True                          # the permanent approval field no longer exists: a leftover key is refused
    with pytest.raises(dd.ConfigError, match="keys must be exactly"):
        dd.validate_config(c)
    for edit in ({"layers__edges_m": [0.0, 0.1, 0.1, 0.3]}, {"layers__theta_residual": [0.02, 0.02]}, {"layers__theta_saturation": [0.4, 0.4, 1.5]}, {"scenarios__0__bottom": "sealed"},
                 {"scenarios__0__potential_et": "plant"}, {"controls__backends": ["tpu"]}, {"controls__backends": ["numpy", "numpy"]}, {"air__level": "000.010"}, {"window__anchor": "nearest"},
                 {"window__max_days": 0}, {"window__max_days": True}, {"window__max_days": 2.0}, {"controls__restart_after_day": True}, {"controls__restart_after_day": 0}, {"site__plot": True},
                 {"window__calendar_utc_offset_hours": None}, {"window__calendar_utc_offset_hours": 7.5}, {"window__calendar_utc_offset_hours": True}, {"scenarios__1__name": "zeroET_d0"},
                 {"pins__inventory_sha256": "abc"}, {"pins__air_manifest_sha256": None}, {"pins__inventory_candidate_key": "MOAB_plot2_2025-05"}, {"controls__numba_rtol": None},
                 {"pins__expected_sources": {"maple_syrup": {"package_sha256": "0" * 64, "files": 1}}}, {"provenance": {}}):
        with pytest.raises(dd.ConfigError):
            dd.validate_config(cfg_with(**edit))
    assert dd.validate_config(copy.deepcopy(FIXTURE_CFG))["layers"]["edges_m"].tolist() == [0.0, 0.1, 0.2, 0.3]


def _key_mismatch():
    """A config whose pinned candidate key is right but whose `site.key` is another string (validate_config alone requires equality, so the mismatch is injected after validation)."""
    c = dd.validate_config(copy.deepcopy(FIXTURE_CFG))
    c["site"] = {**c["site"], "key": "JORN_plot4_2025-05"}
    return c


def test_the_site_key_itself_is_cross_checked_against_the_pinned_candidate():
    inv = json.loads(P.INVENTORY.read_text())
    with pytest.raises(ni.InputError, match="site_key"):
        dd.cross_check_inventory(_key_mismatch(), inv)


def test_the_inventory_candidate_is_actually_cross_checked():
    inv = json.loads(P.INVENTORY.read_text())
    ok = dd.cross_check_inventory(dd.validate_config(copy.deepcopy(FIXTURE_CFG)), inv)
    assert all(ok.values())
    for edit in ({"site__key": "JORN_plot4_2025-05"}, {"site__site": "MOAB"}, {"site__raw_section": "MOAB"}, {"site__hor": "002"}, {"air__month": "2025-05"}, {"air__root": "AIR_MOAB"}, {"window__run_start_utc": "2025-04-06T00:00:00Z"}, {"site__plot": 2}):
        with pytest.raises(ni.InputError, match="disagrees with the pinned inventory"):
            dd.cross_check_inventory(dd.validate_config(cfg_with(**edit)) if "site__key" not in edit else _key_mismatch(), inv)


def test_expected_source_identities_are_asserted_not_just_unchanged():
    cfg = dd.validate_config(copy.deepcopy(FIXTURE_CFG))
    good = {k: {"package_sha256": v["package_sha256"], "files": v["files"], "file": "x"} for k, v in cfg["pins"]["expected_sources"].items()}
    dd.assert_expected_sources(cfg, good)
    for name, field, val in (("maple_syrup", "package_sha256", "1" * 64), ("maple", "files", 411)):
        bad = copy.deepcopy(good)
        bad[name][field] = val
        with pytest.raises(dd.ControlFailure, match="is not the declared"):
            dd.assert_expected_sources(cfg, bad)


# ---------------------------------------------------------------- the actual column, synthetic prepared inputs
def synthetic_prep(n_days=4, theta=(0.06, 0.07, 0.06)):
    edges = np.array([0.0, 0.1, 0.2, 0.3])
    init = ni.integrate_initial_profile([0.058, 0.158, 0.258], theta, edges)
    ends = ni.daily_end_times(ni.parse_utc("2025-04-08T22:00:00Z"), n_days)
    return {"n_days": n_days, "initial": init, "end_times_s": ends, "pet_mm_day": [5.0, 5.5, 4.0, 6.0][:n_days], "sensor_depths_m": np.array([0.058, 0.158, 0.258]), "theta0_points": np.array(theta),
            "obs": {"501": np.full(n_days, 0.05), "502": np.full(n_days, 0.07), "503": np.full(n_days, 0.06), "joint": np.ones(n_days, dtype=bool)}}


@pytest.fixture(scope="module")
def cfg():
    return dd.validate_config(copy.deepcopy(FIXTURE_CFG))


def test_zero_et_zero_exchange_open_bottom_below_field_capacity_holds_the_state(cfg):
    prep = synthetic_prep()
    run = dd.run_column(cfg, prep, cfg["scenarios"][0], "numpy")
    assert run["trajectory"]["water_m"][-1] == pytest.approx(prep["initial"]["layer_water_m"], abs=1e-15)
    assert run["cumulative"].evaporation_m3 == 0.0 and run["cumulative"].drainage_m3 == 0.0 and abs(run["ledger"]["residual_m3"]) <= run["ledger"]["bound_m3"]


def test_weather_evaporation_removes_water_only_from_the_top_layer_and_the_native_ledger_closes(cfg):
    prep = synthetic_prep()
    scen = {**cfg["scenarios"][1], "diffusion_coefficient": 0.0, "bottom": "open"}
    run = dd.run_column(cfg, prep, scen, "numpy")
    assert run["initial_storage_m"] - run["final_storage_m"] == pytest.approx(run["cumulative"].evaporation_m3, rel=1e-12) and run["cumulative"].evaporation_m3 > 0.0
    lg = run["ledger"]
    assert abs(lg["residual_m3"]) <= lg["bound_m3"] and lg["n_terms"] == 4 * 3 * 1 * 4 + 7                                    # the native 4 L N n_steps + 7 terms
    tr = run["trajectory"]
    assert np.all(np.abs(tr["step_residual_m3"]) <= tr["step_bound_m3"])
    assert tr["water_m"][-1][1:] == pytest.approx(prep["initial"]["layer_water_m"][1:], abs=1e-15) and tr["water_m"][-1][0] < prep["initial"]["layer_water_m"][0]


def test_cumulative_ledger_uses_the_native_bound_and_a_zero_bound_with_a_residual_fails():
    from maple_syrup import subsurface as ss

    c0 = ss.CumulativeBudget()
    ok = dd.check_cumulative_ledger(0.01, 0.01, c0, 1, 3, 1)
    assert ok["bound_m3"] > 0 and ok["residual_m3"] == 0.0
    with pytest.raises(dd.ControlFailure, match="exceeds the native FP64 bound"):
        dd.check_cumulative_ledger(0.0, 1e-9, c0, 1, 3, 1)                              # scale = 1e-9 but the residual is the whole final storage: far above the bound
    drift = ss.CumulativeBudget(1, 86400.0, 0.0, 0.0, 0.0, 0.5e-3, 0.0)
    with pytest.raises(dd.ControlFailure):
        dd.check_cumulative_ledger(0.01, 0.0095, drift, 1, 3, 1)                         # 0.0005 lost per the ledger but 0.0005 evaporated: closes; then break it
        dd.check_cumulative_ledger(0.01, 0.0094, drift, 1, 3, 1)
    with pytest.raises(dd.ControlFailure):
        dd.check_cumulative_ledger(0.01, 0.0094, drift, 1, 3, 1)
    with pytest.raises(dd.ControlFailure):
        dd.check_cumulative_ledger(0.0, float("nan"), c0, 1, 3, 1)


def test_a_zero_bound_with_a_nonzero_residual_fails(monkeypatch):
    from maple_syrup import conservation
    from maple_syrup import subsurface as ss

    monkeypatch.setattr(conservation, "volume_roundoff_bound_m3", lambda n, scale: 0.0)
    with pytest.raises(dd.ControlFailure):
        dd.check_cumulative_ledger(0.01, 0.0099, ss.CumulativeBudget(), 1, 3, 1)


def test_exchange_moves_water_between_layers_without_creating_any(cfg):
    prep = synthetic_prep(theta=(0.03, 0.09, 0.06))
    run = dd.run_column(cfg, prep, {**cfg["scenarios"][0], "diffusion_coefficient": 0.05, "potential_et": "zero"}, "numpy")
    assert run["final_storage_m"] == pytest.approx(run["initial_storage_m"], rel=1e-13)
    w = run["trajectory"]["water_m"][-1]
    assert w[0] > prep["initial"]["layer_water_m"][0] and w[1] < prep["initial"]["layer_water_m"][1]
    assert np.any(run["trajectory"]["diffusion_m"] != 0.0)


def test_midpoint_restart_compares_the_whole_trajectory_and_cumulative_budget_and_raises_on_mismatch(cfg):
    prep = synthetic_prep()
    scen = cfg["scenarios"][1]
    straight = dd.run_column(cfg, prep, scen, "numpy")
    restarted = dd.run_column(cfg, prep, scen, "numpy", restart_after_day=2)
    assert restarted["restart_applied"] and "NOT a production lifecycle checkpoint" in restarted["restart_label"]
    dd.assert_runs_identical(straight, restarted, "restart")
    for key in restarted["trajectory"]:                                                          # every recorded array, perturbed by ONE ulp in one element, must be caught
        bad = copy.deepcopy(restarted)
        arr = bad["trajectory"][key].copy()
        arr.flat[-1] = np.nextafter(arr.flat[-1], np.inf)
        bad["trajectory"][key] = arr
        with pytest.raises(dd.ControlFailure, match="differs"):
            dd.assert_runs_identical(straight, bad, "restart")
    import dataclasses

    bad = copy.deepcopy(restarted)
    bad["cumulative"] = dataclasses.replace(bad["cumulative"], evaporation_m3=bad["cumulative"].evaporation_m3 * (1 + 1e-12))
    with pytest.raises(dd.ControlFailure, match="cumulative"):
        dd.assert_runs_identical(straight, bad, "restart")


def test_backend_agreement_checks_trajectories_under_the_inherited_contract_and_raises_on_injected_failures(cfg):
    pytest.importorskip("numba")
    prep = synthetic_prep()
    scen = cfg["scenarios"][1]
    a = dd.run_column(cfg, prep, scen, "numpy")
    b = dd.run_column(cfg, prep, scen, "numba")
    worst = dd.assert_backends_agree(a, b, rtol=2e-12, atol=1e-14, bitwise=False, what="x")
    assert set(worst) >= {"water_m", "percolation_m", "diffusion_m", "evaporation_m"}
    for key in ("percolation_m", "diffusion_m", "evaporation_m", "drainage_m"):
        bad = copy.deepcopy(b)
        bad["trajectory"][key] = bad["trajectory"][key] + 1e-9
        with pytest.raises(dd.ControlFailure, match="differs beyond"):
            dd.assert_backends_agree(a, bad, rtol=2e-12, atol=1e-14, bitwise=False, what="x")
    bad = copy.deepcopy(b)
    bad["trajectory"]["t_s"] = bad["trajectory"]["t_s"] + 1.0
    with pytest.raises(dd.ControlFailure, match="time differs"):
        dd.assert_backends_agree(a, bad, rtol=2e-12, atol=1e-14, bitwise=False, what="x")
    ulp = copy.deepcopy(a)
    ulp["trajectory"]["water_m"] = np.nextafter(a["trajectory"]["water_m"], 1.0)                      # within rtol, but NOT bitwise
    dd.assert_backends_agree(a, ulp, rtol=2e-12, atol=1e-14, bitwise=False, what="x")
    with pytest.raises(dd.ControlFailure, match="bitwise"):
        dd.assert_backends_agree(a, ulp, rtol=2e-12, atol=1e-14, bitwise=True, what="x")


def test_numba_is_bitwise_equal_when_the_diffusion_coefficient_is_zero(cfg):
    pytest.importorskip("numba")
    prep = synthetic_prep()
    scen = {**cfg["scenarios"][1], "diffusion_coefficient": 0.0}
    dd.assert_backends_agree(dd.run_column(cfg, prep, scen, "numpy"), dd.run_column(cfg, prep, scen, "numba"), rtol=2e-12, atol=1e-14, bitwise=True, what="d0")


def test_initial_state_at_or_above_saturation_is_refused_not_clipped(cfg):
    dd.check_initial_state([0.1, 0.1, 0.1], [0.26, 0.26, 0.26])
    for theta in ([0.26, 0.1, 0.1], [0.3, 0.1, 0.1], [0.1, 0.1, float("nan")]):
        with pytest.raises(ni.InputError, match="strictly below saturation|saturation"):
            dd.check_initial_state(theta, [0.26, 0.26, 0.26])
    prep = synthetic_prep(theta=(0.45, 0.07, 0.06))                                    # layer 1 above the fixture saturation 0.40
    with pytest.raises(ni.InputError):
        dd.run_column(cfg, prep, cfg["scenarios"][0], "numpy")


def test_run_counts_reject_bool_and_zero(cfg):
    prep = synthetic_prep()
    with pytest.raises(dd.ConfigError):
        dd.run_column(cfg, prep, cfg["scenarios"][0], "numpy", restart_after_day=True)
    with pytest.raises(dd.ConfigError):
        dd.run_column(cfg, prep, cfg["scenarios"][0], "numpy", restart_after_day=0)


# ---------------------------------------------------------------- scoring
def test_primary_overall_rmse_is_sqrt_of_the_mean_per_layer_mse_and_pooled_is_separate():
    per = {"501": dd.score(np.array([0.05, 0.05]), np.array([0.04, 0.04])), "502": dd.score(np.array([0.06]), np.array([0.06])), "503": dd.score(np.array([0.05, 0.05, 0.05]), np.array([0.04, 0.05, 0.06]))}
    mses = [per[v]["mse"] for v in per]
    assert dd.overall_rmse(per) == pytest.approx(np.sqrt(np.mean(mses)))
    preds = {"501": np.array([0.05, 0.05]), "502": np.array([0.06]), "503": np.array([0.05, 0.05, 0.05])}
    obs = {"501": np.array([0.04, 0.04]), "502": np.array([0.06]), "503": np.array([0.04, 0.05, 0.06])}
    assert dd.pooled_rmse(preds, obs) != pytest.approx(dd.overall_rmse(per))                  # unequal n: the two definitions differ and are labelled apart
    per["502"] = dd.score(np.array([np.nan]), np.array([0.06]))
    assert dd.overall_rmse(per) is None                                                      # a layer without a valid target makes the overall score undefined, not invented


def test_skill_is_undefined_for_a_zero_or_missing_reference():
    assert dd.skill(0.5, 1.0) == pytest.approx(0.5) and dd.skill(0.5, 0.0) is None and dd.skill(None, 1.0) is None and dd.skill(0.5, None) is None


def test_scores_use_only_valid_targets_and_two_persistence_nulls_are_reported(cfg):
    s = dd.score(np.array([0.05, 0.04, np.nan, 0.03]), np.array([0.05, np.nan, 0.05, 0.05]))
    assert s["n"] == 2 and s["bias"] == pytest.approx(-0.01) and s["rmse"] == pytest.approx(np.sqrt(0.0002))
    prep = synthetic_prep()
    pers = dd.persistence_runs(cfg, prep)
    assert set(pers) == {"initial_point_values_held", "initial_layer_means_held"}
    assert pers["initial_point_values_held"]["per_layer"]["501"]["n"] == 4
    assert pers["initial_point_values_held"]["overall_rmse"] != pers["initial_layer_means_held"]["overall_rmse"]


# ---------------------------------------------------------------- real inputs, plan only (no column run)
needs_data = pytest.mark.skipif(not (P.RAW_ROOT.is_dir() and (P.AIR_JORN / "manifest.json").is_file()), reason="the pinned data tree or the Root air retrieval is absent")


def _real_cfg(**edits):
    inv = json.loads(P.INVENTORY.read_text())
    c = cfg_with(**edits)
    c["pins"]["depth_table_sha256"] = inv["corrected_depth_table"]["sha256"]
    c["window"]["max_days"] = None
    return dd.validate_config(c)


@needs_data
def test_prepare_inputs_on_the_real_jorn_4_a_window_plan_only():
    c = _real_cfg()
    prep = dd.prepare_inputs(c)
    plan = dd.plan_report(c, prep)
    assert plan["inventory_files_rehashed"] == 21 and plan["air_manifest_files_rehashed"] == 12 and all(plan["inventory_cross_checks"].values())
    assert plan["anchor_utc"] == "2025-04-08T22:00:00Z" and plan["max_rainfree_days"] == 13
    assert plan["complete_forcing_days"] < plan["max_rainfree_days"] and plan["first_incomplete_forcing_day"]["reason"] == "1 of 48 records fail QA"
    assert plan["model_days"] == plan["complete_forcing_days"] == 10 and len(plan["pet_mm_day"]) == 10 and all(p > 0 for p in plan["pet_mm_day"])
    assert plan["sensor_depths_m"] == pytest.approx([0.058, 0.158, 0.258]) and plan["probe_layer_index"] == [0, 1, 2]
    assert plan["initial_total_water_m"] == pytest.approx(prep["initial"]["layer_water_m"].sum(), rel=1e-14)
    assert plan["pet"]["terrain_factor"] == 1.0 and plan["pet"]["date_rule"].endswith("UTC-7") and "representativeness" in plan["support_note"]


@needs_data
def test_inventory_and_air_manifest_pins_are_enforced():
    with pytest.raises(ni.InputError, match="inventory file differs"):
        dd.prepare_inputs(_real_cfg(pins__inventory_sha256="1" * 64))
    with pytest.raises(ni.InputError, match="air retrieval manifest differs"):
        dd.prepare_inputs(_real_cfg(pins__air_manifest_sha256="2" * 64))


@needs_data
def test_a_window_with_no_forcing_supported_day_is_refused_before_any_run(monkeypatch):
    real = ni.aggregate_daily_air

    def first_day_incomplete(series, starts):
        out = real(series, starts)
        out[0] = ni.DayWeather(out[0].window_start_s, out[0].window_end_s, 47, False, "injected: 1 of 48 records fail QA", None, None, None)
        return out

    monkeypatch.setattr(ni, "aggregate_daily_air", first_day_incomplete)
    with pytest.raises(ni.InputError, match="zero modelled days"):
        dd.prepare_inputs(_real_cfg())


# ---------------------------------------------------------------- execute() wiring on synthetic prepared inputs (fixture properties; plan_report stubbed because it needs real data)
def test_execute_runs_all_controls_scores_both_persistences_and_checks_sources(monkeypatch):
    pytest.importorskip("numba")
    c = dd.validate_config(cfg_with(controls__backends=["numpy", "numba"], controls__restart_after_day=2))
    prep = synthetic_prep()
    monkeypatch.setattr(dd, "plan_report", lambda cfg, p: {"stub": True})
    good = {k: {"package_sha256": v["package_sha256"], "files": v["files"], "file": "x"} for k, v in c["pins"]["expected_sources"].items()}
    monkeypatch.setattr(dd, "source_receipt", lambda: good)
    res = dd.execute(c, "abc", prep)
    assert set(res["runs"]) == {"zeroET_d0", "wx_d0.05"} and set(res["persistence"]) == {"initial_point_values_held", "initial_layer_means_held"}
    r = res["runs"]["wx_d0.05"]
    assert r["restart"]["whole_trajectory_and_cumulative_identical"] and r["backend_agreement_max_abs"] is not None and r["cumulative_ledger"]["n_terms"] == 4 * 3 + 7 + 4 * 3 * 3
    assert "skill_vs_initial_point_values_held" in res["scores"]["wx_d0.05"] and res["scores"]["wx_d0.05"]["per_layer"]["501"]["n"] == 4
    bad = {**good, "maple": {**good["maple"], "files": 1}}
    monkeypatch.setattr(dd, "source_receipt", lambda: bad)
    with pytest.raises(dd.ControlFailure, match="is not the declared"):
        dd.execute(c, "abc", prep)
    monkeypatch.setattr(dd, "source_receipt", lambda: good)
    with pytest.raises(dd.ConfigError, match="inside the modelled days"):
        dd.execute(dd.validate_config(cfg_with(controls__restart_after_day=9)), "abc", prep)


# ---------------------------------------------------------------- correction 02: CUDA support (host-only tests; the device tests below are gated and are NOT run by the author)
def test_cuda_is_a_declared_backend_and_the_driver_import_is_lazy():
    assert dd.validate_config(cfg_with(controls__backends=["numpy", "numba", "cuda"]))["controls"]["backends"] == ["numpy", "numba", "cuda"]
    import subprocess

    code = f"import sys; sys.path.insert(0, {str(Path(__file__).resolve().parents[2])!r}); import benchmarks.subsurface_validation.drydown_driver; assert 'cupy' not in sys.modules; print('ok')"
    out = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, check=False)
    assert out.stdout.strip() == "ok", out.stderr


def test_a_cuda_column_without_cupy_or_a_device_raises_and_never_falls_back(cfg):
    import importlib.util

    from maple_syrup import subsurface as ss

    if importlib.util.find_spec("cupy") is not None:
        pytest.skip("CuPy is importable here: the unavailable-backend control needs a CuPy-free interpreter")
    with pytest.raises(ss.SubsurfaceBackendUnavailableError, match="no fallback"):
        dd.run_column(cfg, synthetic_prep(), cfg["scenarios"][0], "cuda")
    c = dd.validate_config(cfg_with(controls__backends=["numpy", "cuda"]))
    with pytest.raises(ss.SubsurfaceBackendUnavailableError):
        dd.execute(c, "abc", synthetic_prep())                                 # no CPU result is returned for a declared cuda backend


def test_every_declared_backend_pair_is_compared_and_a_cuda_difference_is_caught(cfg, monkeypatch):
    pytest.importorskip("numba")
    c = dd.validate_config(cfg_with(controls__backends=["numpy", "numba", "cuda"], controls__restart_after_day=None))
    prep = synthetic_prep()
    monkeypatch.setattr(dd, "plan_report", lambda cf, p: {"stub": True})
    good = {k: {"package_sha256": v["package_sha256"], "files": v["files"], "file": "x"} for k, v in c["pins"]["expected_sources"].items()}
    monkeypatch.setattr(dd, "source_receipt", lambda: good)
    real = dd.run_column

    def fake(cf, p, scen, backend, restart_after_day=None):
        run = real(cf, p, scen, "numpy" if backend == "cuda" else backend, restart_after_day=restart_after_day)
        if backend == "cuda" and TAMPER["on"]:
            run = copy.deepcopy(run)
            run["trajectory"]["diffusion_m"] = run["trajectory"]["diffusion_m"] + 1e-9
        return {**run, "backend": backend}

    TAMPER = {"on": False}
    monkeypatch.setattr(dd, "run_column", fake)
    res = dd.execute(c, "abc", prep)
    assert set(res["runs"]["wx_d0.05"]["backend_agreement_max_abs"]) == {"numpy_vs_numba", "numpy_vs_cuda", "numba_vs_cuda"}
    TAMPER["on"] = True
    with pytest.raises(dd.ControlFailure, match="cuda"):
        dd.execute(c, "abc", prep)


def _gpu_available():
    import importlib.util

    if importlib.util.find_spec("cupy") is None:
        return False
    from maple.core.backend import cuda_device_count

    return cuda_device_count() > 0


gpu_only = pytest.mark.skipif(not _gpu_available(), reason="actual CuPy device required (Root's GPU lane; the author does not run it)")


@gpu_only
@pytest.mark.parametrize("d", [0.0, 0.05])
def test_gpu_cuda_trajectory_matches_numpy_under_the_inherited_contract_and_restarts_identically(cfg, d):
    prep = synthetic_prep()
    scen = {**cfg["scenarios"][1], "diffusion_coefficient": d}
    a, g = dd.run_column(cfg, prep, scen, "numpy"), dd.run_column(cfg, prep, scen, "cuda")
    dd.assert_backends_agree(a, g, rtol=2e-12, atol=1e-14, bitwise=d == 0.0, what="gpu")
    dd.assert_runs_identical(g, dd.run_column(cfg, prep, scen, "cuda", restart_after_day=2), "gpu restart")


# ---------------------------------------------------------------- correction 02: compact exported inputs and null for missing observations
def test_jsonable_turns_nan_into_null_and_numpy_into_plain_values():
    out = dd.jsonable({"a": np.array([1.0, np.nan]), "b": np.float64("inf"), "c": np.int64(3), "d": np.bool_(True), "e": [float("nan"), 2]})
    assert out == {"a": [1.0, None], "b": None, "c": 3, "d": True, "e": [None, 2]}
    json.dumps(out, allow_nan=False)


@needs_data
def test_prepared_inputs_export_is_compact_auditable_and_has_null_not_nan():
    c = _real_cfg()
    prep = dd.prepare_inputs(c)
    ex = dd.plan_report(c, prep)["prepared_inputs"]
    text = json.dumps(dd.jsonable(ex), allow_nan=False)                                  # strict JSON: no NaN anywhere
    assert "NaN" not in text and len(text) < 20000                                         # compact: no raw full-month series
    assert ex["t0_utc"] == "2025-04-08T22:00:00Z" and len(ex["day_end_utc"]) == 10 and ex["day_end_utc"][0] == "2025-04-09T22:00:00Z"
    assert ex["observed_end_theta_m3m3"]["501"].count(None) == 2 and ex["observed_valid"]["501"].count(False) == 2 and sum(ex["observed_joint_valid"]) == 8
    w = ex["weather_days"]
    assert len(w) == 10 and all(d["n_passing"] == 48 and d["tmin_c"] < d["tmean_c_count_weighted"] < d["tmax_c"] and d["pet_mm_day"] > 0 for d in w)
    assert w[0]["window_start_utc"] == "2025-04-08T22:00:00Z" and w[0]["window_end_utc"] == "2025-04-09T22:00:00Z" and w[0]["pet_calendar_date"] == "2025-04-09"        # midpoint 2025-04-09 10:00 UTC = 03:00 at UTC-7
    assert ex["pet_source"]["terrain_factor"] == 1.0 and ex["raw_initial_point_theta"] and ex["integrated_initial_layer_theta"] != ex["raw_initial_point_theta"]


# ---------------------------------------------------------------- correction 02: portable default paths
def test_default_paths_follow_a_normal_checkout_and_every_location_has_an_environment_override(tmp_path):
    import subprocess

    base = {k: v for k, v in __import__("os").environ.items() if not k.startswith("SYRUP_PHASE3_")}
    code = (f"import sys; sys.path.insert(0, {str(Path(__file__).resolve().parents[2])!r}); from benchmarks.subsurface_validation import paths as p; "
            "print(p.REPO, p.TASKS, p.TASK, p.RAW_ROOT, p.DEPTH_TABLE, p.AIR_JORN, p.INVENTORY, sep='|')")
    out = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, env=base, check=True).stdout.strip().split("|")
    repo = Path(out[0])
    assert Path(out[1]) == repo / "agent_handoffs" / "tasks" and Path(out[2]) == repo / "agent_handoffs" / "tasks" / "subsurface_observed_drydown_20261114"
    assert Path(out[3]) == Path(out[1]) / "jorn_three_layer_20261102" and Path(out[4]).name == "swc_depthsV3" and Path(out[5]) == Path(out[2]) / "downloads_air_JORN"
    assert Path(out[6]) == repo / "docs" / "subsurface_hydrology" / "phase3_observation_inventory.json"
    env = {**base, "SYRUP_PHASE3_TASKS_ROOT": str(tmp_path / "t"), "SYRUP_PHASE3_TASK_DIR": str(tmp_path / "k"), "SYRUP_PHASE3_RAW_ROOT": str(tmp_path / "raw"), "SYRUP_PHASE3_AIR_JORN": str(tmp_path / "aj")}
    out2 = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, env=env, check=True).stdout.strip().split("|")
    assert out2[1] == str(tmp_path / "t") and out2[2] == str(tmp_path / "k") and out2[3] == str(tmp_path / "raw") and out2[5] == str(tmp_path / "aj")
    assert Path(out2[4]) == tmp_path / "t" / "nevada_zero_flux_20261101" / "neon_metadata" / "swc_depthsV3"          # derived locations follow the overridden root; no other fallback
    assert set(P.ENV_VARS) >= {"SYRUP_PHASE3_TASKS_ROOT", "SYRUP_PHASE3_TASK_DIR", "SYRUP_PHASE3_RAW_ROOT", "SYRUP_PHASE3_DEPTH_TABLE", "SYRUP_PHASE3_AIR_JORN", "SYRUP_PHASE3_AIR_MOAB", "SYRUP_PHASE3_INVENTORY"}
