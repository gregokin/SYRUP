"""Engineering tests of the joint d/stress calibration helper. The objectives here are SYNTHETIC toy functions and the column runner/scorer are MOCKED where stated: they test the search mathematics, caching, forwarding,
refusals, provenance and selection logic only. They are NOT physics qualification and no native column, raw data or GPU is used (the real previous fits are only READ and hash-checked)."""
from __future__ import annotations

import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarks.subsurface_joint_stress import joint as J
from benchmarks.subsurface_joint_stress import search2d as S2
from benchmarks.subsurface_stress_calibration import scoring as SC
from benchmarks.subsurface_stress_calibration import search as SE

D_AX, S_AX = J.frozen_axes()


def quad(d0, s0, a=1.0, b=1.0, c=0.0):
    def f(d, s):
        dz, ds = S2.z_of_d(d) - S2.z_of_d(d0), s - s0
        return 1e-4 + a * 1e-6 * dz**2 + b * ds**2 * 1e-3 + c * 1e-5 * dz * ds
    return f


# ---------------------------------------------------------------------------------------------------- grids/coordinates
def test_frozen_axes_counts_endpoints_and_required_points():
    assert D_AX.size == 43 and S_AX.size == 85 and D_AX.size * S_AX.size == 3655
    assert D_AX[0] == 0.0 and D_AX[1] == 1e-6 and D_AX[-1] == 1.0 and 0.05 in D_AX
    assert np.sum((D_AX > 0) & (D_AX < 1) & ~np.isin(D_AX, [0.05])) == 40      # 1e-6 .. 0.7079 log points
    assert S_AX[0] == 0.010001 and S_AX[-1] == 1.0 and 0.18 in S_AX
    for st in J.OLD_STRESS.values():
        assert st in S_AX
    assert np.all(np.diff(D_AX) > 0) and np.all(np.diff(S_AX) > 0)


def test_transform_exact_at_zero_and_saturating_at_bounds():
    assert S2.z_of_d(0.0) == 0.0 and S2.d_of_z(0.0) == 0.0 and S2.d_of_z(-1.0) == 0.0
    assert S2.d_of_z(S2.Z_MAX) == 1.0 and S2.d_of_z(S2.Z_MAX + 1.0) == 1.0
    for d in (1e-6, 3e-4, 0.05, 0.7):
        assert S2.d_of_z(S2.z_of_d(d)) == pytest.approx(d, rel=1e-12)
    zs = np.linspace(0, S2.Z_MAX, 500)
    ds = [S2.d_of_z(z) for z in zs]
    assert all(0.0 <= d <= 1.0 for d in ds) and all(b >= a for a, b in itertools.pairwise(ds))


def test_incumbents_are_method_specific_and_on_grid():
    h, g = J.incumbents("harmonic"), J.incumbents("geometric")
    assert h[0] == g[0] == (0.0, 0.20644317480382438)
    assert h[1] == (0.05, 0.21717029762566153) and g[1] == (0.05, 0.2171797728919738)
    for d, s in h + g:
        assert d in D_AX and s in S_AX


# ------------------------------------------------------------------------------------------------------ 2-D search
def test_interior_recovery_off_grid_with_correlated_coordinates():
    f = quad(0.0123, 0.3137, c=0.8)
    res = S2.search_joint(f, D_AX, S_AX, [(0.0, 0.2), (0.05, 0.25)])
    sel = res["selected"]
    assert not res["boundary_solution"] and not res["requires_root_bounded_extension_decision"]
    assert abs(S2.z_of_d(sel["d"]) - S2.z_of_d(0.0123)) < 2e-3 and abs(sel["stress"] - 0.3137) < 2e-3
    assert not sel["on_grid"] and sel["objective"] <= min(min(r) for r in res["grid_objective"])
    assert len(res["profile_over_d"]) == D_AX.size and all(p["objective"] <= f(p["d"], p["grid_best_stress"]) for p in res["profile_over_d"])
    assert any(m["scipy_success"] for m in res["local_minima"])


def test_d_zero_exact_is_evaluated_admitted_and_selected_when_best():
    seen = []

    def f(d, s):
        seen.append((d, s))
        return 1e-4 + 5e-3 * d + (s - 0.2064) ** 2 * 1e-3
    res = S2.search_joint(f, D_AX, S_AX, J.incumbents("harmonic"))
    assert (0.0, 0.20644317480382438) in seen
    assert res["selected"]["d"] == 0.0 and res["selected"]["d_at_lower_bound"] and res["boundary_solution"]
    assert not res["requires_root_bounded_extension_decision"] and res["selected"]["stress"] == pytest.approx(0.2064, abs=2e-3)
    assert "no unbounded optimum is claimed" in res["limits_note"]


def test_upper_d_bound_flags_root_extension_and_does_not_expand():
    f = lambda d, s: 1e-4 + (1.0 - d) * 1e-3 + (s - 0.4) ** 2 * 1e-3
    res = S2.search_joint(f, D_AX, S_AX, J.incumbents("geometric"))
    assert res["selected"]["d"] == 1.0 and res["selected"]["d_at_upper_bound"]
    assert res["boundary_solution"] and res["requires_root_bounded_extension_decision"]
    assert max(t[0] for t in res["trace"]) <= 1.0 and min(t[1] for t in res["trace"]) >= S2.STRESS_MIN and max(t[1] for t in res["trace"]) <= 1.0


def test_stress_bound_and_edge_optimisation_along_stress_lo_and_hi():
    lo = S2.search_joint(lambda d, s: 1e-4 + (s - S2.STRESS_MIN) * 1e-3 + (S2.z_of_d(d) - S2.z_of_d(0.003)) ** 2 * 1e-6, D_AX, S_AX)
    assert lo["selected"]["stress_at_lower_bound"] and lo["boundary_solution"]
    assert abs(S2.z_of_d(lo["stress_edges"]["stress_lo"]["d"]) - S2.z_of_d(0.003)) < 2e-3          # d refined ALONG the edge
    hi = S2.search_joint(lambda d, s: 1e-4 + (1.0 - s) * 1e-3 + (S2.z_of_d(d) - S2.z_of_d(0.2)) ** 2 * 1e-6, D_AX, S_AX)
    assert hi["selected"]["stress_at_upper_bound"] and abs(S2.z_of_d(hi["stress_edges"]["stress_hi"]["d"]) - S2.z_of_d(0.2)) < 2e-3


def test_local_multiple_minima_both_found_and_global_selected():
    def f(d, s):
        za, zb = (S2.z_of_d(d) - S2.z_of_d(0.002)) ** 2 + 30 * (s - 0.2) ** 2, (S2.z_of_d(d) - S2.z_of_d(0.4)) ** 2 + 30 * (s - 0.7) ** 2
        return 1e-4 + 1e-4 * min(za + 0.05, zb)
    res = S2.search_joint(f, D_AX, S_AX)
    near = [m for m in res["local_minima"] if m["role"] == "separated_candidate"]
    assert any(m["d"] < 0.01 and abs(m["stress"] - 0.2) < 0.01 for m in near) and any(m["d"] > 0.2 and abs(m["stress"] - 0.7) < 0.02 for m in near)
    assert res["selected"]["d"] > 0.2 and abs(res["selected"]["stress"] - 0.7) < 0.01           # the lower (B) well is the global one
    assert len(near) >= 2


def test_old_incumbents_are_included_and_selection_never_worse_than_them():
    inc = J.incumbents("harmonic")
    f = lambda d, s: 1e-4 + (d - 0.3) ** 2 * 1e-3 + (s - 0.5) ** 2 * 1e-3
    res = S2.search_joint(f, D_AX, S_AX, inc)
    roles = [m for m in res["local_minima"] if m["role"] == "old_incumbent"]
    assert [tuple(m["start"]) for m in roles] == inc
    ev = {(t[0], t[1]): t[2] for t in res["trace"]}
    assert all(p in ev for p in inc) and res["selected"]["objective"] <= min(ev[p] for p in inc)


def test_selection_is_minimum_over_every_evaluated_pair(monkeypatch):
    f_true = lambda d, s: 1e-3 + (d - 0.5) ** 2 * 1e-3 + (s - 0.5) ** 2 * 1e-3
    marker = (0.1234567, 0.4321)

    def f(d, s):
        return f_true(d, s) - (1.0 if (d, s) == marker else 0.0)

    def bad_refine(fo, zb, sb, start):
        fo(*marker)                                                         # an intermediate evaluation that beats whatever is returned
        return {"method": "mock", "scipy_success": True, "d": start[0] * 0 + 0.5, "stress": 0.5, "objective": fo(0.5, 0.5)}
    monkeypatch.setattr(S2, "refine_box", bad_refine)
    res = S2.search_joint(f, D_AX, S_AX)
    assert (res["selected"]["d"], res["selected"]["stress"]) == marker and res["selected"]["objective"] == min(t[2] for t in res["trace"])


def test_exact_tie_goes_to_smaller_d_then_smaller_stress():
    res = S2.search_joint(lambda d, s: 1.0, D_AX, S_AX)
    assert res["selected"]["d"] == 0.0 and res["selected"]["stress"] == S2.STRESS_MIN
    cache = {(0.2, 0.3): 1.0, (0.1, 0.9): 1.0, (0.1, 0.4): 1.0, (0.5, 0.1): 2.0}
    assert S2.best_of(cache)[0] == (0.1, 0.4)


def test_failed_scipy_refinement_is_not_reported_successful(monkeypatch):
    import scipy.optimize as so
    real = so.minimize

    def failing(fun, x0, **kw):
        r = real(fun, x0, **kw)
        r.success = False
        return r
    monkeypatch.setattr(so, "minimize", failing)
    res = S2.search_joint(quad(0.01, 0.3), D_AX, S_AX)
    assert all(not m["scipy_success"] and m["method"] == "coordinate_descent_after_scipy_failure" for m in res["local_minima"])
    assert abs(res["selected"]["stress"] - 0.3) < 5e-3


def test_refinement_never_leaves_its_box_or_clips(monkeypatch):
    f = S2.CachedObjective2D(quad(0.01, 0.3))
    r = S2.refine_box(f, (0.0, 1.0), (0.2, 0.4), (0.5, 0.3))
    assert 0.0 <= r["d"] <= 1.0 and 0.2 <= r["stress"] <= 0.4
    with pytest.raises(S2.JointSearchError, match="outside its box"):
        S2.refine_box(f, (0.0, 1.0), (0.2, 0.4), (0.5, 0.9))
    assert all(S2.STRESS_MIN <= s <= 1.0 and 0.0 <= d <= 1.0 for d, s in f.cache)


def test_cache_runs_each_pair_once_and_refuses_bad_values():
    calls = []
    f = S2.CachedObjective2D(lambda d, s: calls.append((d, s)) or d + s)
    f(0.1, 0.3), f(0.1, 0.3), f(0.2, 0.3)
    assert calls == [(0.1, 0.3), (0.2, 0.3)] and f.order == calls
    for bad in ((float("nan"), 0.3), (0.1, float("inf")), (-1e-12, 0.3), (1.0000001, 0.3), (0.1, 0.01), (0.1, 1.5), (True, 0.3), ("0.1", 0.3)):
        with pytest.raises(S2.JointSearchError):
            f(*bad)
    assert len(calls) == 2
    with pytest.raises(S2.JointSearchError, match="not finite"):
        S2.CachedObjective2D(lambda d, s: float("nan"))(0.1, 0.3)


def test_search_axes_validation():
    with pytest.raises(S2.JointSearchError):
        S2.search_joint(lambda d, s: 1.0, np.array([0.0, 0.5, 0.9]), S_AX)           # does not reach d = 1
    with pytest.raises(S2.JointSearchError):
        S2.search_joint(lambda d, s: 1.0, D_AX, S_AX, [(0.0, 2.0)])                   # invalid incumbent refused before any evaluation


def test_near_optimal_region_descriptive_and_components():
    d, s = np.arange(5.0), np.arange(6.0)
    y = np.full((5, 6), 10.0)
    y[1, 1], y[1, 2], y[4, 5] = 1.0, 1.005, 1.04
    r = S2.near_optimal_region(d, s, y)
    assert "not a confidence" in r["descriptive_only"]
    assert r["fractions"]["0.01"]["n_points"] == 2 and r["fractions"]["0.01"]["n_components"] == 1
    assert r["fractions"]["0.05"]["n_points"] == 3 and r["fractions"]["0.05"]["n_components"] == 2


# ------------------------------------------------------------------------------------- pooled objective (mocked column)
class Engine:
    """MOCK column runner + scorer: the 'run' is a tiny dict that depends on (d, stress); records what it was called with. Engineering only."""

    def __init__(self):
        self.calls = []

    def runner(self, cfg, prep, scen, backend):
        self.calls.append({"cfg_d": cfg["scenarios"][0]["diffusion_coefficient"], "scen_d": scen["diffusion_coefficient"], "stress": cfg["layers"]["theta_evaporation_stress"], "backend": backend, "E": scen["evaporation_factor"]})
        return {"state": np.array([scen["diffusion_coefficient"], cfg["layers"]["theta_evaporation_stress"]])}

    def scorer(self, cfg, prep, run):
        d, s = run["state"]
        model = np.array([[0.30 - 0.01 * k * (1 + s + d) for k in range(1, 5)]])
        return SC.window_score(model, [0.30], prep["obs"], prep["obs0"])


def make_pool():
    eng = Engine()
    cfgs = {w: {"scenarios": [{"name": "x", "diffusion_coefficient": 0.05, "evaporation_factor": 1.0, "potential_et": "weather", "bottom": "open"}], "layers": {"theta_evaporation_stress": 0.18}} for w in J.WINDOWS}
    obs = np.array([[0.28, 0.26, np.nan, 0.21]])
    preps = {w: {"obs": obs.copy() + 0.001 * i, "obs0": [0.31 + 0.002 * i]} for i, w in enumerate(J.WINDOWS)}
    return eng, J.JointPooled(cfgs, preps, runner=eng.runner, scorer=eng.scorer), preps


def test_d_and_stress_both_forwarded_to_scenario_config_and_E_fixed():
    eng, pool, _ = make_pool()
    pool.evaluate(0.0123, 0.31)
    assert len(eng.calls) == 3
    assert all(c["cfg_d"] == c["scen_d"] == 0.0123 and c["stress"] == 0.31 and c["E"] == 1.0 and c["backend"] == "numba" for c in eng.calls)
    pool.evaluate(0.0, 0.2)
    assert eng.calls[-1]["cfg_d"] == 0.0 and eng.calls[-1]["stress"] == 0.2


def test_objective_pools_valid_observations_with_offsets_and_no_state_change():
    eng, pool, preps = make_pool()
    before = {w: (p["obs"].copy(), list(p["obs0"])) for w, p in preps.items()}
    r = pool.evaluate(0.02, 0.25)
    expect = [eng.scorer(None, preps[w], eng.runner({"scenarios": [{"diffusion_coefficient": 0.02}], "layers": {"theta_evaporation_stress": 0.25}}, None, {"diffusion_coefficient": 0.02, "evaporation_factor": 1.0}, "numba")) for w in J.WINDOWS]
    sse, n = sum(e["sse"] for e in expect), sum(e["n"] for e in expect)
    assert r["n"] == n == 9 and r["sse"] == pytest.approx(sse) and r["mse"] == pytest.approx(sse / n)     # equal weight per valid observation (NaN excluded)
    for w, p in preps.items():                                                                           # the observations/initial values (offset inputs) are untouched
        assert np.array_equal(p["obs"], before[w][0], equal_nan=True) and list(p["obs0"]) == before[w][1]
    assert pool.evaluate(0.02, 0.25)["mse"] == r["mse"] and len(eng.calls) == 3 + 3                          # cached re-evaluation runs nothing new (6 = 3 pool + 3 expect above)


def test_only_light_scalars_are_retained_per_trial():
    _eng, pool, _ = make_pool()
    for i in range(20):
        pool.evaluate(0.001 * i, 0.2 + 0.01 * i)
    assert len(pool.light) == 20 and pool.n_objective_runs == 60 and pool.n_full_runs == 0
    for rec in pool.light.values():
        assert all(set(v) == {"n", "sse", "zero_change_sse"} and all(isinstance(x, (int, float)) for x in v.values()) for v in rec.values())
    assert not any(isinstance(v, np.ndarray) for v in vars(pool).values())
    assert len(json.dumps(pool.evaluations_table())) < 20000                                              # compact, reproducible record


def test_full_rerun_matches_cached_objective_exactly():
    _eng, pool, _ = make_pool()
    pool.evaluate(0.01, 0.3)
    runs, _wins, pooled = pool.full(0.01, 0.3)
    assert set(runs) == set(J.WINDOWS) and pooled["sse"] == pool.evaluate(0.01, 0.3)["sse"]


def test_invalid_and_nonfinite_refused_before_any_column_runs():
    eng, pool, _ = make_pool()
    for bad in ((float("nan"), 0.2), (0.1, float("nan")), (float("inf"), 0.2), (-0.1, 0.2), (2.0, 0.2), (0.1, 0.0), (0.1, 0.01), (0.1, 1.01), (None, 0.2)):
        with pytest.raises((S2.JointSearchError, TypeError)):
            pool.evaluate(*bad)
    assert eng.calls == [] and pool.light == {}
    with pytest.raises(S2.JointSearchError):
        J.with_params({"scenarios": [{}], "layers": {}}, float("nan"), 0.2)


def test_joint_search_through_pooled_objective_selects_actual_minimum():
    eng, pool, _ = make_pool()
    res = S2.search_joint(pool.objective, D_AX, S_AX, J.incumbents("harmonic"))
    ev = {(r[0], r[1]) for r in pool.evaluations_table()}
    assert len(ev) == res["n_evaluations"] and (res["selected"]["d"], res["selected"]["stress"]) in ev
    assert res["selected"]["objective"] == min(pool.evaluate(d, s)["mse"] for d, s in ev)
    assert all(c["scen_d"] == c["cfg_d"] for c in eng.calls)


# ------------------------------------------------------------------------------------------- configs and provenance
def test_with_params_matches_builder_and_changes_only_d_and_stress():
    base = J.build_configs("harmonic", 0.05, 0.18)
    for w in J.WINDOWS:
        cfg = J.dd.validate_config(base[w])
        a = J.with_params(cfg, 0.0123, 0.3)
        b = J.dd.validate_config(J.build_configs("harmonic", 0.0123, 0.3)[w])
        assert a["scenarios"][0]["diffusion_coefficient"] == b["scenarios"][0]["diffusion_coefficient"] == 0.0123
        assert a["layers"]["theta_evaporation_stress"] == b["layers"]["theta_evaporation_stress"] == 0.3
        assert np.array_equal(a["layers"]["interface_conductivity_m_s"], b["layers"]["interface_conductivity_m_s"]) and a["scenarios"][0]["evaporation_factor"] == 1.0
        assert a["window"] == cfg["window"] and a["pins"] == cfg["pins"] and cfg["scenarios"][0]["diffusion_coefficient"] == 0.05      # the source config is not mutated


def test_config_provenance_is_truthful_about_joint_fit():
    for m in J.METHODS:
        cfgs = J.build_configs(m, 0.0123, 0.3)
        for w in J.WINDOWS:
            p = cfgs[w]["provenance"]
            assert p["role"] == "calibration_all_windows" and "JOINTLY" in p["evaporation_stress"] and "JOINTLY" in p["diffusion"]
            assert "stress_calibration" not in p and "theta1500 (no-refit" not in p["evaporation_stress"]
            jc = p["joint_calibration"]
            assert jc["fitted"] == ["diffusion_coefficient", "theta_evaporation_stress"] and jc["fixed"] == {"E": 1.0, "T": 0} and jc["d"] == 0.0123 and jc["stress"] == 0.3
            assert jc["windows_used"] == list(J.WINDOWS) and jc["held_out_windows"] == [] and jc["method"] == m
            assert p["historical_fixed_d_stress_calibration"]["d"] == 0.0123 and p["historical_template_role"] in ("train", "heldout")   # history retained, labelled
            text = json.dumps({k: v for k, v in p.items() if not k.startswith("historical")})
            assert "d is never fitted" not in text and "fixed d" not in text and "heldout" not in cfgs[w]["name"] and "train" not in cfgs[w]["name"]
            assert cfgs[w]["pins"]["expected_sources"] == J.C.EXPECTED_SOURCES


def test_all_three_windows_and_sources_frozen():
    assert J.WINDOWS == ("JORN4_A", "JORN4_B", "MOAB2_C") and J.E_FACTOR == 1.0
    assert J.C.EXPECTED_SOURCES["maple_syrup"]["package_sha256"] == "37d89f23dbf60626c85f85ea5dd6d1734d3bc7a4aa61d8f2fdfc29f237bcfbd0" and J.C.EXPECTED_SOURCES["maple"]["package_sha256"].startswith("fbc504bb")
    assert J.source_identities({"a": {"package_sha256": "x", "files": 1, "extra": 2}}) == {"a": ("x", 1)}


def test_execute_guards_sources_and_raw_before_preparing_and_fitting(monkeypatch, tmp_path):
    order = []
    monkeypatch.setattr(J.C, "source_guard", lambda c: order.append("source") or {"maple_syrup": {"package_sha256": "a", "files": 1}})
    monkeypatch.setattr(J.C, "raw_guard", lambda c: order.append("raw") or {})
    monkeypatch.setattr(J.dd, "prepare_inputs", lambda c: order.append("prepare") or {})
    monkeypatch.setattr(J.dd, "prepared_inputs_export", lambda c, p: {})

    class Stop(Exception):
        pass

    def stop(*a, **k):
        order.append("fit")
        raise Stop
    monkeypatch.setattr(J, "fit_method", stop)
    with pytest.raises(Stop):
        J.execute(tmp_path / "o")
    assert order[:2] == ["source", "raw"] and order.index("prepare") > 1 and order[-1] == "fit" and order.count("prepare") == 3


# ----------------------------------------------------------------------------------------------- previous data / plan
def test_real_previous_fits_load_with_manifest_hashes():
    prev = J.load_previous()
    assert prev["n_files_verified"] == 26
    assert {k: v["selected_stress"] for k, v in prev["fits"].items()} == J.OLD_STRESS
    assert prev["fits"]["sensitivity_d0"]["selected"]["objective"]["n"] == 53


def _fake_root(tmp_path, tamper=False):
    base = tmp_path / J.OLD_DATA_REL
    (base / "data/actual_cpu/fits").mkdir(parents=True)
    man = {}
    for n, st in J.OLD_STRESS.items():
        t = json.dumps({"selected_stress": st, "selected": {"objective": {"mse": 1e-4, "rmse": 0.01, "n": 53}}})
        (base / f"data/actual_cpu/fits/{n}.json").write_text(t)
        man[f"actual_cpu/fits/{n}.json"] = hashlib.sha256(t.encode()).hexdigest()
    (base / "data/actual_cpu/summary.json").write_text("{}")
    man["actual_cpu/summary.json"] = hashlib.sha256(b"{}").hexdigest()
    (base / "artifact_manifest.json").write_text(json.dumps({"data": man}))
    if tamper:
        (base / "data/actual_cpu/fits/main.json").write_text("{}")
    return tmp_path, hashlib.sha256((base / "artifact_manifest.json").read_bytes()).hexdigest()


def test_previous_data_hash_mismatch_or_missing_refused(tmp_path):
    assert J.load_previous(*_fake_root(tmp_path / "ok"))["n_files_verified"] == 4
    with pytest.raises(J.JointError, match="differs from the manifest hash"):
        J.load_previous(*_fake_root(tmp_path / "bad", tamper=True))
    r, sha = _fake_root(tmp_path / "gone")
    (r / J.OLD_DATA_REL / "data/actual_cpu/summary.json").unlink()
    with pytest.raises(J.JointError, match="missing"):
        J.load_previous(r, sha)


def test_plan_writes_configs_and_protocol_and_refuses_nonempty(tmp_path):
    proto = J.plan(tmp_path / "p")
    assert proto["n_grid_pairs"] == 3655 and proto["n_d"] == 43 and proto["n_stress"] == 85 and len(proto["config_sha256"]) == 6
    assert sorted(p.name for p in (tmp_path / "p/configs").iterdir()) == sorted(f"{m}_{w}.json" for m in J.METHODS for w in J.WINDOWS)
    assert json.loads((tmp_path / "p/protocol.json").read_text())["old_incumbents"]["harmonic"][1][1] == 0.21717029762566153
    with pytest.raises(J.JointError, match="not empty"):
        J.plan(tmp_path / "p")


def test_cli_refusals(tmp_path, capsys):
    assert J.main(["--out", str(tmp_path / "a"), "--verify-cuda"]) == 2 and "needs --execute" in capsys.readouterr().err
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "x").write_text("x")
    assert J.main(["--out", str(tmp_path / "b")]) == 2
    assert J.main(["--out", str(tmp_path / "c")]) == 0
    assert not (tmp_path / "c" / "fits").exists()                                                          # the plan runs no model


def test_old_search_module_unchanged_and_reused():
    assert S2.SE is SE and J.C.SC is SC and J.dd.run_column.__module__.endswith("drydown_driver")


# ------------------------------------------------------------------------------------------------- correction 01
def test_manifest_itself_is_pinned_even_when_self_consistent(tmp_path):
    root, own_sha = _fake_root(tmp_path / "selfconsistent")
    assert J.OLD_MANIFEST_SHA256 == "4af7b08fc2f1234a2130dabdd82e00b9acc9e28e3ed7fbd4c85388dea3504e23"
    assert J.load_previous(root, own_sha)["n_files_verified"] == 4                      # every listed hash matches ...
    with pytest.raises(J.JointError, match="manifest differs from its pinned sha256"):    # ... but the default pin refuses a manifest that is not the immutable e11 one
        J.load_previous(root)
    # the real manifest is the pinned one
    assert hashlib.sha256((Path(J.__file__).resolve().parents[2] / J.OLD_DATA_REL / "artifact_manifest.json").read_bytes()).hexdigest() == J.OLD_MANIFEST_SHA256


def _synthetic_pool_with_real_serializers():
    """Real validated configs (no raw data) + synthetic prepared windows/runs: exercises the ACTUAL case_record -> case_summary -> run_record -> physical_flags path with a mocked column only."""
    from maple_syrup import subsurface as ss
    cfgs = {w: J.dd.validate_config(c) for w, c in J.build_configs("harmonic", 0.05, 0.18).items()}
    n = 4
    preps = {w: {"sensor_depths_m": np.array([0.05, 0.15, 0.25]), "theta0_points": np.array([0.30, 0.25, 0.20]), "initial": {"layer_theta": np.array([0.28, 0.24, 0.22])},
                 "obs": {v: np.array([0.29, np.nan, 0.27, 0.26]) - 0.01 * i for i, v in enumerate(J.dd.SENSOR_VERS)}} for w in cfgs}

    def runner(cfg, prep, scen, backend):
        d, s = scen["diffusion_coefficient"], cfg["layers"]["theta_evaporation_stress"]
        lt = 0.28 - 0.004 * (1 + s + d) * np.arange(1, n + 1)[:, None] * np.ones((1, 3))
        cum = ss.CumulativeBudget(n_steps=n, elapsed_s=86400.0 * n, added_m3=0.0, surface_return_m3=0.0, drainage_m3=1e-4, evaporation_m3=2e-3, evaporation_unmet_m3=0.0)
        return {"trajectory": {"t_s": np.arange(1.0, n + 1)}, "layer_theta": lt, "initial_storage_m": 0.1, "final_storage_m": 0.09, "cumulative": cum, "ledger": {"residual_m3": 0.0}}
    return J.JointPooled(cfgs, preps, runner=runner), cfgs


def test_case_record_end_to_end_serializes_with_joint_provenance():
    pool, cfgs = _synthetic_pool_with_real_serializers()
    rec = J.case_record("harmonic", pool, 0.05, 0.18)
    text = json.dumps(J.dd.jsonable(rec), allow_nan=False)
    assert rec["d"] == 0.05 and rec["stress"] == 0.18 and set(rec["native_trajectories"]) == set(J.WINDOWS) and rec["summary"]["objective"]["n"] > 0
    for w in J.WINDOWS:
        fl = rec["physical_flags"][w]
        assert fl["historical_site_theta1500"] == cfgs[w]["provenance"]["historical_fixed_d_stress_calibration"]["site_theta1500"] and "site_theta1500" not in fl
        assert fl["top_layer_fc"] == float(cfgs[w]["layers"]["theta_field_capacity"][0]) and isinstance(fl["above_fc"], bool)
    assert "stress_calibration\"" not in text or "historical_fixed_d" in text
    with pytest.raises(KeyError):                                                      # the old fixed-d helper is NOT what the joint path uses
        J.C.physical_flags(cfgs, 0.18)
    assert pool.n_objective_runs == 3 and pool.n_full_runs == 3                        # one cached pair + one separately counted full rerun


def test_column_run_counts_are_separately_labelled():
    pool, _ = _synthetic_pool_with_real_serializers()
    for d, s in ((0.0, 0.2), (0.05, 0.2), (0.1, 0.3)):
        pool.evaluate(d, s)
    J.case_record("harmonic", pool, 0.05, 0.2)
    J.case_record("harmonic", pool, 0.1, 0.3)
    assert len(pool.light) == 3 and pool.n_objective_runs == 3 * 3 == len(J.WINDOWS) * len(pool.light) and pool.n_full_runs == 6


def test_near_optimal_uses_best_evaluated_value_and_reports_empty_bands():
    d, s = np.arange(4.0), np.arange(4.0)
    y = np.full((4, 4), 2.0)
    y[2, 2] = 1.5
    coarse = S2.near_optimal_region(d, s, y, best=1.0)                              # a refined interior optimum (1.0) lies far below every grid point
    assert coarse["min_mse"] == 1.0 and coarse["grid_min_mse"] == 1.5 and coarse["reference"] == "best evaluated value"
    for fr in ("0.01", "0.05"):
        r = coarse["fractions"][fr]
        assert r["n_points"] == 0 and r["d_extent"] is None and r["stress_extent"] is None and r["n_components"] == 0 and r["empty"]
    ok = S2.near_optimal_region(d, s, y, best=1.45)
    assert ok["fractions"]["0.05"]["n_points"] == 1 and ok["fractions"]["0.05"]["d_extent"] == [2.0, 2.0] and "not a confidence" in ok["descriptive_only"]
    assert S2.near_optimal_region(d, s, y)["reference"] == "grid minimum"


def test_search_near_optimal_is_against_refined_selected_mse():
    res = S2.search_joint(quad(0.0123, 0.3137), D_AX, S_AX)
    near = res["near_optimal"]
    assert near["min_mse"] == res["selected"]["objective"] and near["min_mse"] <= near["grid_min_mse"]
    assert all(v["empty"] is None or v["n_points"] == 0 for v in near["fractions"].values())


def test_profile_uses_lowest_of_all_cached_stresses_at_each_d(monkeypatch):
    marker_d = float(D_AX[20])
    marker = (marker_d, 0.4321)
    base = quad(0.3, 0.5)

    def f(d, s):
        return base(d, s) - (1.0 if (d, s) == marker else 0.0)

    def sampling_refine(fn, a, b, xatol=1e-10):                                      # an intermediate fixed-d sample that beats the returned minimiser
        if abs(fn.__defaults__[0] - marker_d) < 1e-300:
            fn(marker[1])
        return a, "mock"
    monkeypatch.setattr(S2, "_refine_1d", sampling_refine)
    res = S2.search_joint(f, D_AX, S_AX)
    p = next(p for p in res["profile_over_d"] if p["d"] == marker_d)
    assert p["stress"] == 0.4321 and p["objective"] == f(*marker) and p["n_cached_stresses"] >= S_AX.size + 1
    for q in res["profile_over_d"]:
        assert q["objective"] == min(v for (dd_, _), v in zip(((t[0], t[1]) for t in res["trace"]), (t[2] for t in res["trace"])) if dd_ == q["d"])


def test_axes_and_counts_validated_before_any_objective_call():
    calls = []

    def f(d, s):
        calls.append((d, s))
        return 1.0
    bad_s = S_AX.copy()
    bad_s[40] = np.nan
    bad_d = D_AX.copy()
    bad_d[10] = np.inf
    for ax_d, ax_s in ((D_AX, bad_s), (bad_d, S_AX), (D_AX[None, :], S_AX), (D_AX, S_AX[::-1])):
        with pytest.raises(S2.JointSearchError):
            S2.search_joint(f, ax_d, ax_s)
    for bad in (-1, 2.5, True, None, "6"):
        with pytest.raises(S2.JointSearchError, match="n_refine"):
            S2.search_joint(f, D_AX, S_AX, n_refine=bad)
    for bad in (2, 4.5, True):
        with pytest.raises(S2.JointSearchError, match="n_log"):
            S2.d_grid(bad)
    with pytest.raises(S2.JointSearchError):
        S2.near_optimal_region(D_AX, np.array([0.1, np.nan, 0.5]), np.zeros((D_AX.size, 3)))
    assert calls == []
    assert D_AX.size == 43 and S_AX.size == 85                                        # the declared protocol grid is preserved
