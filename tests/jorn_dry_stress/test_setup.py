"""Stage-1 SETUP and focused NATIVE-component tests of the JORN defaults preset and the six dry-window fixed-stress cases. No dry-window run is executed. They check the preset loader (closed validation, scope, source
identity), the six generated cases against literal anchors and the frozen historical templates (exact whitelist of deltas), MOAB's OWN native SR33 field capacity, the source pins (new pair passes, the old 37d89f23
pin is refused), the real input preparation (53 QA observations, initial inventory below every field capacity), one real native daily step per case that shows gravity is inactive (so the field capacity is not
identifiable from these dry data) with the native conservation bound, the runner recipes and non-mutation. They need the Root runtime (MAPLE importable; SYRUP_PHASE3_* paths set by the runner or the default checkout)."""
from __future__ import annotations

import copy
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarks.jorn_dry_stress import generate as G
from benchmarks.jorn_dry_stress import preset as PS
from benchmarks.jorn_dry_stress import properties as PR
from benchmarks.jorn_dry_stress import run as RUN
from benchmarks.subsurface_validation import drydown_driver as dd
from maple_syrup import subsurface as ss

D_SEL = 0.0010239461376024856
JORN_SAT = [0.4643018867924528, 0.4651712600744756, 0.4586310940906557]
OLD_FC_JORN = [0.39760410394534984, 0.39866650157195227, 0.39508647981211864]
OLD_FC_MOAB = [0.37996672767136985, 0.39218936166045765, 0.39218936166045765]
MOAB_SR33_RECEIPT = [0.1393260220046285, 0.15622473499486667, 0.15622473499486667]          # theta33 in the frozen receipt root_relative_density_probe.json
KEYS = ["JORN4_A_s1", "JORN4_A_s1p5", "JORN4_B_s1", "JORN4_B_s1p5", "MOAB2_C_s1", "MOAB2_C_s1p5"]
OBS = {"JORN4_A": 28, "JORN4_B": 11, "MOAB2_C": 14}
NDAYS = {"JORN4_A": 10, "JORN4_B": 4, "MOAB2_C": 6}


def expect(key):
    w, s = key.rsplit("_", 1)
    return w, (0.2017 if s == "s1" else 0.30255)


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    before = {n: G.sha256_file(G.HIST / n) for n, _ in G.TEMPLATES.values()} | {r: G.sha256_file(G.CANDIDATE / r) for r in G.HELPERS if r.startswith("benchmarks/subsurface_validation")}
    out = tmp_path_factory.mktemp("ds")
    idx = G.generate(out)
    after = {n: G.sha256_file(G.HIST / n) for n, _ in G.TEMPLATES.values()} | {r: G.sha256_file(G.CANDIDATE / r) for r in G.HELPERS if r.startswith("benchmarks/subsurface_validation")}
    assert before == after                                              # generation does not mutate the templates or the unchanged driver files
    return out, idx


def load_case(idx, key):
    return json.loads(Path(idx["cases"][key]["path"]).read_text())


def flat(o, pre=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from flat(v, f"{pre}.{k}" if pre else k)
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from flat(v, f"{pre}[{i}]")
    else:
        yield pre, o


def diff(a, b):
    fa, fb = dict(flat(a)), dict(flat(b))
    return {k for k in set(fa) | set(fb) if fa.get(k, KeyError) != fb.get(k, KeyError)}


# ------------------------------------------------------------------------------------------------------------------------------ preset
def test_preset_loads_with_fixed_values_and_source_identity():
    p = PS.load_preset()
    assert p["theta_field_capacity"] == [0.10, 0.10, 0.10] and p["theta_evaporation_stress"] == 0.30255 and p["evaporation_factor"] == 1.0 and p["diffusion_coefficient"] == D_SEL and p["conductivity_multiplier"] == 1.0
    assert p["source"]["accepted_study_commit"] == "99465040d947ec6ff377003a88f3b2636d923f15" and p["source"]["case_file_sha256"] == "027dfdd313c43646f46763da4fee8096454c511a356ebf57a911c22d9d546c9d"
    assert PS.CANDIDATE.joinpath(p["source"]["case_file"]).is_file() and G.sha256_file(PS.CANDIDATE / p["source"]["case_file"]) == p["source"]["case_file_sha256"]
    assert p["source_layers"]["theta_saturation"] == JORN_SAT and p["source_matrix"]["conductivity_m_s"] == 1.0614362945745998e-05
    assert any("MOAB default" in x for x in p["scope"]["not"]) and "JORN" in p["scope"]["applies_to"] and "third_layer" in "".join(p["deferred"])


@pytest.mark.parametrize("mut", ["extra", "missing", "nan", "stress_below_fc", "scope_moab", "not_refit", "sha", "source_mismatch", "conductivity", "percent_ambiguous"])
def test_preset_refusals(tmp_path, mut):
    p = json.loads(PS.PRESET_PATH.read_text())
    if mut == "extra":
        p["extra"] = 1
    elif mut == "missing":
        del p["deferred"]
    elif mut == "nan":
        p["parameters"]["diffusion_coefficient"] = float("nan")
    elif mut == "stress_below_fc":
        p["parameters"]["theta_evaporation_stress"] = 0.05
    elif mut == "scope_moab":
        p["scope"]["not"] = ["a global SYRUP physics default"]
    elif mut == "not_refit":
        p["decision"]["not_a_refit"] = False
    elif mut == "sha":
        p["source"]["case_file_sha256"] = "0" * 64
    elif mut == "source_mismatch":
        p["parameters"]["theta_field_capacity_all_layers"] = 0.11
    elif mut == "conductivity":
        p["parameters"]["conductivity_multiplier"] = 2.0
    elif mut == "percent_ambiguous":
        p["parameters"]["theta_field_capacity_all_layers"] = 10                    # 10 would be a percent: refused as out of range for a volumetric fraction
    f = tmp_path / "p.json"
    f.write_text(json.dumps(p))
    with pytest.raises(PS.PresetError):
        PS.load_preset(f)


# ------------------------------------------------------------------------------------------------------------------------------ cases
def test_six_cases_only_and_index(generated):
    _, idx = generated
    assert list(idx["cases"]) == KEYS and set(idx["windows"]) == {"JORN4_A", "JORN4_B", "MOAB2_C"} and idx["stress"] == {"s1": 0.2017, "s1p5": 0.30255}
    assert idx["preset"]["sha256"] == PS.load_preset()["preset_sha256"]
    for k, c in idx["cases"].items():
        w, st = expect(k)
        assert c["theta_evaporation_stress"] == st and c["diffusion_coefficient"] == D_SEL and c["evaporation_factor"] == 1.0 and c["window"] == w


@pytest.mark.parametrize("key", KEYS)
def test_deltas_from_the_frozen_historical_template_are_exactly_the_declared_ones(generated, key):
    _, idx = generated
    w, st = expect(key)
    tmpl = json.loads((G.HIST / G.TEMPLATES[w][0]).read_text())
    new = load_case(idx, key)
    assert G.sha256_file(G.HIST / G.TEMPLATES[w][0]) == G.TEMPLATES[w][1]
    changed = {k for k in diff(tmpl, new) if not k.startswith("provenance.dry_stress_comparison")}
    allowed = {"name", "scenarios[0].name", "scenarios[0].diffusion_coefficient", "layers.theta_evaporation_stress", "pins.expected_sources.maple_syrup.package_sha256"} | {f"layers.theta_field_capacity[{i}]" for i in range(3)}
    assert changed == allowed, changed ^ allowed
    # literal values
    assert new["layers"]["theta_evaporation_stress"] == st and new["scenarios"][0]["diffusion_coefficient"] == D_SEL and new["scenarios"][0]["evaporation_factor"] == 1.0 and new["scenarios"][0]["bottom"] == "open"
    assert new["pins"]["expected_sources"]["maple_syrup"] == {"package_sha256": "4e38a472e268a56491b788d5b1271cc3c3da23bb1cc3221a8c59d90f3cc864d2", "files": 180}
    assert new["pins"]["expected_sources"]["maple"] == {"package_sha256": "fbc504bbcd1ec0c02e6a96089134a06187532455ef7ba5e6f146801c8313c02e", "files": 412}
    for k in ("window", "site", "air", "controls"):
        assert new[k] == tmpl[k]
    assert new["layers"]["theta_saturation"] == tmpl["layers"]["theta_saturation"] and new["layers"]["interface_conductivity_m_s"] == tmpl["layers"]["interface_conductivity_m_s"] and new["layers"]["theta_residual"] == [0.01] * 3
    assert new["controls"]["backends"] == ["numba", "numpy"] and new["controls"]["restart_after_day"] == tmpl["controls"]["restart_after_day"] < NDAYS[w]
    rec = new["provenance"]["dry_stress_comparison"]
    assert rec["deltas_from_template"]["syrup_package_pin"]["old"].startswith("37d89f23") and rec["deltas_from_template"]["theta_field_capacity"]["old"] == tmpl["layers"]["theta_field_capacity"] and rec["rain"] == 0 and rec["transpiration"] == 0
    assert "not held-out validation" in rec["d_provenance"] or "not held-out" in rec["d_provenance"]
    if w.startswith("JORN"):
        assert new["layers"]["theta_field_capacity"] == [0.10, 0.10, 0.10] and new["layers"]["theta_saturation"] == JORN_SAT and tmpl["layers"]["theta_field_capacity"] == OLD_FC_JORN
    else:
        fc = new["layers"]["theta_field_capacity"]
        assert fc == pytest.approx(MOAB_SR33_RECEIPT, abs=1e-15) and not np.allclose(fc, 0.10) and not np.allclose(fc, OLD_FC_MOAB, atol=0.05) and tmpl["layers"]["theta_field_capacity"] == OLD_FC_MOAB


def test_moab_properties_use_the_current_native_provider_default():
    with warnings.catch_warnings():
        warnings.simplefilter("error")                                  # the default method emits no deprecation warning
        m = PR.moab_properties()
    assert m["recipe"] == "saxton_rawls_2006_table1_eq1_18_explicit_density_v2_fc33_default" and len(m["provider_input_sha256"]) == 64
    assert m["theta_field_capacity"] == pytest.approx(MOAB_SR33_RECEIPT, abs=1e-15)
    tmpl = json.loads((G.HIST / "harmonic_MOAB2_C.json").read_text())
    assert m["theta_saturation"] == pytest.approx(tmpl["layers"]["theta_saturation"], rel=1e-12)
    assert m["saturated_conductivity_m_s"] == pytest.approx([x["Ks_m_s"] for x in tmpl["provenance"]["layers"]], rel=1e-12)
    assert all(0.0 <= a < 33.0 for a in m["air_entry_kpa"])                # raw MOAB air entries are valid: no override or floor is needed
    assert PR.layer_inputs()[0]["rho"] == pytest.approx(1504.54, rel=1e-12) and PR.layer_inputs()[1] == PR.MOAB_BW


# ------------------------------------------------------------------------------------------------------------------------------ sources
@pytest.mark.parametrize("key", KEYS)
def test_new_source_pair_passes_and_old_pin_is_refused_unmodified(generated, key):
    _, idx = generated
    cfg, sha = dd.load_config(idx["cases"][key]["path"])
    assert sha == idx["cases"][key]["sha256"]
    receipt = dd.source_receipt()
    dd.assert_expected_sources(cfg, receipt)
    assert (receipt["maple_syrup"]["package_sha256"], receipt["maple_syrup"]["files"]) == ("4e38a472e268a56491b788d5b1271cc3c3da23bb1cc3221a8c59d90f3cc864d2", 180)
    assert (receipt["maple"]["package_sha256"], receipt["maple"]["files"]) == ("fbc504bbcd1ec0c02e6a96089134a06187532455ef7ba5e6f146801c8313c02e", 412)
    w, _ = expect(key)
    old, _ = dd.load_config(G.HIST / G.TEMPLATES[w][0])                  # the UNMODIFIED historical template still carries the 37d89f23 pin
    assert old["pins"]["expected_sources"]["maple_syrup"]["package_sha256"].startswith("37d89f23")
    with pytest.raises(Exception, match="maple_syrup|37d89f23|package"):
        dd.assert_expected_sources(old, receipt)


# ------------------------------------------------------------------------------------------------------------------------------ real input preparation / native component
@pytest.mark.parametrize("key", KEYS)
def test_real_input_preparation_unchanged_qa_observations_and_inventory(generated, key):
    _, idx = generated
    w, _ = expect(key)
    cfg, _ = dd.load_config(idx["cases"][key]["path"])
    tmpl, _ = dd.load_config(G.HIST / G.TEMPLATES[w][0])
    prep, ref = dd.prepare_inputs(cfg), dd.prepare_inputs(tmpl)
    assert prep["n_days"] == NDAYS[w] and sum(int(np.isfinite(prep["obs"][v]).sum()) for v in dd.SENSOR_VERS) == OBS[w]
    assert np.array_equal(prep["initial"]["layer_water_m"], ref["initial"]["layer_water_m"]) and np.array_equal(prep["pet_mm_day"], ref["pet_mm_day"])
    for v in dd.SENSOR_VERS:
        assert np.array_equal(np.nan_to_num(prep["obs"][v], nan=-9), np.nan_to_num(ref["obs"][v], nan=-9))
    th = np.asarray(prep["initial"]["layer_theta"])
    assert (th > 0.03).all() and (th < 0.07).all() and (th < np.asarray(cfg["layers"]["theta_field_capacity"])).all() and (th < 0.10).all()      # initial moisture 3-7 vol. %, below every field capacity
    assert (np.asarray(cfg["layers"]["theta_field_capacity"]) > np.asarray(cfg["layers"]["theta_residual"])).all() and (np.asarray(cfg["layers"]["theta_field_capacity"]) <= np.asarray(cfg["layers"]["theta_saturation"])).all()
    assert prep["rain"] is not None


def test_53_observations_in_total():
    assert sum(OBS.values()) == 53 and OBS["JORN4_A"] + OBS["JORN4_B"] == 39


@pytest.mark.parametrize("key", KEYS)
def test_gravity_is_inactive_on_the_first_native_daily_step_so_the_field_capacity_is_not_identifiable(generated, key):
    """One REAL native daily step from the real initial inventory with the real weather PET: gravity on versus off gives bit-identical water (nothing exceeds field capacity), no bottom drainage, native mass budget."""
    _, idx = generated
    cfg, _ = dd.load_config(idx["cases"][key]["path"])
    prep = dd.prepare_inputs(cfg)
    scen = cfg["scenarios"][0]
    on = dd._layers(cfg, scen)
    lay = cfg["layers"]
    off = ss.soil_layers(lay["edges_m"].tolist(), theta_residual=lay["theta_residual"], theta_field_capacity=lay["theta_field_capacity"], theta_saturation=lay["theta_saturation"], theta_evaporation_stress=lay["theta_evaporation_stress"],
                         evaporation_factor=scen["evaporation_factor"], gravity_drainage=False, diffusion_mode="tietjen_donor_depth", interface_conductivity_m_s=lay["interface_conductivity_m_s"],
                         diffusion_coefficient=scen["diffusion_coefficient"], bottom_drainage=scen["bottom"])
    res = {}
    for name, layers in (("on", on), ("off", off)):
        col = ss.prepare_column(layers, np.ones(1, dtype=bool), 1.0, backend="numpy", available_bytes_fn=lambda: 1 << 40)
        w0 = prep["initial"]["layer_water_m"].reshape(-1, 1).copy()
        st = ss.column_state(col, w0.copy(), 0.0)
        res[name] = (w0, ss.step(col, st, 86400.0, added_water_m=np.zeros((3, 1)), potential_evaporation_m_s=np.array([prep["pet_mm_day"][0] / 1000.0 / 86400.0])))
    (w0, a), (_, b) = res["on"], res["off"]
    np.testing.assert_array_equal(a.state.water_m, b.state.water_m)
    assert a.budget.drainage_m3 == 0.0 and abs(a.percolation_m).max() == 0.0
    stored = a.state.water_m.sum() - w0.sum()
    assert abs(stored + a.evaporation_m[0] + a.budget.drainage_m3) <= a.budget.bound_m3 and a.evaporation_m[0] > 0.0


# ------------------------------------------------------------------------------------------------------------------------------ case rules (the comparison helper)
@pytest.mark.parametrize("key", KEYS)
def test_comparison_helper_accepts_the_generated_cases(generated, key):
    _, idx = generated
    cfg, _ = dd.load_config(idx["cases"][key]["path"])
    r = RUN.check_case(cfg)
    assert r["preset_sha256"] == PS.load_preset()["preset_sha256"] and r["kind"] == ("JORN preset" if key.startswith("JORN") else "MOAB own SR33")


@pytest.mark.parametrize("mut", ["jorn_fc", "moab_jorn_fc", "moab_old_fc", "d", "E", "sources", "backend", "stress"])
def test_comparison_helper_refuses_rule_violations(generated, mut):
    _, idx = generated
    key = "MOAB2_C_s1p5" if mut.startswith("moab") else "JORN4_A_s1p5"
    c = copy.deepcopy(load_case(idx, key))
    if mut == "jorn_fc":
        c["layers"]["theta_field_capacity"] = [0.12, 0.12, 0.12]
    elif mut == "moab_jorn_fc":
        c["layers"]["theta_field_capacity"] = [0.10, 0.10, 0.10]
    elif mut == "moab_old_fc":
        c["layers"]["theta_field_capacity"] = [0.37996672767136985, 0.39218936166045765, 0.39218936166045765]
    elif mut == "d":
        c["scenarios"][0]["diffusion_coefficient"] = 0.05
    elif mut == "E":
        c["scenarios"][0]["evaporation_factor"] = 0.8
    elif mut == "sources":
        c["pins"]["expected_sources"]["maple_syrup"]["package_sha256"] = "37d89f23dbf60626c85f85ea5dd6d1734d3bc7a4aa61d8f2fdfc29f237bcfbd0"
    elif mut == "backend":
        c["controls"]["backends"] = ["numpy"]
    elif mut == "stress":
        c["layers"]["theta_evaporation_stress"] = 0.25
    cfg = dd.validate_config(c)
    with pytest.raises(RUN.CaseRuleError):
        RUN.check_case(cfg)


# ------------------------------------------------------------------------------------------------------------------------------ runners
def test_runner_recipes_are_absolute_candidate_bound_and_have_bootstrap(generated):
    out, idx = generated
    boot = G.sha256_file(G.BOOTSTRAP)
    for key, c in idx["cases"].items():
        d = out / key
        assert Path(c["path"]).parent == d and G.sha256_file(d / "root_release_verified_bootstrap.py") == boot
        man = json.loads((d / "frozen_helper_manifest.json").read_text())
        assert set(man) == set(G.HELPERS) and all(G.sha256_file(G.CANDIDATE / k) == v for k, v in man.items())
        rc = json.loads(Path(c["runner_config"]).read_text())
        assert rc["cwd"] == str(G.CANDIDATE) and rc["output"] == c["output_dir"] and rc["probe_file"] == c["probe"] and not Path(rc["output"]).exists() and not Path(c["result_dir"]).exists()
        assert [s["sha256"] for s in rc["sources"]] == [G.MAPLE["package_sha256"], G.SYRUP["package_sha256"]] and rc["sources"][1]["root"] == str(G.CANDIDATE / "src")
        for k in ("path", "runner_config", "probe", "output_dir", "result_dir"):
            assert Path(c[k]).is_absolute()
        env = rc["env"]
        assert env["CUDA_VISIBLE_DEVICES"] == "" and "gpu_uuid" not in rc and all(Path(env[k]).is_absolute() for k in env if k.startswith("SYRUP_PHASE3_"))
        assert Path(env["SYRUP_PHASE3_RAW_ROOT"]).is_dir() and Path(env["SYRUP_PHASE3_INVENTORY"]).is_file()
        text = json.dumps(rc) + Path(rc["probe_file"]).read_text()
        assert str(G.CANDIDATE) in text and "jorn_sr33_160day_20261120/analysis_candidate" not in text and "jorn_fc_sensitivity_20261120" not in text and "jorn_evap_sensitivity_20261120" not in text
        assert "run.main" in Path(rc["probe_file"]).read_text()


def test_relative_out_resolves_absolute_and_writes_nothing_outside(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    idx = G.generate(Path("rel"))
    root = (tmp_path / "rel").resolve()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["rel"]
    for key, c in idx["cases"].items():
        for k in ("path", "runner_config", "probe", "output_dir", "result_dir"):
            assert Path(c[k]).is_absolute() and Path(c[k]).resolve().is_relative_to(root)
        assert (root / key / "root_release_verified_bootstrap.py").is_file()


def test_defaults_documented_and_agents_paragraph_appended():
    """The dated user-decision section must exist in AGENTS.md (anywhere: later unrelated appends must not break this) and the defaults document must state the selected values and the scope."""
    a = (G.CANDIDATE / "AGENTS.md").read_text()
    assert "## JORN case-level defaults (user decision) — 2026-11-22" in a
    assert "cases/subsurface_validation/jorn_defaults.json" in a
    assert "0.30255" in a and "10 vol. %" in a
    assert "NOT a new calibration" in a and "global physics default or MOAB default" in a
    assert "deferred by user decision" in a and "T = 0 is not mandated for future LEAF coupling." in a
    md = (G.CANDIDATE / "docs/subsurface_hydrology/jorn_defaults.md").read_text()
    assert "0.30255" in md
    assert "0.10" in md
    assert "NOT a new calibration" in md
    assert "global physics default" in md and "MOAB default" in md
    assert os.path.isfile(PS.PRESET_PATH)
