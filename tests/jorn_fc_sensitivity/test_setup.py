"""SETUP and focused NATIVE-component tests of the JORN 160-day uniform field-capacity x stress sensitivity (fourteen crossed cases). No 160-day run is executed. They check the generated configs against literal anchors of
the accepted SR33 1xK base (exact whitelist of differences), the real driver's closed validation / source guard / input preparation, the runner recipes (absolute paths, bootstrap beside every runner config, THIS
candidate), non-mutation of the base and helpers, and -- on the REAL native soil step with the real case column -- that gravity is triggered at field capacity rather than saturation, that the threshold does not
touch saturation, and that for one identical initial state a low field capacity drains more than a high one, with the native conservation bound. They need the Root runtime (MAPLE importable)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarks.jorn_fc_sensitivity import generate as G
from benchmarks.subsurface_validation import drydown_driver as dd
from benchmarks.subsurface_wet_validation import config as C
from benchmarks.subsurface_wet_validation import driver as D
from benchmarks.subsurface_wet_validation import inputs as I
from maple_syrup import subsurface as ss

# ---- independent literal anchors (accepted SR33 1xK base) -------------------------------------------------------------------------------------------------------------------------------
FITTED_STRESS = 0.20170581117963332
SAT = [0.4643018867924528, 0.4651712600744756, 0.4586310940906557]
WP = [0.039508152304, 0.04043829918009047, 0.046312178008]
IFACE_1X = [2.113350339316645e-05 / 2, 1.9595261725153497e-05 / 2]
MATRIX_K_1X = 2.1228725891491996e-05 / 2
D_COEF = 0.0010239461376024856
SR33_FC = [0.09746408113338643, 0.09939508831059733, 0.10913571555870182]
FCS = [9, 10, 11, 12, 13, 14, 15]
KEYS = [f"fc{p:02d}_{s}" for s in ("s1", "s1p5") for p in FCS]


def expect(key):
    p, s = int(key[2:4]), key.split("_")[1]
    return p / 100.0, (0.2017 if s == "s1" else 0.2017 * 1.5)


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    before = (G.sha256_file(G.BASE_CONFIG), {rel: G.sha256_file(G.CANDIDATE / rel) for rel in G.G.HELPERS})
    out = tmp_path_factory.mktemp("fc")
    paths = G.generate(out)
    after = (G.sha256_file(G.BASE_CONFIG), {rel: G.sha256_file(G.CANDIDATE / rel) for rel in G.G.HELPERS})
    assert before == after                                              # base and frozen helpers are not mutated by generation
    return out, paths, json.loads(G.BASE_CONFIG.read_text())


def load(paths, key, variant="numba"):
    return json.loads(Path(paths["cases"][key]["variants"][variant]["path"]).read_text())


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


# ------------------------------------------------------------------------------------------------------------------------------ base / structure
def test_base_identity_and_literal_anchors(generated):
    _, _, base = generated
    assert G.sha256_file(G.BASE_CONFIG) == "f024054f5198ec8358c95f7a33683e6228084e6af246f9790cde586d437f3d2a"
    assert base["layers"]["theta_evaporation_stress"] == FITTED_STRESS and base["physics"]["evaporation_factor"] == 1.0 and base["physics"]["diffusion_coefficient"] == D_COEF
    assert base["layers"]["theta_field_capacity"] == SR33_FC and base["layers"]["theta_saturation"] == SAT
    assert base["matrix"]["conductivity_m_s"] == MATRIX_K_1X and base["layers"]["interface_conductivity_m_s"] == IFACE_1X
    assert base["expected_sources"]["maple_syrup"]["package_sha256"].startswith("4e38a472") and base["expected_sources"]["maple"]["package_sha256"].startswith("fbc504bb")


def test_fourteen_crossed_cases_families_and_42_runs(generated):
    _, paths, _ = generated
    assert list(paths["cases"]) == KEYS and len(KEYS) == 14
    assert paths["families"]["stress1"] == KEYS[:7] and paths["families"]["stress1p5"] == KEYS[7:] and all(paths["families"][f"FC{p:02d}"] == [f"fc{p:02d}_s1", f"fc{p:02d}_s1p5"] for p in FCS)
    assert sum(len(v["variants"]) for v in paths["cases"].values()) == 42
    for key in KEYS:
        fc, stress = expect(key)
        c = paths["cases"][key]
        assert c["field_capacity"] == pytest.approx(fc, abs=1e-15) and c["theta_evaporation_stress"] == pytest.approx(stress, abs=1e-15) and c["stress_multiplier"] == (1.0 if key.endswith("_s1") else 1.5)


@pytest.mark.parametrize("key", KEYS)
def test_only_the_requested_fields_differ_from_the_base(generated, key):
    _, paths, base = generated
    new = load(paths, key)
    fc, stress = expect(key)
    assert new["layers"]["theta_field_capacity"] == [fc, fc, fc] and new["layers"]["theta_evaporation_stress"] == stress
    assert new["layers"]["theta_evaporation_stress"] != FITTED_STRESS and abs(new["layers"]["theta_evaporation_stress"] - (0.2017 if key.endswith("_s1") else 0.30255)) < 1e-15
    changed = {k for k in diff(base, new) if not k.startswith("provenance.field_capacity_sensitivity")}
    assert changed == {"layers.theta_field_capacity[0]", "layers.theta_field_capacity[1]", "layers.theta_field_capacity[2]", "layers.theta_evaporation_stress", "name", "provenance.role"}
    for k in ("site", "forcing", "swc", "initial", "matrix", "expected_sources", "physics", "controls", "storm"):
        assert new[k] == base[k], k
    assert new["provenance"]["layer_properties"] == base["provenance"]["layer_properties"]          # the historical SR33 values are kept as context, not rewritten
    assert new["layers"]["theta_saturation"] == SAT and new["layers"]["theta_residual"] == [0.01] * 3 and new["layers"]["edges_m"] == [0.0, 0.1, 0.2, 0.3]
    assert new["layers"]["interface_conductivity_m_s"] == IFACE_1X and new["matrix"]["conductivity_m_s"] == MATRIX_K_1X and new["physics"]["diffusion_coefficient"] == D_COEF
    assert new["physics"]["evaporation_factor"] == 1.0 and new["physics"]["transpiration_factor"] == 0 and new["physics"]["bottom"] == "open"
    rec = new["provenance"]["field_capacity_sensitivity"]
    assert rec["field_capacity"] == fc and rec["theta_evaporation_stress"] == stress and "NOT recomputed" in rec["override_note"] and "0.20170581117963332" in rec["stress_note"] and "never newly fitted" in rec["stress_note"] and "Tietjen" in rec["calibration_chain"]
    assert key in new["name"]


def test_controls_differ_only_in_backend_and_step(generated):
    _, paths, _ = generated
    for key in KEYS:
        base = load(paths, key)
        assert diff(base, load(paths, key, "numpy")) == {"controls.backend"} and diff(base, load(paths, key, "numba_dt900")) == {"storm.max_dt_s"}
        assert load(paths, key, "numba_dt900")["storm"]["max_dt_s"] == 900 and load(paths, key, "numpy")["controls"]["backend"] == "numpy"


# ------------------------------------------------------------------------------------------------------------------------------ domain / real driver
@pytest.mark.parametrize("key", KEYS)
@pytest.mark.parametrize("variant", ["numba", "numpy", "numba_dt900"])
def test_closed_validation_and_new_source_guard(generated, key, variant):
    _, paths, _ = generated
    cfg, sha = C.load_config(paths["cases"][key]["variants"][variant]["path"])
    assert sha == paths["cases"][key]["variants"][variant]["sha256"]
    receipt = D.check_sources(cfg)
    assert (receipt["maple_syrup"]["package_sha256"], receipt["maple_syrup"]["files"]) == (G.SYRUP["package_sha256"], 180) and (receipt["maple"]["package_sha256"], receipt["maple"]["files"]) == (G.MAPLE["package_sha256"], 412)


@pytest.mark.parametrize("key", KEYS)
def test_domain_initial_water_below_fc9_fc_between_wp_and_sat_and_stress_above_fc(generated, key):
    _, paths, _ = generated
    cfg, _ = C.load_config(paths["cases"][key]["variants"]["numba"]["path"])
    prep = I.prepare_inputs(cfg)
    fc = cfg["layers"]["theta_field_capacity"]
    assert all(t < 0.09 for t in prep["initial"]["layer_theta"])                       # every layer starts below even the lowest field capacity (9 vol. %)
    assert (fc <= cfg["layers"]["theta_saturation"]).all() and (fc > cfg["layers"]["theta_residual"]).all() and (fc > np.array(WP)).all()
    assert cfg["layers"]["theta_evaporation_stress"] > fc.max() and cfg["layers"]["theta_evaporation_stress"] <= 1.0
    assert prep["initial"]["total_water_m"] * 1000 == pytest.approx(22.9424, abs=1e-9) and prep["rain"]["total_mm_decimal"] == "25.088" and prep["n_days"] == 160


def test_old_source_pins_refused_not_rebound(generated):
    _, paths, _ = generated
    old = json.loads((G.TASKS / "jorn_160_day_run_20261119/case_numba.json").read_text())
    with pytest.raises(D.ControlFailure, match="maple_syrup"):
        D.check_sources(C.validate_config(old))
    bad = load(paths, "fc12_s1")
    bad["expected_sources"]["maple_syrup"] = {"package_sha256": "37d89f23dbf60626c85f85ea5dd6d1734d3bc7a4aa61d8f2fdfc29f237bcfbd0", "files": 180}
    with pytest.raises(D.ControlFailure):
        D.check_sources(C.validate_config(bad))


def test_real_input_preparation_identical_to_base(generated):
    _, paths, base = generated
    ref = I.prepare_inputs(C.validate_config(base))
    same = lambda a, b: json.dumps(I.jsonable(a), sort_keys=True) == json.dumps(I.jsonable(b), sort_keys=True)
    for key in KEYS:
        prep = I.prepare_inputs(C.load_config(paths["cases"][key]["variants"]["numba"]["path"])[0])
        assert same(prep["initial"], ref["initial"]) and same(prep["target_counts"], ref["target_counts"]) and same(prep["theta0_points"], ref["theta0_points"]) and np.array_equal(prep["pet_mm_day"], ref["pet_mm_day"])


# ------------------------------------------------------------------------------------------------------------------------------ native component control
def _column(paths, key, d=None):
    cfg, _ = C.load_config(paths["cases"][key]["variants"]["numba"]["path"])
    lay = dd._layers(cfg, {"evaporation_factor": 1.0, "diffusion_coefficient": cfg["physics"]["diffusion_coefficient"] if d is None else d, "bottom": cfg["physics"]["bottom"]})
    return ss.prepare_column(lay, np.ones(1, dtype=bool), 1.0, backend="numpy", available_bytes_fn=lambda: 1 << 40), cfg


def _step(col, theta, dt=3600.0):
    water = np.array([[t * 0.1] for t in theta])                       # three 10 cm layers, 1 m2
    return water, ss.step(col, ss.column_state(col, water.copy(), 0.0), dt, added_water_m=np.zeros((3, 1)), potential_evaporation_m_s=np.zeros(1))


def test_gravity_is_triggered_at_field_capacity_not_saturation(generated):
    _, paths, _ = generated
    state = [0.12, 0.05, 0.05]                                          # layer 0 far BELOW saturation (46 vol. %), above FC 9 and below FC 15
    col9, cfg9 = _column(paths, "fc09_s1", d=0.0)                      # diffusion off only to isolate the gravity stage; the real native step otherwise
    col15, cfg15 = _column(paths, "fc15_s1", d=0.0)
    _, r9 = _step(col9, state)
    w15, r15 = _step(col15, state)
    assert cfg9["layers"]["theta_saturation"].tolist() == SAT == cfg15["layers"]["theta_saturation"].tolist() and state[0] < SAT[0]
    # FC 9: layer 0 is cut back to its field capacity and the excess moves to layer 1 (hand values, native bound); no water leaves
    bound = r9.budget.bound_m3
    assert abs(r9.state.water_m[0, 0] - 0.09 * 0.1) <= bound
    assert abs(r9.state.water_m[1, 0] - (0.005 + (0.12 - 0.09) * 0.1)) <= bound
    assert r9.budget.drainage_m3 <= bound
    # FC 15: the very same state is below field capacity: nothing moves
    np.testing.assert_allclose(r15.state.water_m, w15, rtol=0, atol=r15.budget.bound_m3)
    assert r15.budget.drainage_m3 <= r15.budget.bound_m3


def test_low_fc_drains_more_than_high_fc_for_the_same_state_with_native_bound(generated):
    _, paths, _ = generated
    state = [0.14, 0.14, 0.14]
    out, bounds = {}, {}
    for p in FCS:
        col, _ = _column(paths, f"fc{p:02d}_s1", d=0.0)
        w, r = _step(col, state)
        bound = r.budget.bound_m3
        stored = r.state.water_m.sum() - w.sum()
        assert abs(stored + r.evaporation_m[0] + r.budget.drainage_m3) <= bound                  # exact native mass budget (no rain, no E)
        want = max(0.0, w.sum() - 3 * (p / 100.0) * 0.1)                                         # independent hand value: everything above FC leaves through the open bottom
        assert abs(r.budget.drainage_m3 - want) <= bound
        out[p] = r.budget.drainage_m3
        bounds[p] = bound
    assert out[9] > out[10] > out[11] > out[12] > out[13] > 0
    assert out[14] <= bounds[14] and out[15] <= bounds[15]
    with_d = _step(_column(paths, "fc09_s1")[0], state)[1]                                       # with the real d and interface conductivities the conclusion is unchanged
    assert with_d.budget.drainage_m3 > _step(_column(paths, "fc15_s1")[0], state)[1].budget.drainage_m3


def test_stress_value_does_not_change_the_gravity_stage(generated):
    _, paths, _ = generated
    a = _step(_column(paths, "fc10_s1", d=0.0)[0], [0.13, 0.13, 0.13])[1]
    b = _step(_column(paths, "fc10_s1p5", d=0.0)[0], [0.13, 0.13, 0.13])[1]
    np.testing.assert_array_equal(a.state.water_m, b.state.water_m)                    # no E demand: the stress threshold cannot enter


# ------------------------------------------------------------------------------------------------------------------------------ runners
def test_runner_recipes_are_absolute_candidate_bound_and_have_bootstrap(generated):
    out, paths, _ = generated
    boot = G.sha256_file(G.BOOTSTRAP)
    for key in KEYS:
        d = out / key
        assert G.sha256_file(d / "root_release_verified_bootstrap.py") == boot
        man = json.loads((d / "frozen_helper_manifest.json").read_text())
        assert set(man) == set(G.G.HELPERS) and all(G.sha256_file(G.CANDIDATE / k) == v for k, v in man.items())
        for variant, e in paths["cases"][key]["variants"].items():
            rc = json.loads(Path(e["runner_config"]).read_text())
            assert Path(e["runner_config"]).parent == d and all(Path(e[k]).is_absolute() for k in ("path", "runner_config", "probe", "output_dir", "result_dir"))
            assert rc["cwd"] == str(G.CANDIDATE) and rc["output"] == e["output_dir"] and rc["probe_file"] == e["probe"] and not Path(rc["output"]).exists() and not Path(e["result_dir"]).exists()
            assert [s["sha256"] for s in rc["sources"]] == [G.MAPLE["package_sha256"], G.SYRUP["package_sha256"]] and rc["sources"][1]["root"] == str(G.CANDIDATE / "src")
            text = json.dumps(rc) + Path(rc["probe_file"]).read_text()
            assert rc["env"]["CUDA_VISIBLE_DEVICES"] == "" and "gpu_uuid" not in rc and "jorn_sr33_160day_20261120/analysis_candidate" not in text and "jorn_evap_sensitivity_20261120" not in text
            assert str(G.CANDIDATE) in Path(rc["probe_file"]).read_text() and f"CASE='{variant}'" in Path(rc["probe_file"]).read_text()


def test_relative_out_resolves_absolute_and_writes_nothing_outside(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    paths = G.generate(Path("rel"))
    root = (tmp_path / "rel").resolve()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["rel"]
    for key, c in paths["cases"].items():
        for e in c["variants"].values():
            for k in ("path", "runner_config", "probe"):
                assert Path(e[k]).is_absolute() and Path(e[k]).resolve().is_relative_to(root)
            assert Path(e["output_dir"]).resolve().is_relative_to(root)
        assert (root / key / "root_release_verified_bootstrap.py").is_file()
