"""Stage-1 SETUP and focused NATIVE-component tests of the JORN 160-day SR33 1xK evaporation sensitivity (eight unique cases). No 160-day run is executed. They check the generated configs against literal
anchors of the accepted SR33 1xK base (exact whitelist of differences), the real driver's closed validation / source guard / input preparation, the runner recipes (absolute paths, bootstrap beside every runner
config, THIS candidate), and -- on the real native soil step with the real case configuration -- that a stress multiplier r is equivalent to an evaporation factor E/r^2 on the quadratic branch, including the
above-field-capacity gravity stage and the supply limit. They need the Root runtime (MAPLE importable)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarks.jorn_evap_sensitivity import generate as G
from benchmarks.subsurface_validation import drydown_driver as dd
from benchmarks.subsurface_wet_validation import config as C
from benchmarks.subsurface_wet_validation import driver as D
from benchmarks.subsurface_wet_validation import inputs as I
from maple_syrup import subsurface as ss

# ---- independent literal anchors (accepted SR33 1xK base, prior study) -----------------------------------------------------------------------------------------------------------------
S0 = 0.20170581117963332
FC = [0.09746408113338643, 0.09939508831059733, 0.10913571555870182]
SAT = [0.4643018867924528, 0.4651712600744756, 0.4586310940906557]
IFACE_1X = [2.113350339316645e-05 / 2, 1.9595261725153497e-05 / 2]
MATRIX_K_1X = 2.1228725891491996e-05 / 2
D_COEF = 0.0010239461376024856
EXPECT = {"baseline": (1.0, S0), "e0p8": (0.8, S0), "e0p6": (0.6, S0), "e0p4": (0.4, S0), "s1p5": (1.0, S0 * 1.5), "s2": (1.0, S0 * 2.0), "s2p5": (1.0, S0 * 2.5), "s3": (1.0, S0 * 3.0)}


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    out = tmp_path_factory.mktemp("evap")
    return out, G.generate(out), json.loads(G.BASE_CONFIG.read_text())


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
    assert base["layers"]["theta_evaporation_stress"] == S0 and base["physics"]["evaporation_factor"] == 1.0 and base["physics"]["diffusion_coefficient"] == D_COEF
    assert base["layers"]["theta_field_capacity"] == FC and base["layers"]["theta_saturation"] == SAT
    assert base["matrix"]["conductivity_m_s"] == MATRIX_K_1X and base["layers"]["interface_conductivity_m_s"] == IFACE_1X
    assert base["expected_sources"]["maple_syrup"]["package_sha256"].startswith("4e38a472") and base["expected_sources"]["maple"]["package_sha256"].startswith("fbc504bb")


def test_eight_unique_cases_two_families_and_24_runs(generated):
    _, paths, _ = generated
    assert set(paths["cases"]) == set(EXPECT) and len(paths["cases"]) == 8
    assert paths["families"] == {"E": ["baseline", "e0p8", "e0p6", "e0p4"], "stress": ["baseline", "s1p5", "s2", "s2p5", "s3"]}
    assert sum(len(v["variants"]) for v in paths["cases"].values()) == 24
    for key, (factor, stress) in EXPECT.items():
        c = paths["cases"][key]
        assert c["evaporation_factor"] == factor and c["theta_evaporation_stress"] == stress


@pytest.mark.parametrize("key", list(EXPECT))
def test_only_two_swept_locations_differ_from_the_base(generated, key):
    _, paths, base = generated
    new = load(paths, key)
    factor, stress = EXPECT[key]
    assert new["physics"]["evaporation_factor"] == factor and new["layers"]["theta_evaporation_stress"] == stress
    changed = {k for k in diff(base, new) if not k.startswith("provenance.evaporation_sensitivity")}
    allowed = {"name", "provenance.role"}
    allowed |= {"physics.evaporation_factor"} if factor != 1.0 else set()
    allowed |= {"layers.theta_evaporation_stress"} if stress != S0 else set()
    assert changed == allowed, changed ^ allowed
    # everything physical is the 1xK / SR33 / forcing / initial-moisture base, bit for bit
    for k in ("site", "forcing", "swc", "initial", "matrix", "expected_sources", "controls", "storm"):
        assert new[k] == base[k], k
    assert new["layers"]["theta_field_capacity"] == FC and new["layers"]["theta_saturation"] == SAT and new["layers"]["theta_residual"] == [0.01] * 3
    assert new["layers"]["interface_conductivity_m_s"] == IFACE_1X and new["matrix"]["conductivity_m_s"] == MATRIX_K_1X
    assert new["physics"]["diffusion_coefficient"] == D_COEF and new["physics"]["transpiration_factor"] == 0 and new["physics"]["bottom"] == "open"
    rec = new["provenance"]["evaporation_sensitivity"]
    assert rec["theta_evaporation_stress"] == stress and rec["evaporation_factor"] == factor and rec["baseline_stress"] == S0 and "Tietjen" in rec["calibration_chain"] and "joint" in rec["calibration_chain"]
    assert rec["effective_quadratic_coefficient_E_over_r2"] == factor / (stress / S0) ** 2 or abs(rec["effective_quadratic_coefficient_E_over_r2"] - factor / (stress / S0) ** 2) < 1e-14
    assert key in new["name"] and (stress > SAT[0]) == rec["stress_above_saturation"]


def test_controls_differ_only_in_backend_and_step(generated):
    _, paths, _ = generated
    for key in EXPECT:
        base = load(paths, key)
        assert diff(base, load(paths, key, "numpy")) == {"controls.backend"} and diff(base, load(paths, key, "numba_dt900")) == {"storm.max_dt_s"}
        assert load(paths, key, "numba_dt900")["storm"]["max_dt_s"] == 900 and load(paths, key, "numpy")["controls"]["backend"] == "numpy"


# ------------------------------------------------------------------------------------------------------------------------------ real driver validation / domain
@pytest.mark.parametrize("key", list(EXPECT))
@pytest.mark.parametrize("variant", ["numba", "numpy", "numba_dt900"])
def test_closed_validation_and_new_source_guard(generated, key, variant):
    _, paths, _ = generated
    cfg, sha = C.load_config(paths["cases"][key]["variants"][variant]["path"])
    assert sha == paths["cases"][key]["variants"][variant]["sha256"]
    receipt = D.check_sources(cfg)
    assert receipt["maple_syrup"]["package_sha256"] == G.SYRUP["package_sha256"] and receipt["maple"]["package_sha256"] == G.MAPLE["package_sha256"]


def test_stress_above_saturation_is_accepted_and_nothing_else_moves(generated):
    _, paths, _ = generated
    cfg, _ = C.load_config(paths["cases"]["s3"]["variants"]["numba"]["path"])
    stress = float(cfg["layers"]["theta_evaporation_stress"])
    assert stress == S0 * 3.0 and stress > max(SAT) and stress <= 1.0                  # above SAT by design: a normalisation scale, valid for the closed config AND the native layers
    lay = dd._layers(cfg, {"evaporation_factor": 1.0, "diffusion_coefficient": D_COEF, "bottom": "open"})
    assert lay.theta_evaporation_stress == stress
    np.testing.assert_array_equal(lay.theta_saturation, SAT) if hasattr(lay, "theta_saturation") else None
    assert cfg["layers"]["theta_saturation"].tolist() == SAT and cfg["layers"]["theta_field_capacity"].tolist() == FC and stress > cfg["layers"]["theta_saturation"].max()


def test_old_source_pins_refused_not_rebound(generated):
    _, paths, _ = generated
    old = json.loads((G.TASKS / "jorn_160_day_run_20261119/case_numba.json").read_text())
    with pytest.raises(D.ControlFailure, match="maple_syrup"):
        D.check_sources(C.validate_config(old))
    bad = load(paths, "s2")
    bad["expected_sources"]["maple_syrup"] = {"package_sha256": "37d89f23dbf60626c85f85ea5dd6d1734d3bc7a4aa61d8f2fdfc29f237bcfbd0", "files": 180}
    with pytest.raises(D.ControlFailure):
        D.check_sources(C.validate_config(bad))


def test_real_input_preparation_identical_to_base_no_double_water(generated):
    _, paths, base = generated
    ref = I.prepare_inputs(C.validate_config(base))
    same = lambda a, b: json.dumps(I.jsonable(a), sort_keys=True) == json.dumps(I.jsonable(b), sort_keys=True)
    for key in EXPECT:
        cfg, _ = C.load_config(paths["cases"][key]["variants"]["numba"]["path"])
        prep = I.prepare_inputs(cfg)
        assert same(prep["initial"], ref["initial"]) and same(prep["target_counts"], ref["target_counts"]) and same(prep["theta0_points"], ref["theta0_points"])
        assert prep["initial"]["total_water_m"] * 1000 == pytest.approx(22.9424, abs=1e-9) and prep["rain"]["total_mm_decimal"] == "25.088" and prep["n_days"] == 160
        assert np.array_equal(prep["pet_mm_day"], ref["pet_mm_day"])


# ------------------------------------------------------------------------------------------------------------------------------ native component control
def _column(factor, stress):
    c = json.loads(G.BASE_CONFIG.read_text())
    c["layers"]["theta_evaporation_stress"], c["physics"]["evaporation_factor"] = stress, factor
    cfg = C.validate_config(c)
    lay = dd._layers(cfg, {"evaporation_factor": factor, "diffusion_coefficient": cfg["physics"]["diffusion_coefficient"], "bottom": cfg["physics"]["bottom"]})
    return ss.prepare_column(lay, np.ones(1, dtype=bool), 1.0, backend="numpy", available_bytes_fn=lambda: 1 << 40)


def _step(col, theta, pet_m_day, dt=86400.0):
    water = np.array([[t * 0.1] for t in theta])                        # 10 cm layers, 1 m2
    state = ss.column_state(col, water.copy(), 0.0)
    return water, ss.step(col, state, dt, added_water_m=np.zeros((3, 1)), potential_evaporation_m_s=np.array([pet_m_day / dt]))


@pytest.mark.parametrize("r", [1.5, 2.0, 2.5, 3.0])
@pytest.mark.parametrize("theta", [[0.06, 0.08, 0.09], [0.0102, 0.08, 0.09], [0.12, 0.09, 0.07]], ids=["below_FC", "near_residual", "above_FC_gravity_first"])
def test_stress_multiplier_equals_E_over_r2_on_the_real_native_step(r, theta):
    a, b = _column(1.0, S0 * r), _column(1.0 / (r * r), S0)
    wa, ra = _step(a, theta, 0.006)
    _, rb = _step(b, theta, 0.006)
    assert ra.evaporation_m[0] == pytest.approx(rb.evaporation_m[0], rel=1e-14, abs=0) and ra.evaporation_unmet_m[0] == rb.evaporation_unmet_m[0] == 0.0
    np.testing.assert_allclose(ra.state.water_m, rb.state.water_m, rtol=0, atol=1e-18)
    # exact mass budget of the real step, both columns: storage change = - evaporation - drainage (no added water, no surface return)
    for res in (ra, rb):
        stored = res.state.water_m.sum() - wa.sum()
        assert abs(stored + res.evaporation_m[0] + res.budget.drainage_m3) <= res.budget.bound_m3 and res.budget.evaporation_m3 == res.evaporation_m[0]
    # independent hand value: the E stage sees the layer-0 content AFTER the gravity stage (<= field capacity), on the quadratic branch
    th0 = min(theta[0], FC[0])
    assert th0 < S0
    assert ra.evaporation_m[0] == pytest.approx((0.006) * (th0 / (S0 * r)) ** 2, rel=1e-12)


def test_top_moisture_at_the_E_stage_is_field_capacity_not_the_wet_branch():
    """Layer 0 starts far above field capacity (and above the baseline stress): the gravity stage runs first, so the E stage sees exactly field capacity and the quadratic branch (theta/stress)^2."""
    for stress in (S0, S0 * 3.0):
        _, ra = _step(_column(1.0, stress), [0.30, 0.30, 0.30], 0.006)
        assert ra.evaporation_m[0] == pytest.approx(0.006 * (FC[0] / stress) ** 2, rel=1e-12)
        assert ra.evaporation_m[0] < 0.006                                # never the full (wet-branch) potential


def test_supply_limit_is_respected_equally():
    a, b = _column(1.0, S0 * 3.0), _column(1.0 / 9.0, S0)
    wa, ra = _step(a, [0.0102, 0.08, 0.09], 50.0)                       # demand far above the water above residual
    _, rb = _step(b, [0.0102, 0.08, 0.09], 50.0)
    avail = 0.0102 * 0.1 - 0.01 * 0.1
    assert ra.evaporation_m[0] == pytest.approx(avail, rel=1e-12) and rb.evaporation_m[0] == pytest.approx(avail, rel=1e-12)
    assert ra.evaporation_unmet_m[0] > 0 and rb.evaporation_unmet_m[0] > 0
    assert abs(ra.state.water_m.sum() - (wa.sum() - ra.evaporation_m[0] - ra.budget.drainage_m3)) <= ra.budget.bound_m3  # native MAPLE-derived bound; diffusion only redistributes


def test_E_family_scales_the_demand_linearly_on_the_native_step():
    base = _step(_column(1.0, S0), [0.06, 0.08, 0.09], 0.006)[1].evaporation_m[0]
    for f in (0.8, 0.6, 0.4):
        assert _step(_column(f, S0), [0.06, 0.08, 0.09], 0.006)[1].evaporation_m[0] == pytest.approx(f * base, rel=1e-14)


# ------------------------------------------------------------------------------------------------------------------------------ runners
def test_runner_recipes_are_absolute_candidate_bound_and_have_bootstrap(generated):
    out, paths, _ = generated
    boot = G.sha256_file(G.BOOTSTRAP)
    for key in EXPECT:
        d = out / key
        assert G.sha256_file(d / "root_release_verified_bootstrap.py") == boot
        man = json.loads((d / "frozen_helper_manifest.json").read_text())
        assert set(man) == set(G.G.HELPERS) and all(G.sha256_file(G.CANDIDATE / k) == v for k, v in man.items())
        for variant, e in paths["cases"][key]["variants"].items():
            rc = json.loads(Path(e["runner_config"]).read_text())
            assert Path(e["runner_config"]).parent == d and all(Path(e[k]).is_absolute() for k in ("path", "runner_config", "probe", "output_dir", "result_dir"))
            assert rc["cwd"] == str(G.CANDIDATE) and rc["output"] == e["output_dir"] and rc["probe_file"] == e["probe"] and not Path(rc["output"]).exists() and not Path(e["result_dir"]).exists()
            assert [s["sha256"] for s in rc["sources"]] == [G.MAPLE["package_sha256"], G.SYRUP["package_sha256"]] and rc["sources"][1]["root"] == str(G.CANDIDATE / "src")
            assert rc["env"]["CUDA_VISIBLE_DEVICES"] == "" and "gpu_uuid" not in rc and "jorn_sr33_160day_20261120/analysis_candidate" not in json.dumps(rc)
            assert str(G.CANDIDATE) in Path(rc["probe_file"]).read_text() and f"CASE='{variant}'" in Path(rc["probe_file"]).read_text()
            assert "jorn_sr33_160day_20261120/analysis_candidate" not in Path(rc["probe_file"]).read_text()


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
