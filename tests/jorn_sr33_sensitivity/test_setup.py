"""Stage-1 SETUP tests of the JORN 160-day SR33 sensitivity configs. No primary 160-day run is executed: they check the generated configs against (a) hard-coded independent anchors
(literal JORN inputs, an INDEPENDENT Saxton-Rawls 2006 Table 1 implementation written here, the original study's published invariants), (b) the exact whitelist of differences from the original
config, (c) the real native provider, and (d) the real driver's closed validation, source guard and input preparation (initial water). Need the Root runtime (MAPLE importable)."""
from __future__ import annotations

import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarks.jorn_sr33_sensitivity import generate as G
from benchmarks.subsurface_wet_validation import config as C
from benchmarks.subsurface_wet_validation import driver as D
from benchmarks.subsurface_wet_validation import inputs as I

# ---- independent anchors (literal, not read from the generator) ----------------------------------------------------------------------------------------------------------------------
TEX = [(0.867, 0.062, 0.071, 0.3448, 1419.6), (0.862881888208572, 0.06433359668180921, 0.07278451510961881, 0.3211345842385937, 1417.2961608026395), (0.837, 0.079, 0.084, 0.1724, 1434.6276006597622)]
FC_EXPECT = [0.09746408113338648, 0.09939508831059735, 0.10913571555870187]       # Root's 80-digit audit values (rounded); provider float64 within a few ulp
OLD_FC = [0.39760410394534984, 0.39866650157195227, 0.39508647981211864]            # previous hybrid 6 kPa values (original config)
SAT = [0.4643018867924528, 0.4651712600744756, 0.4586310940906557]
WP = [0.039508152304, 0.04043829918009047, 0.046312178008]
KS_M_S = [3.355180988596539e-05, 3.3252157391757436e-05, 2.8981216914528297e-05]
KS_MM_H = [120.78651558947541, 119.70776661032677, 104.33238089230187]
MATRIX_KS, MATRIX_K, IFACE = 3.355180988596539e-05, 1.0614362945745998e-05, [1.0566751696583226e-05, 9.797630862576748e-06]


def sr33(S, Si, C_, OM, rho_kg):
    """Independent Saxton & Rawls (2006) Table 1 Eq 2, 3, 6-10: density-adjusted theta_33 and theta_S (written here from the paper, not from the provider)."""
    t = -0.251 * S + 0.195 * C_ + 0.011 * OM + 0.006 * S * OM - 0.027 * C_ * OM + 0.452 * S * C_ + 0.299
    t33n = t + (1.283 * t * t - 0.374 * t - 0.015)
    u = 0.278 * S + 0.034 * C_ + 0.022 * OM - 0.018 * S * OM - 0.027 * C_ * OM - 0.584 * S * C_ + 0.078
    tg = u + (0.636 * u - 0.107)
    tsn = t33n + tg - 0.097 * S + 0.043
    rho_n = (1.0 - tsn) * 2.65
    ts = 1.0 - rho_kg / 1000.0 / 2.65
    df = rho_kg / 1000.0 / rho_n
    assert 0.9 <= df <= 1.3
    return t33n - 0.2 * (tsn - ts), ts


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    out = tmp_path_factory.mktemp("sr33")
    paths = G.generate(out)
    old = json.loads(G.OLD_CONFIG.read_text())
    return out, paths, old


def load(paths, key, variant="numba"):
    return json.loads(Path(paths[key][variant]["path"]).read_text())


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


# ------------------------------------------------------------------------------------------------------------------------------ baseline / independent FC
def test_original_baseline_identity_and_invariants(generated):
    _, _, old = generated
    assert G.sha256_file(G.OLD_CONFIG) == "e7513f545b074da5fc5627e96b79f62ed787863fb32c9df545a9422547e521ee"
    assert old["layers"]["theta_field_capacity"] == OLD_FC and old["layers"]["theta_saturation"] == SAT
    assert old["matrix"]["ks_top_m_s"] == MATRIX_KS and old["matrix"]["conductivity_m_s"] == MATRIX_K and old["layers"]["interface_conductivity_m_s"] == IFACE
    assert old["expected_sources"]["maple_syrup"]["package_sha256"].startswith("37d89f23") and old["expected_sources"]["maple"]["package_sha256"].startswith("fbc504bb")


def test_independent_sr33_matches_anchor_values():
    for (S, Si, C_, OM, rho), want in zip(TEX, FC_EXPECT):
        t33, _ts = sr33(S, Si, C_, OM, rho)
        assert math.isclose(t33, want, rel_tol=0, abs_tol=2e-16)
    # the corresponding saturation anchors of the original study are reproduced (rho/2.65 identity)
    assert [sr33(*t)[1] for t in TEX] == pytest.approx(SAT, abs=1e-15)


def test_native_provider_default_is_sr33_and_unchanged_properties(generated):
    inp = G.audit_inputs(json.loads(G.OLD_CONFIG.read_text()))
    got = np.array([inp["sand"], inp["silt"], inp["clay"], inp["om_percent"], inp["bulk_density_kg_m3"]]).T
    assert got == pytest.approx(np.array(TEX), rel=1e-14, abs=0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")      # the default method must not raise the legacy deprecation warning
        nat = G.native_properties(inp)
    assert nat["recipe"]["field_capacity_method"] == "saxton_rawls_33kpa" and nat["recipe"]["field_capacity_deprecated"] is False
    assert nat["theta_field_capacity"] == pytest.approx(FC_EXPECT, abs=1e-15)
    assert nat["theta_saturation"] == pytest.approx(SAT, abs=1e-15) and nat["theta_1500"] == pytest.approx(WP, abs=1e-15)
    assert nat["saturated_conductivity_m_s"] == pytest.approx(KS_M_S, rel=1e-13)
    for fc, (S, Si, C_, OM, rho) in zip(nat["theta_field_capacity"], TEX):
        assert fc == pytest.approx(sr33(S, Si, C_, OM, rho)[0], abs=1e-15)
    assert max(nat["theta_field_capacity"]) < 0.11 and min(OLD_FC) > 0.39      # not the old (~39.8 vol%) 6 kPa value, and not relabelled


# ------------------------------------------------------------------------------------------------------------------------------ generated configs
@pytest.mark.parametrize("key,factor", [("k0p5", 0.5), ("k1", 1.0), ("k2", 2.0), ("k4", 4.0)])
def test_whitelisted_changes_and_exact_scaling(generated, key, factor):
    _, paths, old = generated
    new = load(paths, key)
    allowed_exact = {"name", "expected_sources.maple_syrup.package_sha256", "provenance.role", "layers.theta_field_capacity[0]", "layers.theta_field_capacity[1]", "layers.theta_field_capacity[2]"}
    for i in range(3):
        allowed_exact |= {f"provenance.layer_properties[{i}].theta_field_capacity", f"provenance.layer_properties[{i}].theta_field_capacity_previous_hybrid_6kpa"}
        if factor != 1.0:
            allowed_exact |= {f"provenance.layer_properties[{i}].Ks_mm_h", f"provenance.layer_properties[{i}].Ks_m_s"}
    if factor != 1.0:
        allowed_exact |= {"matrix.ks_top_m_s", "matrix.conductivity_m_s", "layers.interface_conductivity_m_s[0]", "layers.interface_conductivity_m_s[1]"}
    changed = {k for k in diff(old, new) if not k.startswith("provenance.field_capacity_update") and not k.startswith("provenance.conductivity_sensitivity")}
    assert changed == allowed_exact
    assert {k for k in diff(old, new) if k.startswith("provenance.")} - allowed_exact == {k for k in diff(old, new) if k.startswith(("provenance.field_capacity_update", "provenance.conductivity_sensitivity"))}
    # exact scaling of the ten conductivity entries (powers of two: exact in binary64), literal anchors
    for i in range(3):
        p = new["provenance"]["layer_properties"][i]
        assert p["Ks_m_s"] == KS_M_S[i] * factor and p["Ks_mm_h"] == KS_MM_H[i] * factor
        assert p["theta_field_capacity"] == pytest.approx(FC_EXPECT[i], abs=1e-15) and p["theta_field_capacity_previous_hybrid_6kpa"] == OLD_FC[i]
    assert new["matrix"]["ks_top_m_s"] == MATRIX_KS * factor and new["matrix"]["conductivity_m_s"] == MATRIX_K * factor
    assert new["layers"]["interface_conductivity_m_s"] == [v * factor for v in IFACE]
    assert new["layers"]["theta_field_capacity"] == [p["theta_field_capacity"] for p in new["provenance"]["layer_properties"]]
    assert new["matrix"]["conductivity_m_s"] / new["matrix"]["ks_top_m_s"] == old["matrix"]["conductivity_m_s"] / old["matrix"]["ks_top_m_s"]


@pytest.mark.parametrize("key", ["k0p5", "k1", "k2", "k4"])
def test_unchanged_physics_inputs_forcing_observations_and_matrix_only(generated, key):
    _, paths, old = generated
    new = load(paths, key)
    for k in ("site", "forcing", "swc", "initial", "physics"):
        assert new[k] == old[k], k
    assert new["layers"]["theta_saturation"] == SAT and new["layers"]["theta_residual"] == [0.01] * 3 and new["layers"]["edges_m"] == [0.0, 0.1, 0.2, 0.3]
    assert new["layers"]["theta_evaporation_stress"] == 0.20170581117963332 and new["physics"]["diffusion_coefficient"] == 0.0010239461376024856
    assert new["physics"]["evaporation_factor"] == 1.0 and new["physics"]["transpiration_factor"] == 0 and new["physics"]["bottom"] == "open"
    assert new["matrix"]["local_conductivity_factor"] == 0.31635738822500753 and new["matrix"]["suction_m"] == 0.166 and new["matrix"]["vegetation_cover_sum"] == 0.0
    assert new["forcing"]["n_days"] == 160 and new["forcing"]["expected"]["total_rain_mm"] == "25.088" and new["forcing"]["start_utc"] == "2024-01-18T07:00:00Z"
    assert not any("pref" in k.lower() and "provenance" not in k for k, _ in flat({kk: v for kk, v in new.items() if kk != "provenance"}))   # matrix only: no preferential parameter exists
    for i in range(3):
        p, o = new["provenance"]["layer_properties"][i], old["provenance"]["layer_properties"][i]
        for k in ("theta_saturation", "theta1500", "raw_air_entry_kpa", "effective_air_entry_kpa", "effective_density_g_cm3", "raw_density_g_cm3", "density_factor", "record"):
            assert p[k] == o[k], k
        assert p["effective_air_entry_kpa"] == 0


def test_provenance_records_new_physical_method_not_historical(generated):
    _, paths, _ = generated
    u = load(paths, "k2")["provenance"]["field_capacity_update"]
    assert "saxton_rawls_33kpa" in u["method"] and "NOT unchanged" in u["status"] and "REPLACED" in u["status"]
    assert u["provider_recipe"] == "saxton_rawls_2006_table1_eq1_18_explicit_density_v2_fc33_default" and u["air_entry_kpa_override"] == 0.0 and len(u["provider_input_sha256"]) == 64
    assert u["texture_source"]["sha256"] == G.FIELD_AUDIT_SHA256 and "tietjen_6kpa" not in json.dumps(load(paths, "k2")["provenance"]["field_capacity_update"]["provider_recipe"])
    assert "x2" in load(paths, "k2")["name"] and load(paths, "k2")["provenance"]["conductivity_sensitivity"]["relative_factor"] == 2.0


def test_controls_differ_only_in_backend_and_max_dt(generated):
    _, paths, _ = generated
    for key in G.FACTORS:
        base = load(paths, key, "numba")
        assert diff(base, load(paths, key, "numpy")) == {"controls.backend"}
        assert diff(base, load(paths, key, "numba_dt900")) == {"storm.max_dt_s"}
        assert load(paths, key, "numpy")["controls"]["backend"] == "numpy" and load(paths, key, "numba_dt900")["storm"]["max_dt_s"] == 900
    # same relative factor only across the four settings: every non-conductivity / non-name / non-provenance field is identical
    a, b = load(paths, "k1"), load(paths, "k4")
    assert {k for k in diff(a, b) if not k.startswith("provenance")} == {"name", "matrix.ks_top_m_s", "matrix.conductivity_m_s", "layers.interface_conductivity_m_s[0]", "layers.interface_conductivity_m_s[1]"}


# ------------------------------------------------------------------------------------------------------------------------------ real driver: validation, sources, inputs
@pytest.mark.parametrize("key", ["k0p5", "k1", "k2", "k4"])
@pytest.mark.parametrize("variant", ["numba", "numpy", "numba_dt900"])
def test_closed_driver_validation_and_new_source_guard(generated, key, variant):
    _, paths, _ = generated
    cfg, sha = C.load_config(paths[key][variant]["path"])
    assert sha == paths[key][variant]["sha256"]
    assert (cfg["layers"]["theta_field_capacity"] < cfg["layers"]["theta_saturation"]).all() and (cfg["layers"]["theta_field_capacity"] > cfg["layers"]["theta_residual"]).all()
    receipt = D.check_sources(cfg)      # the real imported packages must be the NEW 4e38 SYRUP + fbc MAPLE
    assert receipt["maple_syrup"]["package_sha256"] == G.NEW_SYRUP["package_sha256"] and receipt["maple"]["package_sha256"] == G.MAPLE["package_sha256"]


def test_old_source_pins_are_refused_not_rebound(generated):
    _, paths, old = generated
    with pytest.raises(D.ControlFailure, match="maple_syrup"):
        D.check_sources(C.validate_config(old))      # the ORIGINAL config's 37d89f23 pin must fail against the new package
    bad = load(paths, "k1")
    bad["expected_sources"]["maple_syrup"] = dict(G.OLD_SYRUP)
    with pytest.raises(D.ControlFailure):
        D.check_sources(C.validate_config(bad))
    bad = load(paths, "k1")
    bad["expected_sources"]["maple"]["package_sha256"] = "0" * 64
    with pytest.raises(D.ControlFailure):
        D.check_sources(C.validate_config(bad))


def test_same_observations_and_initial_water_no_double_water(generated):
    """Real prepared inputs (CSV parsing only, no integration): observations, forcing, schedule and initial water are bit-identical between the original and every new config."""
    _, paths, old = generated
    ref = I.prepare_inputs(C.validate_config(old))
    for key in G.FACTORS:
        cfg, _ = C.load_config(paths[key]["numba"]["path"])
        prep = I.prepare_inputs(cfg)
        same = lambda a, b: json.dumps(I.jsonable(a), sort_keys=True) == json.dumps(I.jsonable(b), sort_keys=True)
        assert same(prep["initial"], ref["initial"]) and same(prep["target_counts"], ref["target_counts"]) and same(prep["theta0_points"], ref["theta0_points"])
        assert prep["n_days"] == 160 and prep["n_wet_days"] == ref["n_wet_days"] and prep["rain"]["total_mm_decimal"] == ref["rain"]["total_mm_decimal"] == "25.088"
        assert np.array_equal(prep["pet_mm_day"], ref["pet_mm_day"])
        # the initial water is below the NEW field capacity in every layer (so no initial water is created or removed by the FC change)
        assert all(t < fc for t, fc in zip(prep["initial"]["layer_theta"], cfg["layers"]["theta_field_capacity"]))


def test_runner_configs_pin_new_sources_and_cpu_only(generated):
    out, paths, _ = generated
    for key in G.FACTORS:
        d = out / key
        man = json.loads((d / "frozen_helper_manifest.json").read_text())
        assert set(man) == set(G.HELPERS) and all(G.sha256_file(G.CANDIDATE / k) == v for k, v in man.items())
        for variant in G.VARIANTS:
            rc = json.loads(Path(paths[key][variant]["runner_config"]).read_text())
            assert [s["sha256"] for s in rc["sources"]] == [G.MAPLE["package_sha256"], G.NEW_SYRUP["package_sha256"]]
            assert rc["env"]["CUDA_VISIBLE_DEVICES"] == "" and rc["env"]["NUMBA_NUM_THREADS"] == "1" and "gpu_uuid" not in rc and rc["cwd"] == str(G.CANDIDATE)
            assert Path(rc["probe_file"]).is_file() and f"CASE='{variant}'" in Path(rc["probe_file"]).read_text()


def test_relative_out_is_resolved_absolute_and_stays_inside(tmp_path, monkeypatch):
    """A relative `out` must give absolute table/runner/probe paths under the resolved directory, independent of the later cwd, and write nothing outside it."""
    monkeypatch.chdir(tmp_path)
    paths = G.generate(Path("rel_study"))
    root = (tmp_path / "rel_study").resolve()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["rel_study"]
    for key in G.FACTORS:
        for variant in G.VARIANTS:
            e = paths[key][variant]
            for name in ("path", "runner_config", "probe"):
                q = Path(e[name])
                assert q.is_absolute() and q.is_file() and q.resolve().is_relative_to(root), (name, q)
            rc = json.loads(Path(e["runner_config"]).read_text())
            for name in ("output", "probe_file"):
                assert Path(rc[name]).is_absolute() and Path(rc[name]).resolve().is_relative_to(root), (name, rc[name])
            assert rc["env"]["NUMBA_CACHE_DIR"].startswith(str(root)) and not Path(rc["output"]).exists()
    u = json.loads(Path(paths["k2"]["numba"]["path"]).read_text())["provenance"]["field_capacity_update"]
    assert "unchanged_vs_original_and_provider" not in u and "before_conductivity_sensitivity_scaling" in json.dumps(u)
