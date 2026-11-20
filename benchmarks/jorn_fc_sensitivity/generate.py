"""Generate the JORN 160-day uniform field-capacity sensitivity configs for the UNCHANGED `benchmarks.subsurface_wet_validation` driver.

Base: the accepted SR33 1xK case (`jorn_sr33_160day_20261120/study/k1/case_numba.json`, SHA-256 pinned). Fourteen UNIQUE crossed cases:
  field capacity (ALL THREE layers equal) 9, 10, 11, 12, 13, 14, 15 vol. %  x  evaporation stress 0.2017 and 1.5 x 0.2017 = 0.30255  (E = 1).
Only `layers.theta_field_capacity` (the explicit user override [fc, fc, fc]) and `layers.theta_evaporation_stress` (the EXACT rounded 0.2017 or 0.30255, not the earlier fitted 0.20170581117963332) differ from the base,
plus the case name, an added provenance record and the backend/step control of the variant. These are explicit field-capacity overrides on the SR33-derived 1x conductivities: no K, SAT, wilting point, residual or
Green-Ampt value is recomputed, no self-consistent retention curve or new soil texture is implied, and nothing is refit. Each case gets numba1800, numpy1800 and numba900 configs (42 runs) with absolute
launcher/probe paths, a byte-identical copy of the guarded bootstrap beside its runner configs and a frozen-helper manifest of THIS candidate.
"""
from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path

# READ-ONLY use of the earlier generator's constants/templates (its own CANDIDATE global is NOT used)
from benchmarks.jorn_sr33_sensitivity import generate as G

CANDIDATE = Path(__file__).resolve().parents[2]
TASKS = Path("/home/okin/SYRUP/agent_handoffs/tasks")
TASK = TASKS / "jorn_fc_sensitivity_20261120"
BASE_CONFIG = TASKS / "jorn_sr33_160day_20261120/study/k1/case_numba.json"
BASE_SHA256 = "f024054f5198ec8358c95f7a33683e6228084e6af246f9790cde586d437f3d2a"
BOOTSTRAP = TASK / "root_release_verified_bootstrap.py"
FITTED_STRESS = 0.20170581117963332            # the earlier jointly fitted value: historical, NOT used here
STRESS_BASE = 0.2017                           # the exact rounded value requested for this study
FC_PERCENT = (9, 10, 11, 12, 13, 14, 15)
STRESS = {"s1": 1.0, "s1p5": 1.5}              # multiplier of STRESS_BASE
SYRUP, MAPLE = G.NEW_SYRUP, G.MAPLE            # 4e38a472.../180 and fbc504bb.../412
VARIANTS = G.VARIANTS                          # numba (1800 s), numpy (1800 s), numba_dt900
CASES = {f"fc{p:02d}_{s}": (p / 100.0, STRESS_BASE * m, m) for s, m in STRESS.items() for p in FC_PERCENT}      # key: (field capacity, stress, multiplier)
FAMILIES = {"stress1": [f"fc{p:02d}_s1" for p in FC_PERCENT], "stress1p5": [f"fc{p:02d}_s1p5" for p in FC_PERCENT], **{f"FC{p:02d}": [f"fc{p:02d}_s1", f"fc{p:02d}_s1p5"] for p in FC_PERCENT}}
CHAIN = ("Tietjen et al. 2009 (doi 10.1029/2007WR006589) section 2.3: threshold for combined ET (Table 2: -1500 kPa); EcoHyD WaterCell.cpp uses WP in the quadratic form; the earlier SYRUP value 0.2017058 was fitted jointly with d "
         "to 53 JORN/MOAB 2025 dry observations at E = 1, T = 0 and the OLD field capacity (docs/subsurface_hydrology/joint_d_stress_calibration/report.md). The values here (0.2017, 0.30255) are rounded/scaled sensitivity "
         "settings, never newly fitted.")


def sha256_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_base() -> dict:
    assert sha256_file(BASE_CONFIG) == BASE_SHA256, "accepted SR33 1xK base case changed"
    base = json.loads(BASE_CONFIG.read_text())
    assert base["layers"]["theta_evaporation_stress"] == FITTED_STRESS and base["physics"]["evaporation_factor"] == 1.0
    return base


def build_case(base: dict, key: str, variant: str) -> dict:
    fc, stress, mult = CASES[key]
    cfg = copy.deepcopy(base)
    cfg["layers"]["theta_field_capacity"] = [fc, fc, fc]
    cfg["layers"]["theta_evaporation_stress"] = stress
    cfg["provenance"]["field_capacity_sensitivity"] = {
        "case_key": key, "families": [f for f, ks in FAMILIES.items() if key in ks], "field_capacity": fc, "field_capacity_percent": FC_PERCENT[int(key[2:4]) - 9], "theta_evaporation_stress": stress, "stress_multiplier_of_0p2017": mult,
        "changed_controls": ["layers.theta_field_capacity", "layers.theta_evaporation_stress"], "evaporation_factor": 1.0,
        "override_note": "field capacity is an explicit user override, equal in ALL THREE layers; Ks, matrix K, interface K, saturation, wilting point, residual, Green-Ampt geometry and d are the SR33-derived 1x values and were NOT recomputed; "
                         "this is not a self-consistent Saxton-Rawls retention curve or a new soil texture",
        "historical_layer_values": "provenance.layer_properties[*].theta_field_capacity are the original depth-specific SR33 values (9.746 / 9.940 / 10.914 vol. %), kept as historical context; the run uses layers.theta_field_capacity",
        "stress_note": "0.2017 is the exact rounded value requested here, not the earlier fitted 0.20170581117963332 (historical, joint d/stress fit at E = 1, T = 0, old field capacity); 0.30255 = 1.5 x 0.2017; never newly fitted",
        "fc_vs_stress": "every stress here is above every field capacity, so the quadratic evaporation branch always applies at the evaporation stage (top water <= FC)",
        "base_config": {"path": str(BASE_CONFIG), "sha256": BASE_SHA256}, "calibration_chain": CHAIN,
        "not_done": "no refit, no transpiration, no K/default adoption; weather, rain, observations, initial water, clocks, E = 1, open bottom, matrix-only identical to the base"}
    cfg["provenance"]["role"] = base["provenance"]["role"] + f"; uniform field-capacity sensitivity case {key} (FC {fc * 100:g} vol. % in all layers, stress {stress:g}), not a calibration"
    cfg["name"] = base["name"] + f" uniform FC sensitivity {key}"
    cfg["controls"] = {"backend": VARIANTS[variant]["backend"]}
    cfg["storm"] = {"max_dt_s": VARIANTS[variant]["max_dt_s"]}
    return cfg


def runner_config(d: Path, variant: str) -> dict:
    rc = G.runner_config(d, variant)               # template only; every path re-pointed to THIS candidate below
    old, new = str(G.CANDIDATE), str(CANDIDATE)
    rc = json.loads(json.dumps(rc).replace(old, new).replace("/tmp/syrup-jorn-sr33-mpl", "/tmp/syrup-jorn-fc-mpl"))
    assert not any(old in str(v) for v in rc.values()) and rc["cwd"] == new
    assert [s["sha256"] for s in rc["sources"]] == [MAPLE["package_sha256"], SYRUP["package_sha256"]] and rc["sources"][1]["root"] == str(CANDIDATE / "src")
    return rc


def generate(out) -> dict:
    out = Path(out).resolve()                      # absolute BEFORE any path is generated
    base = load_base()
    paths = {"families": FAMILIES, "base": {"path": str(BASE_CONFIG), "sha256": BASE_SHA256}, "candidate": str(CANDIDATE), "cases": {}}
    manifest = {rel: sha256_file(CANDIDATE / rel) for rel in G.HELPERS}
    for key, (fc, stress, mult) in CASES.items():
        d = out / key
        d.mkdir(parents=True, exist_ok=True)
        for variant in VARIANTS:
            p = d / f"case_{variant}.json"
            p.write_text(json.dumps(build_case(base, key, variant), indent=2) + "\n")
            (d / f"root_execute_{variant}.py").write_text(G.PROBE.format(candidate=str(CANDIDATE), case=variant))
            rcp = d / f"actual_{variant}_config.json"
            rcp.write_text(json.dumps(runner_config(d, variant), indent=2) + "\n")
            paths["cases"].setdefault(key, {"field_capacity": fc, "stress_multiplier": mult, "theta_evaporation_stress": stress, "variants": {}})["variants"][variant] = {
                "path": str(p), "sha256": sha256_file(p), "runner_config": str(rcp), "probe": str(d / f"root_execute_{variant}.py"), "output_dir": str(d / f"actual_{variant}_runs"), "result_dir": str(d / f"actual_{variant}")}
        (d / "frozen_helper_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        shutil.copyfile(BOOTSTRAP, d / "root_release_verified_bootstrap.py")      # the launcher expects the bootstrap beside every per-case runner config
        assert sha256_file(d / "root_release_verified_bootstrap.py") == sha256_file(BOOTSTRAP)
    return paths


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    print(json.dumps(generate(Path(ap.parse_args().out)), indent=2))
