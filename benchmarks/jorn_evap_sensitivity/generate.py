"""Generate the JORN 160-day SR33 1xK evaporation sensitivity configs for the UNCHANGED `benchmarks.subsurface_wet_validation` driver.

Base: the accepted SR33 1xK case (`jorn_sr33_160day_20261120/study/k1/case_numba.json`, SHA-256 pinned). Eight UNIQUE cases (the baseline is shared by both families):
  family E      : baseline (E 1), e0p8, e0p6, e0p4          -> physics.evaporation_factor 1, 0.8, 0.6, 0.4 at the baseline stress
  family stress : baseline, s1p5, s2, s2p5, s3              -> layers.theta_evaporation_stress = baseline stress x 1, 1.5, 2, 2.5, 3 at E 1
Only those two numeric locations (plus the case name, an added provenance record and the backend/step control of the variant) differ from the base. Nothing is refit, clamped or refused: a stress above
the saturation is a hypothetical normalisation scale of the quadratic reduction, never a storage capacity. Each case gets numba1800, numpy1800 and numba900 configs (24 runs), each with absolute launcher/probe paths,
a copy of the guarded bootstrap beside its runner configs, and a frozen-helper manifest of THIS candidate.
"""
from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path

# READ-ONLY use of the prior generator's constants/templates (its own CANDIDATE global is NOT used)
from benchmarks.jorn_sr33_sensitivity import generate as G

CANDIDATE = Path(__file__).resolve().parents[2]
TASKS = Path("/home/okin/SYRUP/agent_handoffs/tasks")
TASK = TASKS / "jorn_evap_sensitivity_20261120"
BASE_CONFIG = TASKS / "jorn_sr33_160day_20261120/study/k1/case_numba.json"
BASE_SHA256 = "f024054f5198ec8358c95f7a33683e6228084e6af246f9790cde586d437f3d2a"
BOOTSTRAP = TASK / "root_release_verified_bootstrap.py"
BASE_STRESS = 0.20170581117963332
SYRUP, MAPLE = G.NEW_SYRUP, G.MAPLE                                    # 4e38a472.../180 and fbc504bb.../412
VARIANTS = G.VARIANTS                                                 # numba (1800 s), numpy (1800 s), numba_dt900
CASES = {"baseline": (1.0, 1.0), "e0p8": (0.8, 1.0), "e0p6": (0.6, 1.0), "e0p4": (0.4, 1.0), "s1p5": (1.0, 1.5), "s2": (1.0, 2.0), "s2p5": (1.0, 2.5), "s3": (1.0, 3.0)}   # key: (E factor, stress multiplier)
FAMILIES = {"E": ["baseline", "e0p8", "e0p6", "e0p4"], "stress": ["baseline", "s1p5", "s2", "s2p5", "s3"]}
CHAIN = ("Tietjen et al. 2009 (doi 10.1029/2007WR006589) section 2.3: the water content wsc at which stomatal closure begins for combined ET, associated in Table 2 with -1500 kPa; the later downloaded EcoHyD WaterCell.cpp uses WP in the "
         "quadratic theta/WP reduction below the threshold (Tabernas input 0.18 is a CASE value). The SYRUP baseline 0.2017058 is jointly fitted with d to the 53 JORN/MOAB 2025 dry observations at E = 1 under the OLD field capacity "
         "(docs/subsurface_hydrology/joint_d_stress_calibration/report.md; site_properties_mixing/evaporation_explanation/explanation.md): not a site-measured wilting point and not derived by Saxton-Rawls.")


def sha256_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_base() -> dict:
    assert sha256_file(BASE_CONFIG) == BASE_SHA256, "accepted SR33 1xK base case changed"
    base = json.loads(BASE_CONFIG.read_text())
    assert base["layers"]["theta_evaporation_stress"] == BASE_STRESS and base["physics"]["evaporation_factor"] == 1.0
    return base


def build_case(base: dict, key: str, variant: str) -> dict:
    factor, mult = CASES[key]
    cfg = copy.deepcopy(base)
    stress = BASE_STRESS * mult
    cfg["physics"]["evaporation_factor"] = factor
    cfg["layers"]["theta_evaporation_stress"] = stress
    cfg["provenance"]["evaporation_sensitivity"] = {
        "case_key": key, "families": [f for f, ks in FAMILIES.items() if key in ks], "evaporation_factor": factor, "theta_evaporation_stress": stress, "stress_multiplier": mult, "baseline_stress": BASE_STRESS,
        "changed_controls": [c for c, v in (("physics.evaporation_factor", factor != 1.0), ("layers.theta_evaporation_stress", mult != 1.0)) if v] or ["none (shared baseline)"],
        "effective_quadratic_coefficient_E_over_r2": factor / (mult * mult),
        "branch_note": "top moisture at the E stage is <= field capacity (9.75 vol. %) < every stress here, so every case stays on the quadratic branch (theta/stress)^2: stress multiplier r is equivalent to E/r^2",
        "stress_above_saturation": bool(stress > base["layers"]["theta_saturation"][0]),
        "stress_note": "a stress above saturation is a hypothetical normalisation scale of the reduction; it does not change storage capacity, SAT, WP or FC and is not a measured wilting point; never refit",
        "base_config": {"path": str(BASE_CONFIG), "sha256": BASE_SHA256}, "calibration_chain": CHAIN,
        "not_done": "no refit, no transpiration, no K/default adoption; d, K, retention, initial water and forcing identical to the base"}
    cfg["provenance"]["role"] = base["provenance"]["role"] + f"; evaporation sensitivity case {key} (E {factor:g}, stress x{mult:g}), not a calibration"
    cfg["name"] = base["name"] + f" evaporation sensitivity {key}"
    cfg["controls"] = {"backend": VARIANTS[variant]["backend"]}
    cfg["storm"] = {"max_dt_s": VARIANTS[variant]["max_dt_s"]}
    return cfg


def runner_config(d: Path, variant: str) -> dict:
    rc = G.runner_config(d, variant)               # template only; every path re-pointed to THIS candidate below
    old = str(G.CANDIDATE)
    new = str(CANDIDATE)
    text = json.dumps(rc).replace(old, new).replace("MPLCONFIGDIR\": \"/tmp/syrup-jorn-sr33-mpl", "MPLCONFIGDIR\": \"/tmp/syrup-jorn-evap-mpl")
    rc = json.loads(text)
    assert not any(old in str(v) for v in rc.values()) and rc["cwd"] == new
    assert [s["sha256"] for s in rc["sources"]] == [MAPLE["package_sha256"], SYRUP["package_sha256"]] and rc["sources"][1]["root"] == str(CANDIDATE / "src")
    return rc


def generate(out) -> dict:
    out = Path(out).resolve()                      # absolute BEFORE any path is generated
    base = load_base()
    paths = {"families": FAMILIES, "base": {"path": str(BASE_CONFIG), "sha256": BASE_SHA256}, "candidate": str(CANDIDATE), "cases": {}}
    manifest = {rel: sha256_file(CANDIDATE / rel) for rel in G.HELPERS}
    for key, (factor, mult) in CASES.items():
        d = out / key
        d.mkdir(parents=True, exist_ok=True)
        for variant in VARIANTS:
            p = d / f"case_{variant}.json"
            p.write_text(json.dumps(build_case(base, key, variant), indent=2) + "\n")
            (d / f"root_execute_{variant}.py").write_text(G.PROBE.format(candidate=str(CANDIDATE), case=variant))
            rcp = d / f"actual_{variant}_config.json"
            rcp.write_text(json.dumps(runner_config(d, variant), indent=2) + "\n")
            paths["cases"].setdefault(key, {"evaporation_factor": factor, "stress_multiplier": mult, "theta_evaporation_stress": BASE_STRESS * mult, "variants": {}})["variants"][variant] = {
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
