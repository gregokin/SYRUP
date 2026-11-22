"""Generate the six JORN/MOAB dry-window fixed-stress cases (3 windows x stress 0.2017 and 0.30255) for the UNCHANGED `benchmarks.subsurface_validation.drydown_driver`.

Templates: the historical harmonic joint-calibration configs (read-only, SHA pinned) of the three windows JORN4_A (2025-04-08..18), JORN4_B (04-26..30) and MOAB2_C (05-22..28). Relative to the template each case changes ONLY:
the field capacity (JORN: the user preset 0.10 in all layers, loaded from cases/subsurface_validation/jorn_defaults.json; MOAB: its OWN SR33 field capacity from the current native provider), the evaporation stress
(0.2017 or the preset's 0.30255), the scenario's diffusion coefficient (historical 0.05 -> the selected 0.0010239461376024856) and name, the SYRUP package pin (37d89f23 -> 4e38a472), the case name and the provenance.
Saturation, residual, interface conductivities, windows, weather, initial inventory, QA observations, MAPLE pin and all controls are the template's. No fitting, no rain, no transpiration.
Each case: one config (backends numba + numpy and the midpoint public-state reconstruction are inside it), one runner config, one probe, a copy of the guarded bootstrap and a frozen-helper manifest, all with absolute
paths of THIS candidate; `config_paths.json` is the single explicit run index.
"""
from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path

from benchmarks.jorn_dry_stress import preset as PS
from benchmarks.jorn_dry_stress import properties as PR

# READ-ONLY use of the earlier generator's constants (its CANDIDATE global is NOT used)
from benchmarks.jorn_sr33_sensitivity import generate as G

CANDIDATE = Path(__file__).resolve().parents[2]
TASKS = Path("/home/okin/SYRUP/agent_handoffs/tasks")
TASK = TASKS / "jorn_dry_stress_20261122"
BOOTSTRAP = TASK / "root_release_verified_bootstrap.py"
HIST = CANDIDATE / "docs/subsurface_hydrology/joint_d_stress_calibration/data/actual_cpu/configs"
TEMPLATES = {"JORN4_A": ("harmonic_JORN4_A.json", "d468888ca53566a3e0b4fc7cb6a948a1a5335983d871f7129266ef887451f301"), "JORN4_B": ("harmonic_JORN4_B.json", "8addfb6a20bff22e33656176355e52ff226c6244705b082415942dbfa1071c94"),
             "MOAB2_C": ("harmonic_MOAB2_C.json", "50d678a0695a04c85bc329231061916048bfd720e5b81b4c25a0f67f51dcd370")}
WINDOW_TEXT = {"JORN4_A": "JORN plot 004, 2025-04-08 to 2025-04-18", "JORN4_B": "JORN plot 004, 2025-04-26 to 2025-04-30", "MOAB2_C": "MOAB plot 002, 2025-05-22 to 2025-05-28"}
STRESS_COMPARISON = 0.2017          # the exact rounded earlier value; NOT the fitted 0.20170581117963332
SYRUP, MAPLE = G.NEW_SYRUP, G.MAPLE
OLD_SYRUP_SHA = "37d89f23dbf60626c85f85ea5dd6d1734d3bc7a4aa61d8f2fdfc29f237bcfbd0"
KEYS = [f"{w}_{s}" for w in TEMPLATES for s in ("s1", "s1p5")]
HELPERS = ("benchmarks/subsurface_validation/__init__.py", "benchmarks/subsurface_validation/drydown_driver.py", "benchmarks/subsurface_validation/neon_inputs.py", "benchmarks/subsurface_validation/paths.py",
           "benchmarks/jorn_dry_stress/__init__.py", "benchmarks/jorn_dry_stress/preset.py", "benchmarks/jorn_dry_stress/properties.py", "benchmarks/jorn_dry_stress/run.py", "cases/subsurface_validation/jorn_defaults.json")
PROBE = """from pathlib import Path
import json,hashlib,time,sys,platform,resource
P=Path(__file__).resolve().parent
A=Path({candidate!r})
manifest=json.loads((P/'frozen_helper_manifest.json').read_text())
def check():
 for rel,want in manifest.items():assert hashlib.sha256((A/rel).read_bytes()).hexdigest()==want,rel
check()
t=time.perf_counter()
from benchmarks.jorn_dry_stress import run
output=P/'actual'
rc=run.main(['--config',str(P/'case.json'),'--out',str(output),'--execute'])
assert rc==0
check()
record={{'driver_call_wall_s':time.perf_counter()-t,'scope':'includes driver import/case validation/preparation/column runs (numba+numpy+restart control)/scoring/output; excludes outer interpreter/bootstrap/source checks','python':sys.version,'platform':platform.platform(),'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'model_scope':'one 1 m2 three-layer column dry window, CPU, not a GPU/scaling benchmark','source_helper_unchanged':True}}
(P/'runtime.json').write_text(json.dumps(record,indent=2)+chr(10))
"""


def sha256_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_template(window: str) -> dict:
    name, want = TEMPLATES[window]
    assert sha256_file(HIST / name) == want, f"historical template {name} changed"
    return json.loads((HIST / name).read_text())


def window_properties(window: str, preset: dict, template: dict, moab: dict | None) -> tuple:
    """(field capacity list, provenance text) for the window; checks the unchanged properties against the template."""
    if window.startswith("JORN"):
        return list(preset["theta_field_capacity"]), "JORN: user preset theta_field_capacity_all_layers (cases/subsurface_validation/jorn_defaults.json), scalar applied to all three layers"
    lay = template["layers"]
    tp = template["provenance"]["layers"]
    for i in range(3):                       # the provider reproduces the template's SAT / Ks / wilting point (unchanged properties) exactly enough to bind the same texture inputs
        assert abs(moab["theta_saturation"][i] - lay["theta_saturation"][i]) <= 1e-12 * lay["theta_saturation"][i]
        assert abs(moab["saturated_conductivity_m_s"][i] - tp[i]["Ks_m_s"]) <= 1e-12 * tp[i]["Ks_m_s"]
        assert abs(moab["theta_1500"][i] - tp[i]["theta1500"]) <= 1e-12 * tp[i]["theta1500"]
    return [float(x) for x in moab["theta_field_capacity"]], "MOAB: its OWN SR33 field capacity (33 kPa, native provider default) from the original MOAB texture/OM/measured density; not the old 6 kPa template value and not JORN's 10 vol. %"


def build_case(window: str, skey: str, preset: dict, moab: dict | None) -> dict:
    tmpl = load_template(window)
    cfg = copy.deepcopy(tmpl)
    fc, fc_text = window_properties(window, preset, tmpl, moab)
    stress = STRESS_COMPARISON if skey == "s1" else preset["theta_evaporation_stress"]
    d = preset["diffusion_coefficient"]
    old_fc, old_stress, old_d = tmpl["layers"]["theta_field_capacity"], tmpl["layers"]["theta_evaporation_stress"], tmpl["scenarios"][0]["diffusion_coefficient"]
    cfg["layers"]["theta_field_capacity"] = fc
    cfg["layers"]["theta_evaporation_stress"] = stress
    cfg["scenarios"][0]["diffusion_coefficient"] = d
    cfg["scenarios"][0]["name"] = f"fixed_d_stress_{skey}_E1_open"
    cfg["pins"]["expected_sources"]["maple_syrup"] = dict(SYRUP)
    key = f"{window}_{skey}"
    cfg["name"] = f"{window} dry-window fixed-stress comparison {skey} (stress {stress:g}, E=1, d={d:g})"
    cfg["provenance"]["dry_stress_comparison"] = {
        "case_key": key, "window": window, "window_text": WINDOW_TEXT[window], "stress": stress, "stress_label": "0.2017 (rounded earlier value)" if skey == "s1" else "0.30255 = 1.5 x 0.2017 (user-selected JORN setting)",
        "evaporation_factor": 1.0, "diffusion_coefficient": d, "field_capacity": fc, "field_capacity_source": fc_text, "transpiration": 0, "rain": 0,
        "template": {"file": f"docs/subsurface_hydrology/joint_d_stress_calibration/data/actual_cpu/configs/{TEMPLATES[window][0]}", "sha256": TEMPLATES[window][1]},
        "deltas_from_template": {"theta_field_capacity": {"old": old_fc, "new": fc}, "theta_evaporation_stress": {"old": old_stress, "new": stress}, "diffusion_coefficient": {"old": old_d, "new": d},
                                 "syrup_package_pin": {"old": OLD_SYRUP_SHA, "new": SYRUP["package_sha256"]}, "scenario_name": cfg["scenarios"][0]["name"]},
        "unchanged": "saturation, residual, interface conductivities, window, weather, initial inventory, QA observations, MAPLE pin, controls (backends numba+numpy, the template's midpoint public-state reconstruction day) and the 86400 s daily step",
        "preset": {"path": "cases/subsurface_validation/jorn_defaults.json", "sha256": preset["preset_sha256"], "used_for": "JORN field capacity, d, E and the selected stress" if window.startswith("JORN") else "d and E only (MOAB keeps its own properties)"},
        "d_provenance": "0.0010239461376024856: earlier joint fit of d and stress at E = 1, T = 0, old field capacity to 53 JORN/MOAB 2025 dry observations; retained, not re-fitted; these windows were part of that fit (not held-out validation)",
        "stress_provenance": "0.2017 and 0.30255 are fixed, rounded/user-selected values, never fitted here",
        "role": "fixed-stress dry-window comparison: no refit, no added transpiration, no new physics",
        "historical_note": "template provenance keys (historical_*, joint_calibration, layers, interface) retain their historical values (stress 0.18, d 0.05, old 6 kPa field capacity, 37d89f23 pin) as read-only context"}
    return cfg


def runner_config(d: Path, candidate: Path = CANDIDATE) -> dict:
    m = G.MAPLE_MATERIALIZED
    prefix = [str(m / "editable"), str(m / "source/src"), str(candidate / "src"), str(candidate)]
    env = {"CUDA_VISIBLE_DEVICES": "", "NUMBA_NUM_THREADS": "1", "NUMBA_CACHE_DIR": str(d / "numba_cache"), "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1", "MPLCONFIGDIR": "/tmp/syrup-jorn-dry-mpl",
           "MAPLE_SYRUP_EXPECTED_MAPLE_ROOT": str(m / "source"), "E3_PINNED_MAPLE_SRC": str(m / "source/src"), "SYRUP_PHASE3_TASKS_ROOT": str(TASKS), "SYRUP_PHASE3_TASK_DIR": str(TASKS / "subsurface_observed_drydown_20261114"),
           "SYRUP_PHASE3_RAW_ROOT": str(TASKS / "jorn_three_layer_20261102"), "SYRUP_PHASE3_DEPTH_TABLE": str(TASKS / "nevada_zero_flux_20261101/neon_metadata/swc_depthsV3"),
           "SYRUP_PHASE3_AIR_JORN": str(TASKS / "subsurface_observed_drydown_20261114/downloads_air_JORN"), "SYRUP_PHASE3_AIR_MOAB": str(TASKS / "subsurface_observed_drydown_20261114/downloads_air_MOAB"),
           "SYRUP_PHASE3_MEGAPIT_MOAB": str(TASKS / "subsurface_observed_drydown_20261114/downloads_megapit_MOAB"), "SYRUP_PHASE3_INVENTORY": str(candidate / "docs/subsurface_hydrology/phase3_observation_inventory.json")}
    return {"python": "/home/okin/MAPLE/.venv/bin/python", "output": str(d / "actual_runs"), "cwd": str(candidate), "prefix": prefix,
            "required_files": [str(candidate / "src/maple_syrup/subsurface.py"), str(m / "source/src/maple/__init__.py")],
            "sources": [{"module": "maple", "root": str(m / "source/src"), "sha256": MAPLE["package_sha256"], "files": MAPLE["files"]}, {"module": "maple_syrup", "root": str(candidate / "src"), "sha256": SYRUP["package_sha256"], "files": SYRUP["files"]}],
            "env": env, "probe_file": str(d / "root_execute.py")}


def generate(out) -> dict:
    out = Path(out).resolve()                      # absolute BEFORE any path is generated
    preset = PS.load_preset()
    moab = PR.moab_properties()
    manifest = {rel: sha256_file(CANDIDATE / rel) for rel in HELPERS}
    index = {"candidate": str(CANDIDATE), "preset": {"path": preset["path"], "sha256": preset["preset_sha256"]}, "moab_properties": {"field_capacity": moab["theta_field_capacity"], "recipe": moab["recipe"], "provider_input_sha256": moab["provider_input_sha256"]},
             "windows": WINDOW_TEXT, "stress": {"s1": STRESS_COMPARISON, "s1p5": preset["theta_evaporation_stress"]}, "cases": {}}
    for key in KEYS:
        window, skey = key.rsplit("_", 1)[0], key.rsplit("_", 1)[1]
        d = out / key
        d.mkdir(parents=True, exist_ok=True)
        cfg = build_case(window, skey, preset, moab)
        p = d / "case.json"
        p.write_text(json.dumps(cfg, indent=2) + "\n")
        (d / "root_execute.py").write_text(PROBE.format(candidate=str(CANDIDATE)))
        rcp = d / "actual_config.json"
        rcp.write_text(json.dumps(runner_config(d), indent=2) + "\n")
        (d / "frozen_helper_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        shutil.copyfile(BOOTSTRAP, d / "root_release_verified_bootstrap.py")      # the launcher expects the bootstrap beside every per-case runner config
        assert sha256_file(d / "root_release_verified_bootstrap.py") == sha256_file(BOOTSTRAP)
        index["cases"][key] = {"window": window, "site": window[:4], "stress_label": skey, "theta_evaporation_stress": cfg["layers"]["theta_evaporation_stress"], "theta_field_capacity": cfg["layers"]["theta_field_capacity"],
                               "diffusion_coefficient": cfg["scenarios"][0]["diffusion_coefficient"], "evaporation_factor": 1.0, "path": str(p), "sha256": sha256_file(p), "runner_config": str(rcp), "probe": str(d / "root_execute.py"),
                               "output_dir": str(d / "actual_runs"), "result_dir": str(d / "actual")}
    return index


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    print(json.dumps(generate(Path(ap.parse_args().out)), indent=2))
