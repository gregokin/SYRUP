"""Generate the JORN 160-day SR33 sensitivity configs (relative all-K/Ks 0.5, 1, 2, 4) for the UNCHANGED `benchmarks.subsurface_wet_validation` driver.

Only these items differ from the original `jorn_160_day_run_20261119/case_numba.json` (exactly pinned by SHA-256):
  * layers.theta_field_capacity and provenance.layer_properties[i].theta_field_capacity: CURRENT provider `estimate_soil_texture_properties` DEFAULT (`saxton_rawls_33kpa`),
    with the explicit study air-entry override 0, the approved effective densities and the primary texture/OM of the field audit; the previous hybrid value is kept as a provenance record;
  * the ten conductivity entries (3 provenance Ks_mm_h, 3 provenance Ks_m_s, matrix.ks_top_m_s, matrix.conductivity_m_s, 2 interface) scaled once by the exact power-of-two factor;
  * expected_sources (new SYRUP package, same MAPLE), name, and added provenance records.
Everything else (SAT, residual, stress, d, E, suction, local factor, forcing, observations, initial water, layers, clocks, matrix-only, no growth/T) is carried over unchanged.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np

TASKS = Path("/home/okin/SYRUP/agent_handoffs/tasks")
OLD_CONFIG = TASKS / "jorn_160_day_run_20261119" / "case_numba.json"
OLD_CONFIG_SHA256 = "e7513f545b074da5fc5627e96b79f62ed787863fb32c9df545a9422547e521ee"
FIELD_AUDIT = TASKS / "jorn_field_capacity_audit_20261120" / "root_audit.json"
FIELD_AUDIT_SHA256 = "96528a5e507939565245a74d9d531c183236001659a04af85be479f15a3d7a70"
NEW_SYRUP = {"package_sha256": "4e38a472e268a56491b788d5b1271cc3c3da23bb1cc3221a8c59d90f3cc864d2", "files": 180}
OLD_SYRUP = {"package_sha256": "37d89f23dbf60626c85f85ea5dd6d1734d3bc7a4aa61d8f2fdfc29f237bcfbd0", "files": 180}
MAPLE = {"package_sha256": "fbc504bbcd1ec0c02e6a96089134a06187532455ef7ba5e6f146801c8313c02e", "files": 412}
FACTORS = {"k0p5": 0.5, "k1": 1.0, "k2": 2.0, "k4": 4.0}
VARIANTS = {"numba": {"backend": "numba", "max_dt_s": 1800}, "numpy": {"backend": "numpy", "max_dt_s": 1800}, "numba_dt900": {"backend": "numba", "max_dt_s": 900}}
PROVIDER_TEXT = ("maple_syrup.soil_texture_properties.estimate_soil_texture_properties, DEFAULT field_capacity_method='saxton_rawls_33kpa' "
                 "(recipe saxton_rawls_2006_table1_eq1_18_explicit_density_v2_fc33_default; theta_field_capacity = final density-adjusted theta_33; Saxton & Rawls 2006 doi 10.2136/sssaj2005.0117)")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_old() -> dict:
    assert sha256_file(OLD_CONFIG) == OLD_CONFIG_SHA256, "original JORN160 config changed"
    return json.loads(OLD_CONFIG.read_text())


def audit_inputs(old: dict) -> dict:
    """Primary texture (fractions), OM (%) and the approved effective densities (kg/m3) for the three layers."""
    assert sha256_file(FIELD_AUDIT) == FIELD_AUDIT_SHA256, "accepted field audit changed"
    layers = json.loads(FIELD_AUDIT.read_text())["layers"]
    assert len(layers) == 3
    tex = [l["texture_percent"] for l in layers]
    rho = [p["effective_density_g_cm3"] * 1000.0 for p in old["provenance"]["layer_properties"]]
    for l, p in zip(layers, old["provenance"]["layer_properties"]):   # the audit's density_used and the old config's approved effective density must agree
        assert abs(float(l["calculations"]["density_used"]) - p["effective_density_g_cm3"]) <= 1e-14 * p["effective_density_g_cm3"]
    return {"sand": [float(t["sandTotal"]) / 100.0 for t in tex], "silt": [float(t["siltTotal"]) / 100.0 for t in tex], "clay": [float(t["clayTotal"]) / 100.0 for t in tex],
            "om_percent": [float(l["OM_percent"]) for l in layers], "bulk_density_kg_m3": rho}


def native_properties(inp: dict) -> dict:
    """Run the CURRENT provider (default SR33 method) with the study's explicit air-entry override 0. NumPy reference backend; returns plain lists."""
    from maple_syrup import soil_texture_properties as st
    r = st.estimate_soil_texture_properties(sand_fraction=np.array(inp["sand"]), silt_fraction=np.array(inp["silt"]), clay_fraction=np.array(inp["clay"]), organic_matter_percent=np.array(inp["om_percent"]),
                                            bulk_density_kg_m3=np.array(inp["bulk_density_kg_m3"]), n_layers=3, n_cells=1, air_entry_kpa_override=np.zeros(3), diagnostics=True, backend="numpy",
                                            memory_budget_bytes=10**8)
    return {"theta_field_capacity": r.array("theta_field_capacity").tolist(), "theta_saturation": r.array("theta_saturation").tolist(), "theta_1500": r.array("theta_1500").tolist(),
            "saturated_conductivity_m_s": r.array("saturated_conductivity_m_s").tolist(), "provider_input_sha256": r.sha256, "recipe": st.recipe_record(), "source_digests": st.source_digests()}


def build_case(old: dict, native: dict, inp: dict, factor: float, key: str, variant: str) -> dict:
    cfg = copy.deepcopy(old)
    prov, lay, mat = cfg["provenance"], cfg["layers"], cfg["matrix"]
    fc = native["theta_field_capacity"]
    # the unchanged properties must be the provider's own (Ks, SAT, WP): refuse silently different inputs
    for i, p in enumerate(prov["layer_properties"]):
        assert np.isclose(native["saturated_conductivity_m_s"][i], p["Ks_m_s"], rtol=1e-12, atol=0.0)
        assert np.isclose(native["theta_saturation"][i], p["theta_saturation"], rtol=1e-12, atol=0.0)
        assert np.isclose(native["theta_1500"][i], p["theta1500"], rtol=1e-12, atol=0.0)
        p["theta_field_capacity_previous_hybrid_6kpa"] = p["theta_field_capacity"]
        p["theta_field_capacity"] = fc[i]
        p["Ks_mm_h"] = p["Ks_mm_h"] * factor
        p["Ks_m_s"] = p["Ks_m_s"] * factor
    lay["theta_field_capacity"] = list(fc)
    mat["ks_top_m_s"] = mat["ks_top_m_s"] * factor
    mat["conductivity_m_s"] = mat["conductivity_m_s"] * factor
    lay["interface_conductivity_m_s"] = [v * factor for v in lay["interface_conductivity_m_s"]]
    prov["field_capacity_update"] = {
        "status": "NEW physical field-capacity method; the previous hybrid Tietjen-6kPa/Saxton-Rawls wet-branch FC (~39.8 vol%) is REPLACED, retention is NOT unchanged",
        "method": "saxton_rawls_33kpa (current package default)", "provider": PROVIDER_TEXT, "provider_recipe": native["recipe"]["recipe"], "provider_input_sha256": native["provider_input_sha256"],
        "provider_source_digests": native["source_digests"], "air_entry_kpa_override": 0.0,
        "air_entry_note": "study-explicit air-entry override 0 for all layers (as in the original study); approved effective densities (not a new floor implementation); primary texture and OM unchanged",
        "primary_texture_fractions_sand_silt_clay": [inp["sand"], inp["silt"], inp["clay"]], "organic_matter_percent": inp["om_percent"], "bulk_density_kg_m3_effective": inp["bulk_density_kg_m3"],
        "texture_source": {"path": str(FIELD_AUDIT), "sha256": FIELD_AUDIT_SHA256}, "unchanged_vs_original_and_provider_before_conductivity_sensitivity_scaling": ["Ks", "theta_saturation", "theta_1500 (wilting)"],
        "unchanged_note": "Ks equals the original/provider value ONLY before the conductivity-sensitivity scaling; the scaled Ks entries are listed in provenance.conductivity_sensitivity",
        "not_refit": "d, stress, residual, suction, local factor, E, initial water unchanged and conditional on the OLD field capacity; no calibration claim"}
    prov["conductivity_sensitivity"] = {"relative_factor": factor, "base_config": {"path": str(OLD_CONFIG), "sha256": OLD_CONFIG_SHA256},
                                        "scaled_entries": ["provenance.layer_properties[0..2].Ks_mm_h", "provenance.layer_properties[0..2].Ks_m_s", "matrix.ks_top_m_s", "matrix.conductivity_m_s", "layers.interface_conductivity_m_s[0..1]"],
                                        "rule": "each entry = original x factor, once (not compounded); local factor and Tietjen d unchanged"}
    prov["role"] = old["provenance"]["role"] + f"; NEW SR33 field capacity and all-K/Ks relative factor {factor:g} (sensitivity, not calibration)"
    cfg["name"] = old["name"] + f" SR33 FC all-K/Ks x{factor:g}"
    cfg["controls"] = {"backend": VARIANTS[variant]["backend"]}
    cfg["storm"] = {"max_dt_s": VARIANTS[variant]["max_dt_s"]}
    cfg["expected_sources"] = {"maple_syrup": dict(NEW_SYRUP), "maple": dict(MAPLE)}
    return cfg


CANDIDATE = Path("/home/okin/SYRUP/agent_handoffs/tasks/jorn_sr33_160day_20261120/analysis_candidate")
MAPLE_MATERIALIZED = TASKS / "phase1_publication_c1_20261112/runs/runtime-store-cpu/objects/maple-materialized/5d850789f9e82dc456227cb1eb4dfcc255b3ee4dfb0ed74883813c93e98d0ab3"
HELPERS = ("benchmarks/subsurface_wet_validation/__init__.py", "benchmarks/subsurface_wet_validation/config.py", "benchmarks/subsurface_wet_validation/driver.py", "benchmarks/subsurface_wet_validation/inputs.py",
           "benchmarks/subsurface_validation/__init__.py", "benchmarks/subsurface_validation/drydown_driver.py", "benchmarks/subsurface_validation/neon_inputs.py", "benchmarks/subsurface_validation/paths.py")
PROBE = """from pathlib import Path
import json,hashlib,time,sys,platform,resource
P=Path(__file__).resolve().parent
A=Path({candidate!r})
manifest=json.loads((P/'frozen_helper_manifest.json').read_text())
def check():
 for rel,want in manifest.items():assert hashlib.sha256((A/rel).read_bytes()).hexdigest()==want,rel
check()
t=time.perf_counter()
from benchmarks.subsurface_wet_validation import driver
CASE={case!r}
output=P/('actual_'+CASE)
rc=driver.main(['--config',str(P/('case_'+CASE+'.json')),'--out',str(output),'--execute'])
assert rc==0
check()
record={{'driver_call_wall_s':time.perf_counter()-t,'scope':'includes driver import/case validation/preparation/disposable kernel warmup/integration/output; excludes outer interpreter/bootstrap/source checks','python':sys.version,'platform':platform.platform(),'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'model_scope':'one 1 m2 three-layer column, not domain-scaling/GPU benchmark','source_helper_unchanged':True}}
(P/('runtime_'+CASE+'.json')).write_text(json.dumps(record,indent=2)+chr(10))
"""


def runner_config(directory: Path, variant: str) -> dict:
    """Launcher config in the format of the prior studies (CPU only, CUDA hidden, single Numba thread, per-case cache); consumed by the task's root_release_verified_launcher.py."""
    m = MAPLE_MATERIALIZED
    prefix = [str(m / "editable"), str(m / "source/src"), str(CANDIDATE / "src"), str(CANDIDATE)]
    return {"python": "/home/okin/MAPLE/.venv/bin/python", "output": str(directory / f"actual_{variant}_runs"), "cwd": str(CANDIDATE), "prefix": prefix,
            "required_files": [str(CANDIDATE / "src/maple_syrup/matrix_infiltration.py"), str(m / "source/src/maple/__init__.py")],
            "sources": [{"module": "maple", "root": str(m / "source/src"), "sha256": MAPLE["package_sha256"], "files": MAPLE["files"]}, {"module": "maple_syrup", "root": str(CANDIDATE / "src"), "sha256": NEW_SYRUP["package_sha256"], "files": NEW_SYRUP["files"]}],
            "env": {"CUDA_VISIBLE_DEVICES": "", "NUMBA_NUM_THREADS": "1", "NUMBA_CACHE_DIR": str(directory / f"numba_cache_{variant}"), "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1",
                    "MPLCONFIGDIR": "/tmp/syrup-jorn-sr33-mpl", "MAPLE_SYRUP_EXPECTED_MAPLE_ROOT": str(m / "source"), "E3_PINNED_MAPLE_SRC": str(m / "source/src")},
            "probe_file": str(directory / f"root_execute_{variant}.py")}


def generate(out: Path) -> dict:
    """Write out/<key>/case_<variant>.json for every factor and control; returns the path table."""
    out = Path(out).resolve()      # absolute BEFORE any path is generated (launcher config output/probe and the returned table must not depend on the caller's cwd)
    old = load_old()
    inp = audit_inputs(old)
    native = native_properties(inp)
    paths = {}
    for key, f in FACTORS.items():
        for variant in VARIANTS:
            d = Path(out) / key
            d.mkdir(parents=True, exist_ok=True)
            p = d / f"case_{variant}.json"
            p.write_text(json.dumps(build_case(old, native, inp, f, key, variant), indent=2) + "\n")
            (d / f"root_execute_{variant}.py").write_text(PROBE.format(candidate=str(CANDIDATE), case=variant))
            rc = d / f"actual_{variant}_config.json"
            rc.write_text(json.dumps(runner_config(d, variant), indent=2) + "\n")
            paths.setdefault(key, {})[variant] = {"path": str(p), "sha256": sha256_file(p), "factor": f, "runner_config": str(rc), "probe": str(d / f"root_execute_{variant}.py")}
        (d / "frozen_helper_manifest.json").write_text(json.dumps({rel: sha256_file(CANDIDATE / rel) for rel in HELPERS}, indent=2) + "\n")
    return paths


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    print(json.dumps(generate(Path(a.out)), indent=2))
