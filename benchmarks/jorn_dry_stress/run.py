"""Run one dry-window fixed-stress case through the UNCHANGED `drydown_driver`, after enforcing the case-level property rules.

    python -m benchmarks.jorn_dry_stress.run --config <case.json> --out <dir> --execute

Rules checked before anything runs (a violation raises, nothing is rebound or defaulted): a JORN config must agree with the loaded user preset (field capacity all layers, d, E; stress = the preset's or the declared
comparison value 0.2017); a MOAB config must carry its OWN SR33 field capacity from the native provider, never JORN's value; exactly one scenario with E = 1, weather PET, open bottom; the controls keep both
backends and the midpoint public-state reconstruction; the config's expected sources are the new 4e38a472 / fbc504bb pair (the driver then refuses any other imported package).
"""
from __future__ import annotations

import sys

import numpy as np

from benchmarks.jorn_dry_stress import preset as PS
from benchmarks.jorn_dry_stress import properties as PR
from benchmarks.subsurface_validation import drydown_driver as dd

COMPARISON_STRESS = 0.2017
NEW_SYRUP_SHA = "4e38a472e268a56491b788d5b1271cc3c3da23bb1cc3221a8c59d90f3cc864d2"
MAPLE_SHA = "fbc504bbcd1ec0c02e6a96089134a06187532455ef7ba5e6f146801c8313c02e"


class CaseRuleError(ValueError):
    pass


def check_case(cfg: dict) -> dict:
    """cfg: the validated driver config. Returns a small receipt of what was verified."""
    lay, scen = cfg["layers"], cfg["scenarios"]
    if len(scen) != 1 or scen[0]["evaporation_factor"] != 1.0 or scen[0]["potential_et"] != "weather" or scen[0]["bottom"] != "open":
        raise CaseRuleError("exactly one scenario with E = 1, weather PET and an open bottom is required")
    if sorted(cfg["controls"]["backends"]) != ["numba", "numpy"] or cfg["controls"]["restart_after_day"] is None:
        raise CaseRuleError("controls must keep both CPU backends and the midpoint public-state reconstruction")
    src = cfg["pins"]["expected_sources"]
    if (src["maple_syrup"]["package_sha256"], src["maple"]["package_sha256"]) != (NEW_SYRUP_SHA, MAPLE_SHA) or src["maple_syrup"]["files"] != 180 or src["maple"]["files"] != 412:
        raise CaseRuleError("the config must expect the new SYRUP 4e38a472 (180 files) and MAPLE fbc504bb (412 files) packages")
    pre = PS.load_preset()
    fc, stress, d = np.asarray(lay["theta_field_capacity"]), lay["theta_evaporation_stress"], scen[0]["diffusion_coefficient"]
    if d != pre["diffusion_coefficient"]:
        raise CaseRuleError("d must be the selected value of the preset")
    if cfg["site"]["site"] == "JORN":
        if not np.array_equal(fc, np.asarray(pre["theta_field_capacity"])):
            raise CaseRuleError("a JORN config must use the preset field capacity in all layers")
        if stress not in (pre["theta_evaporation_stress"], COMPARISON_STRESS):
            raise CaseRuleError("a JORN config must use the preset stress or the declared comparison stress 0.2017")
        kind = "JORN preset"
    elif cfg["site"]["site"] == "MOAB":
        own = np.asarray(PR.moab_properties()["theta_field_capacity"])
        if not np.allclose(fc, own, rtol=1e-14, atol=0.0) or np.allclose(fc, pre["theta_field_capacity"][0]):
            raise CaseRuleError("a MOAB config must carry its OWN SR33 field capacity, not the JORN preset value or the old 6 kPa value")
        if stress not in (pre["theta_evaporation_stress"], COMPARISON_STRESS):
            raise CaseRuleError("MOAB stress must be one of the two compared fixed values")
        kind = "MOAB own SR33"
    else:
        raise CaseRuleError("site must be JORN or MOAB")
    return {"kind": kind, "preset_sha256": pre["preset_sha256"], "theta_field_capacity": fc.tolist(), "theta_evaporation_stress": stress, "diffusion_coefficient": d}


def main(argv: list | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args(argv)
    try:
        cfg, _ = dd.load_config(a.config)
        receipt = check_case(cfg)
    except (dd.ConfigError, CaseRuleError, PS.PresetError) as exc:
        print(f"CASE REFUSED: {exc}", file=sys.stderr)
        return 2
    print(f"case rules satisfied: {receipt['kind']}", file=sys.stderr)
    args = ["--config", a.config] + (["--out", a.out] if a.out else []) + (["--execute"] if a.execute else [])
    return dd.main(args)


if __name__ == "__main__":
    raise SystemExit(main())
