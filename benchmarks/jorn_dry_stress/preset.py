"""Load and validate the user-selected JORN case-level defaults preset (`cases/subsurface_validation/jorn_defaults.json`).

The preset is authoritative data, not prose: every JORN configuration of the dry-window comparison is BUILT from it and the comparison helper refuses a JORN config that disagrees with it. Validation is closed (exact
key sets), refuses non-finite or out-of-range values, checks the declared JORN-only scope and the source commit/case-file identity, and cross-checks the values against the tracked source case. Fractions are
volumetric fractions (0.10 means 10 vol. %); no percent conversion exists anywhere.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

CANDIDATE = Path(__file__).resolve().parents[2]
PRESET_PATH = CANDIDATE / "cases" / "subsurface_validation" / "jorn_defaults.json"
SCHEMA = "syrup.jorn_defaults/1"
TOP = frozenset({"schema", "version", "status", "decision", "scope", "units", "parameters", "geometry_context", "preserved_from_accepted_jorn", "source", "d_provenance", "deferred"})
PARAMS = frozenset({"theta_field_capacity_all_layers", "theta_evaporation_stress", "evaporation_factor", "diffusion_coefficient", "conductivity_multiplier", "interface_conductivity_method"})
SOURCE = frozenset({"accepted_study_commit", "case_file", "case_file_sha256"})


class PresetError(ValueError):
    pass


def _closed(obj, keys, what):
    if not isinstance(obj, dict) or set(obj) != set(keys):
        raise PresetError(f"{what}: keys must be exactly {sorted(keys)}, got {sorted(obj) if isinstance(obj, dict) else type(obj).__name__}")
    return obj


def _finite(x, what, lo=None, hi=None):
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(float(x)):
        raise PresetError(f"{what} must be a finite number, got {x!r}")
    if (lo is not None and x < lo) or (hi is not None and x > hi):
        raise PresetError(f"{what} out of range [{lo}, {hi}]: {x!r}")
    return float(x)


def load_preset(path=PRESET_PATH, *, check_source: bool = True) -> dict:
    """Return the validated preset (parameters as floats). Raises PresetError for anything missing, extra, non-finite, out of scope or inconsistent with its tracked source case."""
    raw = Path(path).read_bytes()
    p = json.loads(raw)
    _closed(p, TOP, "preset")
    if p["schema"] != SCHEMA or p["version"] != 1 or p["status"] != "user_selected_case_level_default":
        raise PresetError("schema/version/status invalid")
    dec = p["decision"]
    if dec.get("not_a_calibration") is not True or dec.get("not_a_refit") is not True:
        raise PresetError("the decision must declare not_a_calibration and not_a_refit")
    sc = p["scope"]
    if "JORN" not in str(sc.get("applies_to", "")) or not any("MOAB default" in x for x in sc.get("not", [])) or not any("global SYRUP physics default" in x for x in sc.get("not", [])):
        raise PresetError("scope must be JORN case-level only and exclude global and MOAB defaults")
    q = _closed(p["parameters"], PARAMS, "parameters")
    fc = _finite(q["theta_field_capacity_all_layers"], "theta_field_capacity_all_layers", 0.0, 1.0)
    st = _finite(q["theta_evaporation_stress"], "theta_evaporation_stress", 0.0, 1.0)
    e = _finite(q["evaporation_factor"], "evaporation_factor", 0.0)
    d = _finite(q["diffusion_coefficient"], "diffusion_coefficient", 0.0)
    mult = _finite(q["conductivity_multiplier"], "conductivity_multiplier", 0.0)
    if q["interface_conductivity_method"] != "harmonic" or mult != 1.0:
        raise PresetError("the preset keeps the accepted 1x SR33 conductivities with harmonic interfaces")
    if not fc < st:
        raise PresetError("the evaporation stress must exceed the field capacity (quadratic branch)")
    geo = p["geometry_context"]
    if geo.get("layers") != 3 or [float(x) for x in geo.get("layer_edges_m", [])] != [0.0, 0.1, 0.2, 0.3]:
        raise PresetError("geometry context must be the 3 x 10 cm profile")
    src = _closed(p["source"], SOURCE, "source")
    if len(src["accepted_study_commit"]) != 40 or len(src["case_file_sha256"]) != 64:
        raise PresetError("source commit/sha lengths invalid")
    out = {"theta_field_capacity": [fc, fc, fc], "theta_evaporation_stress": st, "evaporation_factor": e, "diffusion_coefficient": d, "conductivity_multiplier": mult, "source": src, "preset_sha256": hashlib.sha256(raw).hexdigest(),
           "path": str(Path(path).resolve()), "scope": sc, "deferred": p["deferred"]}
    if check_source:
        case = Path(CANDIDATE) / src["case_file"]
        b = case.read_bytes()
        if hashlib.sha256(b).hexdigest() != src["case_file_sha256"]:
            raise PresetError("the tracked source case file does not match the recorded SHA-256")
        c = json.loads(b)
        lay = c["layers"]
        if lay["theta_field_capacity"] != [fc, fc, fc] or abs(lay["theta_evaporation_stress"] - st) > 1e-15 or c["physics"]["evaporation_factor"] != e or c["physics"]["diffusion_coefficient"] != d:
            raise PresetError("the preset disagrees with its tracked source case")
        out["source_layers"] = lay
        out["source_matrix"] = c["matrix"]
    return out
