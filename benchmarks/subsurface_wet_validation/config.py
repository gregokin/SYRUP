"""Schema-versioned, closed, explicit configuration of the JORN 160-day field-column driver. No physical value, path or pin has a default: a missing or null item is refused and named, before any file is opened for modelling."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from benchmarks.subsurface_validation import drydown_driver as dd

SCHEMA = "syrup.jorn160_wet.config/1"
PROFILE = "syrup.jorn160_wet.daily_weather_with_storm_substeps/1"
BACKENDS = ("numpy", "numba")          # cuda is NOT offered by this driver (no GPU claim)
LEVELS = ("501", "502", "503")
DAY_S = 86400
HALF_HOUR_S = 1800
ConfigError = dd.ConfigError

FILE_KEYS = frozenset({"path", "sha256", "bytes"})
TOP_KEYS = frozenset({"schema", "name", "provenance", "site", "layers", "matrix", "physics", "storm", "forcing", "swc", "initial", "controls", "expected_sources"})
SITE_KEYS = frozenset({"key", "swc_site", "swc_hor", "latitude_deg", "latitude_source", "utc_offset_hours", "air_height_note"})
LAYER_KEYS = frozenset({"edges_m", "theta_residual", "theta_field_capacity", "theta_saturation", "theta_evaporation_stress", "interface_conductivity_m_s"})
MATRIX_KEYS = frozenset({"ks_top_m_s", "local_conductivity_factor", "conductivity_m_s", "suction_m", "suction_provenance", "vegetation_cover_sum", "vegetation_a1", "vegetation_a2"})
PHYSICS_KEYS = frozenset({"diffusion_coefficient", "evaporation_factor", "transpiration_factor", "bottom", "gravity_drainage"})
STORM_KEYS = frozenset({"max_dt_s"})
FORCING_KEYS = frozenset({"start_utc", "n_days", "files", "expected"})
FORCING_FILES = frozenset({"daily_air", "derived_30min", "rainfall_30min", "receipt"})
FORCING_EXPECTED = frozenset({"rain_rows", "total_rain_mm", "peak_rain_mm_per_h", "interpolated_half_hours"})
SWC_KEYS = frozenset({"depth_table", "levels"})
INITIAL_KEYS = frozenset({"end_utc", "expected_theta", "expected_depths_m"})
CONTROL_KEYS = frozenset({"backend"})


def _file(entry: Any, what: str) -> dict:
    e = dd._closed(entry, FILE_KEYS, what)
    if not isinstance(e["path"], str) or not Path(e["path"]).is_absolute():
        raise ConfigError(f"{what}.path must be an absolute path string")
    dd._sha(e["sha256"], f"{what}.sha256")
    dd._int(e["bytes"], f"{what}.bytes", minimum=1)
    return e


def validate_config(cfg: Any) -> dict:
    """Closed validation; returns a normalised copy (NumPy vectors). Raises ConfigError naming the first missing/invalid item."""
    import numpy as np

    c = dd._closed(cfg, TOP_KEYS, "config")
    if c["schema"] != SCHEMA or not isinstance(c["name"], str) or not c["name"] or not isinstance(c["provenance"], dict) or not c["provenance"]:
        raise ConfigError("schema / name / provenance invalid (the property source must be recorded in `provenance`)")
    site, lay, mat, phy, sto, frc, swc, ini, ctl = (dd._closed(c[k], ks, k) for k, ks in (("site", SITE_KEYS), ("layers", LAYER_KEYS), ("matrix", MATRIX_KEYS), ("physics", PHYSICS_KEYS), ("storm", STORM_KEYS),
                                                                                         ("forcing", FORCING_KEYS), ("swc", SWC_KEYS), ("initial", INITIAL_KEYS), ("controls", CONTROL_KEYS)))
    for k in ("key", "swc_site", "swc_hor", "latitude_source", "air_height_note"):
        if not isinstance(dd._need(site[k], f"site.{k}"), str) or not site[k]:
            raise ConfigError(f"site.{k} must be a non-empty string")
    dd._num(site["latitude_deg"], "site.latitude_deg", lo=-90.0, hi=90.0)
    if dd._int(site["utc_offset_hours"], "site.utc_offset_hours", minimum=-12) > 14:
        raise ConfigError("site.utc_offset_hours must be a fixed whole-hour offset within [-12, 14]")
    edges = dd._vec(lay["edges_m"], 4, "layers.edges_m", lo=0.0)
    if edges[0] != 0.0 or np.any(np.diff(edges) <= 0.0):
        raise ConfigError("layers.edges_m must increase strictly from 0 (three layers)")
    out_lay = {"edges_m": edges, "theta_residual": dd._vec(lay["theta_residual"], 3, "layers.theta_residual", lo=0.0, hi=1.0), "theta_field_capacity": dd._vec(lay["theta_field_capacity"], 3, "layers.theta_field_capacity", lo=0.0, hi=1.0),
               "theta_saturation": dd._vec(lay["theta_saturation"], 3, "layers.theta_saturation", lo=0.0, hi=1.0), "theta_evaporation_stress": dd._num(lay["theta_evaporation_stress"], "layers.theta_evaporation_stress", lo=0.0, hi=1.0),
               "interface_conductivity_m_s": dd._vec(lay["interface_conductivity_m_s"], 2, "layers.interface_conductivity_m_s", lo=0.0)}
    if not isinstance(mat["suction_provenance"], str) or not mat["suction_provenance"]:
        raise ConfigError("matrix.suction_provenance must name the (unmeasured reference) source of psi")
    out_mat = {**mat, **{k: dd._num(mat[k], f"matrix.{k}", lo=0.0) for k in ("ks_top_m_s", "local_conductivity_factor", "conductivity_m_s", "suction_m", "vegetation_cover_sum", "vegetation_a1", "vegetation_a2")}}
    if out_mat["vegetation_a2"] <= 0.0 or out_mat["conductivity_m_s"] <= 0.0:
        raise ConfigError("matrix.vegetation_a2 and matrix.conductivity_m_s must be > 0")
    if not math.isclose(out_mat["conductivity_m_s"], out_mat["local_conductivity_factor"] * out_mat["ks_top_m_s"], rel_tol=1e-12, abs_tol=0.0):
        raise ConfigError("matrix.conductivity_m_s must equal local_conductivity_factor * ks_top_m_s (the retained local top conductivity)")
    if phy["bottom"] != "open" or phy["gravity_drainage"] is not True or phy["transpiration_factor"] != 0:
        raise ConfigError("physics: bottom must be 'open', gravity_drainage true and transpiration_factor 0 for this experiment")
    out_phy = {**phy, "diffusion_coefficient": dd._num(phy["diffusion_coefficient"], "physics.diffusion_coefficient", lo=0.0), "evaporation_factor": dd._num(phy["evaporation_factor"], "physics.evaporation_factor", lo=0.0)}
    if dd._int(sto["max_dt_s"], "storm.max_dt_s", minimum=1) not in (1800, 900):
        raise ConfigError("storm.max_dt_s must be 1800 (production) or 900 (control): the step must divide the 1800 s rain interval")
    from benchmarks.subsurface_validation import neon_inputs as ni

    ni.parse_utc(frc["start_utc"])
    n_days = dd._int(frc["n_days"], "forcing.n_days", minimum=1)
    files = dd._closed(frc["files"], FORCING_FILES, "forcing.files")
    for k in FORCING_FILES:
        _file(files[k], f"forcing.files.{k}")
    ex = dd._closed(frc["expected"], FORCING_EXPECTED, "forcing.expected")
    if dd._int(ex["rain_rows"], "forcing.expected.rain_rows", minimum=1) != n_days * 48:
        raise ConfigError("forcing.expected.rain_rows must be 48 x n_days")
    for k in ("total_rain_mm", "peak_rain_mm_per_h"):
        if not isinstance(ex[k], str):
            raise ConfigError(f"forcing.expected.{k} must be a decimal string (exact comparison)")
    dd._int(ex["interpolated_half_hours"], "forcing.expected.interpolated_half_hours", minimum=0)
    dt = dd._closed(swc["depth_table"], frozenset({"path", "sha256"}), "swc.depth_table")
    if not isinstance(dt["path"], str) or not Path(dt["path"]).is_absolute():
        raise ConfigError("swc.depth_table.path must be an absolute path string")
    dd._sha(dt["sha256"], "swc.depth_table.sha256")
    lv = dd._closed(swc["levels"], frozenset(LEVELS), "swc.levels")
    for k in LEVELS:
        if not isinstance(lv[k], list) or not lv[k]:
            raise ConfigError(f"swc.levels.{k} must be a non-empty list of pinned files")
        for i, f in enumerate(lv[k]):
            _file(f, f"swc.levels.{k}[{i}]")
    ni.parse_utc(ini["end_utc"])
    if ini["end_utc"] != frc["start_utc"]:
        raise ConfigError("initial.end_utc (the END of the initial 30-minute SWC record) must equal forcing.start_utc")
    out_ini = {"end_utc": ini["end_utc"], "expected_theta": dd._vec(ini["expected_theta"], 3, "initial.expected_theta", lo=0.0, hi=1.0), "expected_depths_m": dd._vec(ini["expected_depths_m"], 3, "initial.expected_depths_m", lo=0.0)}
    if ctl["backend"] not in BACKENDS:
        raise ConfigError(f"controls.backend must be one of {BACKENDS} (explicit; no fallback; no GPU claim)")
    exp = dd._closed(c["expected_sources"], frozenset({"maple_syrup", "maple"}), "expected_sources")
    for k, v in exp.items():
        dd._closed(v, dd.SOURCE_KEYS, f"expected_sources.{k}")
        dd._sha(v["package_sha256"], f"expected_sources.{k}.package_sha256")
        dd._int(v["files"], f"expected_sources.{k}.files", minimum=1)
    return {**c, "site": site, "layers": out_lay, "matrix": out_mat, "physics": out_phy, "storm": sto, "forcing": {**frc, "n_days": n_days}, "swc": swc, "initial": out_ini, "controls": ctl}


def load_config(path: Any) -> tuple:
    raw = Path(path).read_bytes()
    return validate_config(json.loads(raw)), hashlib.sha256(raw).hexdigest()
