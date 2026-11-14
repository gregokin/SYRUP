"""Explicit-config analysis driver for the observed shallow-profile dry-down experiments (stage 1).

It PREPARES (pinned inputs -> anchor, initial layer inventories, complete forcing days, END-matched targets) and, only when the config says Root approved execution, RUNS the actual SYRUP fixed-layer column
(`maple_syrup.subsurface`: `soil_layers`, `prepare_column`, `column_state`, `step`) for a declared scenario list. No physical property has a default: every retention value, the stress threshold, the interface conductivity,
the evaporation factor and the diffusion coefficient must be written in the config, with the layer edges. A missing or null value makes the driver refuse and name it. There is no tuning loop, no clipping of observations and no
filled data.

    python -m benchmarks.subsurface_validation.drydown_driver --config CONFIG.json [--out DIR] [--execute]

Without `--execute` only the plan is printed (`--out` then receives `plan.json`); `--execute` plus the explicit config (properties, pins, expected source identities) is all that runs the column. Backends: "numpy", "numba" and "cuda" (the native column on the current CUDA device; the author never runs it, Root's hardware does; CuPy is imported lazily by the native module and a missing device raises, it never
falls back to the CPU). Controls built in: two persistence nulls (initial raw POINT values; initial integrated LAYER means), zero-exchange and zero-ET variants as ordinary scenarios, per-step ledger (the
native MAPLE bound) and the cumulative ledger with the native `initialize_profile` closure formula (`conservation.volume_roundoff_bound_m3`, 4 L N n_steps + 7 terms), agreement of EVERY declared backend pair on whole trajectories under the inherited contract
(`tests/subsurface/test_numba.py`: rtol 2e-12, atol 1e-14; BITWISE when the diffusion coefficient is 0), and a midpoint restart through the PUBLIC column state API (`column_state(column, water_m, t_s)`) whose whole trajectory and cumulative budget must be
identical. That restart is an experiment-driver state export, NOT a production lifecycle checkpoint and no checkpoint acceptance. Every control raises on failure.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

from . import neon_inputs as ni
from . import paths as P

SCHEMA = "syrup.observed_drydown.config/2"
BACKEND_NAMES = ("numpy", "numba", "cuda")
TOP_KEYS = frozenset({"schema", "name", "provenance", "site", "air", "window", "layers", "scenarios", "controls", "pins"})
SITE_KEYS = frozenset({"key", "raw_section", "site", "hor", "plot", "latitude_deg", "latitude_source"})
AIR_KEYS = frozenset({"root", "month", "level"})
WINDOW_KEYS = frozenset({"run_start_utc", "min_dry_hours", "exclusion_hours", "anchor", "boundary_second_of_day", "max_days", "calendar_utc_offset_hours"})
LAYER_KEYS = frozenset({"edges_m", "theta_residual", "theta_field_capacity", "theta_saturation", "theta_evaporation_stress", "interface_conductivity_m_s"})
SCENARIO_KEYS = frozenset({"name", "diffusion_coefficient", "evaporation_factor", "potential_et", "bottom"})
CONTROL_KEYS = frozenset({"backends", "numba_rtol", "numba_atol", "restart_after_day"})
PIN_KEYS = frozenset({"depth_table_sha256", "inventory_sha256", "inventory_candidate_key", "air_manifest_sha256", "expected_sources"})
SOURCE_KEYS = frozenset({"package_sha256", "files"})
ANCHORS = ("first_joint", "boundary_coincident")
SENSOR_VERS = ("501", "502", "503")
RESTART_LABEL = "column_state(column, water_m, t_s) export by this driver: NOT a production lifecycle checkpoint and no checkpoint acceptance"


class ConfigError(ValueError):
    pass


class ControlFailure(RuntimeError):
    """A built-in control (ledger, restart, backend agreement, source identity) failed."""


def _int(value: Any, what: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigError(f"{what} must be an int >= {minimum} (not a bool or float), got {value!r}")
    return value


def _closed(obj: Any, keys: frozenset, what: str) -> dict:
    if not isinstance(obj, dict) or set(obj) != set(keys):
        raise ConfigError(f"{what}: keys must be exactly {sorted(keys)}, got {sorted(obj) if isinstance(obj, dict) else type(obj).__name__}")
    return obj


def _need(value: Any, what: str) -> Any:
    if value is None:
        raise ConfigError(f"{what} is null: no physical property or choice has a default; it must be declared explicitly")
    return value


def _num(value: Any, what: str, *, lo: float | None = None, hi: float | None = None) -> float:
    _need(value, what)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ConfigError(f"{what} must be a finite number, got {value!r}")
    if (lo is not None and value < lo) or (hi is not None and value > hi):
        raise ConfigError(f"{what}={value} outside [{lo}, {hi}]")
    return float(value)


def _vec(value: Any, n: int, what: str, **kw: Any) -> np.ndarray:
    _need(value, what)
    if not isinstance(value, list) or len(value) != n:
        raise ConfigError(f"{what} must be a list of {n} numbers")
    return np.array([_num(v, f"{what}[{i}]", **kw) for i, v in enumerate(value)])


def _sha(value: Any, what: str) -> str:
    if not (isinstance(value, str) and len(value) == 64 and set(value) <= set("0123456789abcdef")):
        raise ConfigError(f"{what} must be a 64-hex sha256")
    return value


def validate_config(cfg: Any) -> dict:
    """Strict, closed validation. Returns a normalised copy (NumPy vectors for the layers). Raises ConfigError naming the first missing/invalid item."""
    c = _closed(cfg, TOP_KEYS, "config")
    if c["schema"] != SCHEMA or not isinstance(c["name"], str) or not c["name"] or not isinstance(c["provenance"], dict) or not c["provenance"]:
        raise ConfigError("schema / name / provenance invalid (the property source must be recorded in `provenance`)")
    site, air, win, lay, ctl, pins = (_closed(c[k], ks, k) for k, ks in (("site", SITE_KEYS), ("air", AIR_KEYS), ("window", WINDOW_KEYS), ("layers", LAYER_KEYS), ("controls", CONTROL_KEYS), ("pins", PIN_KEYS)))
    for k in ("key", "raw_section", "site", "hor", "latitude_source"):
        if not isinstance(_need(site[k], f"site.{k}"), str) or not site[k]:
            raise ConfigError(f"site.{k} must be a non-empty string")
    _int(site["plot"], "site.plot", minimum=1)
    _num(site["latitude_deg"], "site.latitude_deg", lo=-90.0, hi=90.0)
    if air["root"] not in ("AIR_JORN", "AIR_MOAB") or not isinstance(air["month"], str) or air["level"] != "000.020":
        raise ConfigError("air: root must be AIR_JORN/AIR_MOAB, month a string and level the a-priori 2 m level '000.020'")
    if win["anchor"] not in ANCHORS:
        raise ConfigError(f"window.anchor must be one of {ANCHORS}")
    ni.parse_utc(win["run_start_utc"])
    _num(win["min_dry_hours"], "window.min_dry_hours", lo=1.0)
    _num(win["exclusion_hours"], "window.exclusion_hours", lo=0.0)
    _int(_need(win["calendar_utc_offset_hours"], "window.calendar_utc_offset_hours"), "window.calendar_utc_offset_hours", minimum=-12)
    if win["calendar_utc_offset_hours"] > 14:
        raise ConfigError("window.calendar_utc_offset_hours must be a fixed whole-hour offset within [-12, 14]")
    if win["anchor"] == "boundary_coincident":
        _need(win["boundary_second_of_day"], "window.boundary_second_of_day")
    if win["max_days"] is not None:
        _int(win["max_days"], "window.max_days", minimum=1)
    edges = _vec(lay["edges_m"], 4, "layers.edges_m", lo=0.0)
    if edges[0] != 0.0 or np.any(np.diff(edges) <= 0.0):
        raise ConfigError("layers.edges_m must increase strictly from 0 (three layers)")
    n = len(edges) - 1
    out_lay = {"edges_m": edges, "theta_residual": _vec(lay["theta_residual"], n, "layers.theta_residual", lo=0.0, hi=1.0), "theta_field_capacity": _vec(lay["theta_field_capacity"], n, "layers.theta_field_capacity", lo=0.0, hi=1.0),
               "theta_saturation": _vec(lay["theta_saturation"], n, "layers.theta_saturation", lo=0.0, hi=1.0), "theta_evaporation_stress": _num(lay["theta_evaporation_stress"], "layers.theta_evaporation_stress", lo=0.0, hi=1.0),
               "interface_conductivity_m_s": _vec(lay["interface_conductivity_m_s"], n - 1, "layers.interface_conductivity_m_s", lo=0.0)}
    if not isinstance(c["scenarios"], list) or not c["scenarios"]:
        raise ConfigError("scenarios must be a non-empty list")
    names, scen = set(), []
    for i, s in enumerate(c["scenarios"]):
        s = _closed(s, SCENARIO_KEYS, f"scenarios[{i}]")
        if not isinstance(s["name"], str) or s["name"] in names or s["potential_et"] not in ("zero", "weather") or s["bottom"] not in ("open", "closed"):
            raise ConfigError(f"scenarios[{i}]: name unique string, potential_et in (zero, weather), bottom in (open, closed)")
        names.add(s["name"])
        scen.append({**s, "diffusion_coefficient": _num(s["diffusion_coefficient"], f"scenarios[{i}].diffusion_coefficient", lo=0.0), "evaporation_factor": _num(s["evaporation_factor"], f"scenarios[{i}].evaporation_factor", lo=0.0)})
    if not isinstance(ctl["backends"], list) or not ctl["backends"] or len(set(ctl["backends"])) != len(ctl["backends"]) or any(b not in BACKEND_NAMES for b in ctl["backends"]):
        raise ConfigError(f"controls.backends must be a non-empty list of distinct entries of {BACKEND_NAMES} (a declared cuda backend never falls back to the CPU)")
    _num(ctl["numba_rtol"], "controls.numba_rtol", lo=0.0)
    _num(ctl["numba_atol"], "controls.numba_atol", lo=0.0)
    if ctl["restart_after_day"] is not None:
        _int(ctl["restart_after_day"], "controls.restart_after_day", minimum=1)
    for k in ("depth_table_sha256", "inventory_sha256", "air_manifest_sha256"):
        _sha(pins[k], f"pins.{k}")
    if not isinstance(pins["inventory_candidate_key"], str) or pins["inventory_candidate_key"] != site["key"]:
        raise ConfigError("pins.inventory_candidate_key must equal site.key")
    exp = _closed(pins["expected_sources"], frozenset({"maple_syrup", "maple"}), "pins.expected_sources")
    for k, v in exp.items():
        _closed(v, SOURCE_KEYS, f"pins.expected_sources.{k}")
        _sha(v["package_sha256"], f"pins.expected_sources.{k}.package_sha256")
        _int(v["files"], f"pins.expected_sources.{k}.files", minimum=1)
    return {**c, "layers": out_lay, "scenarios": scen}


def load_config(path: Any) -> tuple:
    raw = Path(path).read_bytes()
    return validate_config(json.loads(raw)), hashlib.sha256(raw).hexdigest()


# ------------------------------------------------------------------ inputs
def cross_check_inventory(cfg: dict, inv: dict) -> dict:
    """The config's site / key / raw section / plot / HOR / air month and root / run start must agree with the PINNED inventory candidate (no free-floating key)."""
    key = cfg["pins"]["inventory_candidate_key"]
    cand = inv["candidates"].get(key)
    site, win, air = cfg["site"], cfg["window"], cfg["air"]
    if cand is None:
        raise ni.InputError(f"inventory has no candidate {key!r}")
    runs = [r["run_start_utc"] for r in cand["candidate_dry_runs_declared_rule_primary_gauge"]]
    checks = {"site": cand["site"] == site["site"], "raw_section": site["raw_section"] == cand["site"], "plot": cand["plot"] == str(site["plot"]), "hor": site["hor"] == f"{site['plot']:03d}",
              "month": cand["month"] == air["month"], "air_root": air["root"] == f"AIR_{cand['site']}", "key": key == f"{cand['site']}_plot{cand['plot']}_{cand['month']}", "site_key": site["key"] == key,
              "run_start": win["run_start_utc"] in runs}
    bad = [k for k, v in checks.items() if not v]
    if bad:
        raise ni.InputError(f"the config disagrees with the pinned inventory candidate {key}: {bad}")
    return checks


def prepare_inputs(cfg: dict, *, inventory_path: Path = P.INVENTORY) -> dict:
    """Pinned data -> the experiment inputs. Re-hashes the inventory file, the selected raw files and the air retrieval, cross-checks the config against the inventory candidate, joins corrected depths, finds the run, the anchor, the rain-free
    horizon and the COMPLETE forcing days. Reports every limit instead of filling. Zero modelled days is refused."""
    inv_bytes = Path(inventory_path).read_bytes()
    if hashlib.sha256(inv_bytes).hexdigest() != cfg["pins"]["inventory_sha256"]:
        raise ni.InputError("the inventory file differs from the config's inventory pin")
    inv = json.loads(inv_bytes)
    site, win, lay, air_cfg = cfg["site"], cfg["window"], cfg["layers"], cfg["air"]
    cross = cross_check_inventory(cfg, inv)
    if cfg["pins"]["depth_table_sha256"] != inv["corrected_depth_table"]["sha256"]:
        raise ni.InputError("the config's depth-table pin is not the inventory's")
    n_checked = ni.verify_inventory_files(inv, P.RAW_ROOT)
    air_root = getattr(P, air_cfg["root"])
    if hashlib.sha256((air_root / "manifest.json").read_bytes()).hexdigest() != cfg["pins"]["air_manifest_sha256"]:
        raise ni.InputError("the air retrieval manifest differs from the config's pin")
    n_air = ni.verify_manifest_files(air_root / "manifest.json", air_root)
    depth_tab = ni.load_depth_table(P.DEPTH_TABLE, cfg["pins"]["depth_table_sha256"])
    sec = site["raw_section"]
    file_of = lambda role: P.RAW_ROOT / next(e["path"] for e in inv["selected_files"][sec] if e["role"] == role)
    levels = {v: ni.read_swc(file_of(f"soil_moisture_plot{site['plot']}_vertical{v}"), v) for v in SENSOR_VERS}
    rain = ni.read_precip(file_of("tipping_bucket_precip"))
    air_file = glob.glob(str(air_root / "DP1.00002.001" / air_cfg["month"] / f"*{air_cfg['level']}.030*"))
    if len(air_file) != 1:
        raise ni.InputError("exactly one air-temperature file of the declared level is required")
    air = ni.read_air_single(air_file[0])
    joint = ni.joint_passing_end_times(*levels.values())
    run_start = ni.parse_utc(win["run_start_utc"])
    runs = [r for r in ni.find_dry_runs(rain, min_hours=win["min_dry_hours"], exclude_hours=win["exclusion_hours"]) if r.start_s == run_start]
    if len(runs) != 1:
        raise ni.InputError("the declared rain-free run is not found under the declared rule")
    run = runs[0]
    sel = ni.select_initial_state(joint, run.exclusion_end_s, win["boundary_second_of_day"])
    anchor = sel["first_joint_at_or_after_exclusion_end_s"] if win["anchor"] == "first_joint" else sel["first_boundary_coincident_joint_s"]
    if anchor is None:
        raise ni.InputError("no joint passing observation END satisfies the declared anchor rule")
    max_rain_free = ni.max_rainfree_days(rain, anchor)
    windows = ni.daily_end_times(anchor, max(max_rain_free, 1))[:-1] if max_rain_free else np.array([], dtype=np.int64)
    wdays = ni.aggregate_daily_air(air, windows)
    complete = 0
    for d in wdays:
        if not d.complete:
            break
        complete += 1
    n_days = min(max_rain_free, complete, win["max_days"] or max_rain_free)
    if n_days < 1:
        raise ni.InputError(f"zero modelled days (rain-free {max_rain_free}, complete forcing {complete}): nothing can be scored; refused before any run")
    ends = ni.daily_end_times(anchor, n_days)
    depths = np.array([ni.sensor_depth_m(depth_tab, site["site"], site["hor"], v, anchor) for v in SENSOR_VERS])
    obs0 = ni.match_end_observations(levels, np.array([anchor]))
    if not bool(obs0["joint"][0]):
        raise ni.InputError("the anchor is not a joint passing observation END")
    theta0 = np.array([obs0[v][0] for v in SENSOR_VERS])
    init = ni.integrate_initial_profile(depths, theta0, lay["edges_m"])
    check_initial_state(init["layer_theta"], lay["theta_saturation"])
    pet = [ni.hargreaves_daily_pet_mm(d, site["latitude_deg"], win["calendar_utc_offset_hours"]) for d in wdays[:n_days]]
    return {"inventory_files_checked": n_checked, "air_manifest_files_checked": n_air, "inventory_cross_checks": cross, "run": run, "anchor_s": anchor, "anchor_candidates": sel, "max_rainfree_days": max_rain_free,
            "complete_forcing_days": complete, "n_days": n_days, "first_incomplete_day": None if complete == len(wdays) else {"day_index": complete, "reason": wdays[complete].reason}, "end_times_s": ends,
            "sensor_depths_m": depths, "theta0_points": theta0, "initial": init, "pet_mm_day": pet, "weather_days": wdays[:n_days], "obs": ni.match_end_observations(levels, ends[1:]), "levels": levels, "rain": rain,
            "pet_note": {**ni.pet_provenance_note(site["latitude_deg"], site["latitude_source"]), "date_rule": f"calendar date of the 24 h window midpoint at fixed UTC{win['calendar_utc_offset_hours']:+d}"}}


def check_initial_state(layer_theta: Any, theta_saturation: Any) -> None:
    """For THESE experiments an initial layer at or above saturation is REFUSED (no clipping, no relabelling); the historical column behaviour is untouched."""
    th, sat = np.asarray(layer_theta, dtype=float), np.asarray(theta_saturation, dtype=float)
    if not np.all(np.isfinite(th)) or np.any(th >= sat):
        raise ni.InputError(f"initial layer moisture {th.tolist()} is not strictly below saturation {sat.tolist()}: the observed initialization is incompatible with the declared properties (refused, nothing clipped)")


def jsonable(obj: Any) -> Any:
    """Recursively JSON-safe: NumPy arrays/scalars become lists/numbers, NaN and infinities become null (a missing observation is null, never NaN)."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        return None if not math.isfinite(float(obj)) else float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def prepared_inputs_export(cfg: dict, prep: dict) -> dict:
    """Compact, auditable copy of what the run used, so a report can be checked WITHOUT the ignored raw data: day-end times, END-matched observed theta per level (null = missing/failing target) with validity, the raw initial point
    values and sensor depths, the integrated initial layers, the complete-day weather (exact window, midpoint calendar date, Tmin, count-weighted Tmean, Tmax, passing count) and the PET per day with its source note. No raw series."""
    off = cfg["window"]["calendar_utc_offset_hours"]
    ends = [ni.format_utc(int(t)) for t in prep["end_times_s"][1:]]
    obs = {v: [None if not math.isfinite(x) else float(x) for x in prep["obs"][v]] for v in SENSOR_VERS}
    return {"day_end_utc": ends, "t0_utc": ni.format_utc(int(prep["end_times_s"][0])), "observed_end_theta_m3m3": obs, "observed_valid": {v: [x is not None for x in obs[v]] for v in SENSOR_VERS},
            "observed_joint_valid": [bool(x) for x in prep["obs"]["joint"]], "sensor_depths_m": prep["sensor_depths_m"].tolist(), "raw_initial_point_theta": prep["theta0_points"].tolist(),
            "integrated_initial_layer_theta": prep["initial"]["layer_theta"].tolist(), "integrated_initial_layer_water_m": prep["initial"]["layer_water_m"].tolist(),
            "weather_days": [{"window_start_utc": ni.format_utc(d.window_start_s), "window_end_utc": ni.format_utc(d.window_end_s), "pet_calendar_date": ni.day_date(d.window_start_s, off).isoformat(), "n_passing": d.n_passing,
                              "tmin_c": d.tmin_c, "tmean_c_count_weighted": d.tmean_c, "tmax_c": d.tmax_c, "pet_mm_day": float(p)} for d, p in zip(prep["weather_days"], prep["pet_mm_day"])],
            "pet_source": {**prep["pet_note"], "air_series": f"NEON DP1.00002.001 single aspirated, level {cfg['air']['level']}, month {cfg['air']['month']}, RELEASE-2026"}}


def plan_report(cfg: dict, prep: dict) -> dict:
    lay, edges = cfg["layers"], cfg["layers"]["edges_m"]
    probe_layer = [int(np.searchsorted(edges, d, side="right") - 1) for d in prep["sensor_depths_m"]]
    return {"name": cfg["name"], "inventory_files_rehashed": prep["inventory_files_checked"], "air_manifest_files_rehashed": prep["air_manifest_files_checked"], "inventory_cross_checks": prep["inventory_cross_checks"],
            "run_start_utc": ni.format_utc(prep["run"].start_s), "run_hours": prep["run"].run_hours, "exclusion_end_utc": ni.format_utc(prep["run"].exclusion_end_s),
            "anchor_utc": ni.format_utc(prep["anchor_s"]), "anchor_rule": cfg["window"]["anchor"], "anchor_candidates_utc": {k: (None if v is None else ni.format_utc(v)) for k, v in prep["anchor_candidates"].items()},
            "max_rainfree_days": prep["max_rainfree_days"], "complete_forcing_days": prep["complete_forcing_days"], "first_incomplete_forcing_day": prep["first_incomplete_day"], "model_days": prep["n_days"],
            "day_end_utc": [ni.format_utc(t) for t in prep["end_times_s"]], "sensor_depths_m": prep["sensor_depths_m"].tolist(), "probe_layer_index": probe_layer,
            "initial_point_theta": prep["theta0_points"].tolist(), "initial_layer_theta": prep["initial"]["layer_theta"].tolist(), "initial_layer_water_m": prep["initial"]["layer_water_m"].tolist(),
            "initial_total_water_m": prep["initial"]["total_water_m"], "pet_mm_day": prep["pet_mm_day"], "pet": prep["pet_note"],
            "valid_targets_per_level": {v: int(np.isfinite(prep["obs"][v]).sum()) for v in SENSOR_VERS}, "valid_joint_targets": int(prep["obs"]["joint"].sum()),
            "initial_below_residual": [bool(t < r) for t, r in zip(prep["initial"]["layer_theta"], lay["theta_residual"])],
            "prepared_inputs": prepared_inputs_export(cfg, prep),
            "support_note": "state = instantaneous end-of-day column state at the anchor + k x 86400 s; observation = 30-minute mean ending at that instant; the initial layer means are integrals of the point profile (representativeness differs from the point sensors and is not hidden)"}


# ------------------------------------------------------------------ running the actual column
def _layers(cfg: dict, scen: dict):
    from maple_syrup import subsurface as ss

    lay = cfg["layers"]
    return ss.soil_layers(lay["edges_m"].tolist(), theta_residual=lay["theta_residual"], theta_field_capacity=lay["theta_field_capacity"], theta_saturation=lay["theta_saturation"],
                          theta_evaporation_stress=lay["theta_evaporation_stress"], evaporation_factor=scen["evaporation_factor"], gravity_drainage=True, diffusion_mode="tietjen_donor_depth",
                          interface_conductivity_m_s=lay["interface_conductivity_m_s"], diffusion_coefficient=scen["diffusion_coefficient"], bottom_drainage=scen["bottom"])


def _namespace(backend: str):
    """The array namespace of the backend: NumPy for numpy/numba; the CuPy module of the native CUDA module for cuda (lazy import; no device, no CuPy -> `SubsurfaceBackendUnavailableError`, never a CPU fallback)."""
    if backend == "cuda":
        from maple_syrup import subsurface_cuda

        return subsurface_cuda.require_cuda()
    return np


def _host(array: Any, backend: str) -> np.ndarray:
    """A float64 NumPy copy for diagnostics (MAPLE's counted `to_host` on CUDA)."""
    if backend == "cuda":
        from maple.core.backend import to_host

        return np.array(to_host(array), dtype=float)
    return np.array(array, dtype=float)


def check_cumulative_ledger(storage0: float, storage_final: float, cum: Any, n_steps: int, n_layers: int, n_cells: int) -> dict:
    """The cumulative closure with the NATIVE formula of `subsurface.initialize_profile`: residual = (final - initial) - ((added - return) - drainage - evaporation) must be finite and within
    `volume_roundoff_bound_m3(4 L N n_steps + 7, max(|initial|, |final|, added, return, drainage, evaporation))`. A zero bound with a non-zero residual fails; no new floor exists."""
    from maple_syrup.conservation import volume_roundoff_bound_m3

    residual = (storage_final - storage0) - (((cum.added_m3 - cum.surface_return_m3) - cum.drainage_m3) - cum.evaporation_m3)
    n_terms = 4 * n_layers * n_cells * max(n_steps, 1) + 7
    scale = max(abs(storage0), abs(storage_final), cum.added_m3, cum.surface_return_m3, cum.drainage_m3, cum.evaporation_m3)
    bound = volume_roundoff_bound_m3(n_terms, scale)
    if not math.isfinite(residual) or not math.isfinite(bound) or abs(residual) > bound:
        raise ControlFailure(f"cumulative water closure violated over {n_steps} step(s): residual {residual!r} m3 exceeds the native FP64 bound {bound!r} m3 (n_terms {n_terms})")
    return {"residual_m3": residual, "bound_m3": bound, "n_terms": n_terms}


def run_column(cfg: dict, prep: dict, scen: dict, backend: str, *, restart_after_day: int | None = None) -> dict:
    """One scenario on the ACTUAL column: daily steps of 86400 s, no added water (the window is rain-free), potential soil evaporation from the complete-day Hargreaves PET (or zero). Initial water = the integrated profile;
    the column's own validity check applies and a state at or above saturation is refused beforehand. Records the WHOLE trajectory (layer water, percolation and diffusion fluxes, evaporation, drainage, return, per-step
    residual and bound, time) and the native cumulative budget."""
    from maple_syrup import subsurface as ss

    if restart_after_day is not None:
        _int(restart_after_day, "restart_after_day", minimum=1)
    check_initial_state(prep["initial"]["layer_theta"], cfg["layers"]["theta_saturation"])
    layers = _layers(cfg, scen)
    xp = _namespace(backend)
    column = ss.prepare_column(layers, xp.asarray(np.array([True])), 1.0, backend=backend)
    water0 = prep["initial"]["layer_water_m"].reshape(-1, 1).copy()
    state = ss.column_state(column, xp.asarray(water0), 0.0)                    # the explicit initial host-to-device upload on CUDA; the identity on the CPU backends
    thick = np.diff(cfg["layers"]["edges_m"])
    L, n = column.n_layers, prep["n_days"]
    traj = {"water_m": np.zeros((n, L)), "percolation_m": np.zeros((n, L - 1)), "diffusion_m": np.zeros((n, L - 1)), "evaporation_m": np.zeros(n), "evaporation_unmet_m": np.zeros(n), "drainage_m": np.zeros(n),
            "surface_return_m": np.zeros(n), "step_residual_m3": np.zeros(n), "step_bound_m3": np.zeros(n), "t_s": np.zeros(n)}
    cum, restarted = ss.CumulativeBudget(), False
    for k in range(n):
        pet_m_s = 0.0 if scen["potential_et"] == "zero" else prep["pet_mm_day"][k] / 1000.0 / ni.DAY_S
        res = ss.step(column, state, float(ni.DAY_S), added_water_m=xp.zeros((L, 1)), potential_evaporation_m_s=xp.asarray(np.array([pet_m_s])))
        b = res.budget
        cum = cum.accumulate(b, res.dt_s)
        state = res.state
        traj["water_m"][k] = _host(state.water_m, backend)[:, 0]               # one small per-day diagnostic device-to-host read on CUDA (N = 1)
        traj["percolation_m"][k] = _host(res.percolation_m, backend)[:, 0]
        traj["diffusion_m"][k] = _host(res.diffusion_m, backend)[:, 0]
        traj["evaporation_m"][k], traj["evaporation_unmet_m"][k] = b.evaporation_m3, b.evaporation_unmet_m3
        traj["drainage_m"][k], traj["surface_return_m"][k] = b.drainage_m3, b.surface_return_m3
        traj["step_residual_m3"][k], traj["step_bound_m3"][k], traj["t_s"][k] = b.residual_m3, b.bound_m3, float(state.t_s)
        if restart_after_day is not None and k + 1 == restart_after_day and not restarted:
            saved_water, saved_t = _host(state.water_m, backend).copy(), float(state.t_s)
            state = ss.column_state(column, xp.asarray(saved_water), saved_t)            # the public constructor rebuilds the state from its two exported values
            restarted = True
    storage0, storage1 = float(water0.sum()), float(_host(state.water_m, backend).sum())
    ledger = check_cumulative_ledger(storage0, storage1, cum, n, L, 1)
    return {"scenario": scen["name"], "backend": backend, "trajectory": traj, "layer_theta": traj["water_m"] / thick, "initial_storage_m": storage0, "final_storage_m": storage1, "cumulative": cum,
            "ledger": ledger, "restart_applied": restarted, "restart_label": RESTART_LABEL if restarted else None}


def _equal_arrays(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and a.tobytes() == b.tobytes()


def assert_runs_identical(a: dict, b: dict, what: str) -> None:
    """Restart control: the WHOLE trajectory (every state, flux, per-step residual/bound, time) and the cumulative budget must be bit-identical; a mismatch RAISES."""
    for k, va in a["trajectory"].items():
        if not _equal_arrays(va, b["trajectory"][k]):
            raise ControlFailure(f"{what}: trajectory array {k!r} differs")
    if a["cumulative"] != b["cumulative"] or a["final_storage_m"] != b["final_storage_m"]:
        raise ControlFailure(f"{what}: the cumulative budget or final storage differs")


def assert_backends_agree(a: dict, b: dict, *, rtol: float, atol: float, bitwise: bool, what: str) -> dict:
    """NumPy-versus-Numba on the WHOLE trajectory under the inherited contract (rtol/atol of `tests/subsurface/test_numba.py`; bitwise equality when the diffusion coefficient is 0) plus the cumulative ledger; raises on failure."""
    worst = {}
    for k, va in a["trajectory"].items():
        vb = b["trajectory"][k]
        if k == "t_s":
            if not _equal_arrays(va, vb):
                raise ControlFailure(f"{what}: time differs")
            continue
        if bitwise and k != "step_bound_m3" and not _equal_arrays(va, vb):
            raise ControlFailure(f"{what}: {k} is not bitwise equal although the diffusion coefficient is 0")
        if not np.allclose(va, vb, rtol=rtol, atol=atol):
            raise ControlFailure(f"{what}: {k} differs beyond rtol {rtol} atol {atol} (max abs {float(np.max(np.abs(va - vb)))})")
        worst[k] = float(np.max(np.abs(va - vb))) if va.size else 0.0
    ca, cb = a["cumulative"], b["cumulative"]
    for f in ("added_m3", "surface_return_m3", "drainage_m3", "evaporation_m3", "evaporation_unmet_m3"):
        if not np.isclose(getattr(ca, f), getattr(cb, f), rtol=rtol, atol=atol):
            raise ControlFailure(f"{what}: cumulative {f} differs")
    for run in (a, b):
        if not math.isfinite(run["ledger"]["residual_m3"]):
            raise ControlFailure(f"{what}: a cumulative ledger is not finite")
    return worst


def score(pred: np.ndarray, obs: np.ndarray) -> dict:
    ok = np.isfinite(pred) & np.isfinite(obs)
    if not ok.any():
        return {"n": 0, "bias": None, "rmse": None, "mae": None, "mse": None}
    e = pred[ok] - obs[ok]
    return {"n": int(ok.sum()), "bias": float(e.mean()), "rmse": float(np.sqrt(np.mean(e**2))), "mae": float(np.mean(np.abs(e))), "mse": float(np.mean(e**2))}


def overall_rmse(per_layer: dict) -> float | None:
    """PRIMARY overall RMSE = sqrt(mean over the three layers of the per-layer MSE); undefined (None) unless every layer has at least one valid target. The pooled all-sample RMSE is a separately labelled alternative."""
    mses = [per_layer[v]["mse"] for v in SENSOR_VERS]
    return None if any(m is None for m in mses) else float(np.sqrt(np.mean(mses)))


def pooled_rmse(preds: dict, obs: dict) -> float | None:
    e = np.concatenate([(preds[v] - obs[v])[np.isfinite(preds[v]) & np.isfinite(obs[v])] for v in SENSOR_VERS])
    return float(np.sqrt(np.mean(e**2))) if e.size else None


def skill(mse_model: float | None, mse_reference: float | None) -> float | None:
    """1 - MSE_model / MSE_reference; undefined (None) when the reference MSE is missing or zero: no skill is invented."""
    if mse_model is None or mse_reference is None or mse_reference == 0.0:
        return None
    return 1.0 - mse_model / mse_reference


def probe_layers(cfg: dict, prep: dict) -> list:
    return [int(np.searchsorted(cfg["layers"]["edges_m"], d, side="right") - 1) for d in prep["sensor_depths_m"]]


def score_prediction(cfg: dict, prep: dict, theta_by_layer: np.ndarray) -> dict:
    """Model layer theta (days, L) scored at the layer CONTAINING each probe (representativeness not hidden), on the common forcing-supported days."""
    probe = probe_layers(cfg, prep)
    per = {v: score(theta_by_layer[:, li], prep["obs"][v]) for v, li in zip(SENSOR_VERS, probe)}
    preds = {v: theta_by_layer[:, li] for v, li in zip(SENSOR_VERS, probe)}
    return {"per_layer": per, "overall_rmse": overall_rmse(per), "pooled_rmse_alternative": pooled_rmse(preds, {v: prep["obs"][v] for v in SENSOR_VERS})}


def persistence_runs(cfg: dict, prep: dict) -> dict:
    """The two null predictors on the same days: the initial raw POINT values held, and the initial integrated LAYER means held."""
    n, probe = prep["n_days"], probe_layers(cfg, prep)
    point = {v: np.full(n, prep["theta0_points"][i]) for i, v in enumerate(SENSOR_VERS)}
    layer = {v: np.full(n, prep["initial"]["layer_theta"][probe[i]]) for i, v in enumerate(SENSOR_VERS)}
    out = {}
    for name, preds in (("initial_point_values_held", point), ("initial_layer_means_held", layer)):
        per = {v: score(preds[v], prep["obs"][v]) for v in SENSOR_VERS}
        out[name] = {"per_layer": per, "overall_rmse": overall_rmse(per), "pooled_rmse_alternative": pooled_rmse(preds, {v: prep["obs"][v] for v in SENSOR_VERS})}
    return out


def source_receipt() -> dict:
    import maple

    import maple_syrup
    from maple_syrup import provenance

    out = {}
    for name, mod in (("maple_syrup", maple_syrup), ("maple", maple)):
        d = provenance.source_tree_digest(Path(mod.__file__).resolve().parent)
        out[name] = {"package_sha256": d.digest_sha256, "files": d.file_count, "file": str(mod.__file__)}
    return out


def assert_expected_sources(cfg: dict, receipt: dict) -> None:
    """The imported packages must be EXACTLY the identities the config declares (not merely unchanged during the run)."""
    for name, want in cfg["pins"]["expected_sources"].items():
        got = receipt[name]
        if got["package_sha256"] != want["package_sha256"] or got["files"] != want["files"]:
            raise ControlFailure(f"{name}: imported package {got['package_sha256'][:12]}/{got['files']} is not the declared {want['package_sha256'][:12]}/{want['files']}")


def execute(cfg: dict, cfg_sha256: str, prep: dict) -> dict:
    before = source_receipt()
    assert_expected_sources(cfg, before)
    runs, scores = {}, {}
    ctl = cfg["controls"]
    pers = persistence_runs(cfg, prep)
    for scen in cfg["scenarios"]:
        per_backend = {b: run_column(cfg, prep, scen, b) for b in ctl["backends"]}
        first = ctl["backends"][0]
        run = per_backend[first]
        backend_worst = {}
        names = ctl["backends"]
        for i, ba in enumerate(names):                                          # EVERY declared backend pair, not only the first two
            for bb in names[i + 1:]:
                backend_worst[f"{ba}_vs_{bb}"] = assert_backends_agree(per_backend[ba], per_backend[bb], rtol=ctl["numba_rtol"], atol=ctl["numba_atol"], bitwise=scen["diffusion_coefficient"] == 0.0,
                                                                      what=f"{scen['name']} {ba} vs {bb}")
        backend_worst = backend_worst or None
        restart = None
        if ctl["restart_after_day"] is not None:
            if ctl["restart_after_day"] >= prep["n_days"]:
                raise ConfigError("controls.restart_after_day must be inside the modelled days")
            for b in names:                                                       # the restart control runs on EVERY declared backend against that backend's own uninterrupted run
                assert_runs_identical(per_backend[b], run_column(cfg, prep, scen, b, restart_after_day=ctl["restart_after_day"]), f"{scen['name']} {b} midpoint restart")
            restart = {"after_day": ctl["restart_after_day"], "backends": list(names), "whole_trajectory_and_cumulative_identical": True, "label": RESTART_LABEL}
        tr = run["trajectory"]
        runs[scen["name"]] = {"trajectory": {k: v.tolist() for k, v in tr.items()}, "layer_theta": run["layer_theta"].tolist(), "initial_storage_m": run["initial_storage_m"], "final_storage_m": run["final_storage_m"],
                              "cumulative": {k: getattr(run["cumulative"], k) for k in ("n_steps", "elapsed_s", "added_m3", "surface_return_m3", "drainage_m3", "evaporation_m3", "evaporation_unmet_m3")},
                              "cumulative_ledger": run["ledger"], "max_step_residual_over_bound": float(np.max(np.abs(tr["step_residual_m3"]) / np.where(tr["step_bound_m3"] > 0, tr["step_bound_m3"], np.inf))),
                              "backend_agreement_max_abs": backend_worst, "per_backend_ledger": {b: per_backend[b]["ledger"] for b in names}, "restart": restart}
        sc = score_prediction(cfg, prep, run["layer_theta"])
        for pname, p in pers.items():
            sc[f"skill_vs_{pname}"] = skill(None if sc["overall_rmse"] is None else sc["overall_rmse"] ** 2, None if p["overall_rmse"] is None else p["overall_rmse"] ** 2)
        scores[scen["name"]] = sc
    after = source_receipt()
    assert_expected_sources(cfg, after)
    if {k: (v["package_sha256"], v["files"]) for k, v in before.items()} != {k: (v["package_sha256"], v["files"]) for k, v in after.items()}:
        raise ControlFailure("the source packages changed during the run")
    return {"config_sha256": cfg_sha256, "sources_before": before, "sources_after": after, "plan": plan_report(cfg, prep), "persistence": pers, "runs": runs, "scores": scores}


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--execute", action="store_true", help="run the actual column for the declared scenarios (the explicit config is the only authority)")
    a = ap.parse_args(argv)
    try:
        cfg, sha = load_config(a.config)
    except (ConfigError, json.JSONDecodeError) as exc:
        print(f"CONFIG REFUSED: {exc}", file=sys.stderr)
        return 2
    prep = prepare_inputs(cfg)
    result = {"config_sha256": sha, "plan": plan_report(cfg, prep)}
    if a.execute:
        result = execute(cfg, sha, prep)
    text = json.dumps(jsonable(result), indent=1, allow_nan=False)
    if a.out:
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / ("result.json" if a.execute else "plan.json")).write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
