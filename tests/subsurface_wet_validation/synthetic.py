"""Builders for the engineering tests: small SYNTHETIC input files in the pinned formats (NOT field data) and a matching config. The physics under test is always the real native SYRUP/MAPLE code."""
from __future__ import annotations

import copy
import csv
import datetime as dt
import json
from decimal import Decimal
from pathlib import Path

from benchmarks.subsurface_validation import drydown_driver as dd
from benchmarks.subsurface_validation import neon_inputs as ni
from benchmarks.subsurface_wet_validation import config as C

START = "2024-01-18T07:00:00Z"
T0 = ni.parse_utc(START)
IC_THETA = (0.0711, 0.0921, 0.0658)
DEPTHS = (0.058, 0.158, 0.258)
KS_TOP, FACTOR = 3.355180988596539e-05, 0.31635738822500753


def _write(path: Path, header: list, rows: list) -> dict:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    h, n = ni.sha256_file(path)
    return {"path": str(path), "sha256": h, "bytes": n}


def build(tmp: Path, n_days: int = 4, rain_mm: dict | None = None, swc_fail: set | None = None, swc_missing: set | None = None, backend: str = "numpy", max_dt_s: int = 1800, interpolated_days: tuple = (), split_swc: bool = False, low_weight: dict | None = None) -> dict:
    """rain_mm: {half-hour row index: mm}. swc_fail / swc_missing: {(level, end_s)} records written with FinalQF 1 / omitted from the file (the file stays gap-free by being split)."""
    rain_mm = rain_mm or {}
    swc_fail, swc_missing = swc_fail or set(), swc_missing or set()
    n = 48 * n_days
    rows = [[ni.format_utc(T0 + i * 1800), ni.format_utc(T0 + (i + 1) * 1800), f"{rain_mm.get(i, 0):.3f}", "0", ""] for i in range(n)]
    rain = _write(tmp / "rainfall_30min.csv", ["startDateTime", "endDateTime", "precipBulk_mm", "original_finalQF", "original_finalQFSciRvw"], rows)
    total = sum((Decimal(r[2]) for r in rows), Decimal(0))
    peak = max(Decimal(r[2]) for r in rows) * 2
    tz = dt.timezone(dt.timedelta(hours=-7))
    low_weight = low_weight or {}                                            # {half-hour row index: aggregation weight < 1800 (fewer samples)}
    half, air_rows = [], []
    for k in range(n_days):
        date = dt.datetime.fromtimestamp(T0 + k * 86400, tz=tz).date().isoformat()
        interp_row = 48 * k if k in interpolated_days else None
        day = []
        for i in range(48 * k, 48 * (k + 1)):
            mean = 10.0 + k + 0.1 * (i % 48) / 48
            day.append([ni.format_utc(T0 + i * 1800), f"{mean - 1}", f"{mean + 1}", f"{mean}", "true" if i == interp_row else "false", "true", f"{low_weight.get(i, 1800.0)}"])
        half += day
        w = [float(r[6]) for r in day]
        tmean = sum(a * float(r[3]) for a, r in zip(w, day)) / sum(w)
        interp = 1 if k in interpolated_days else 0
        air_rows.append([date, f"{min(float(r[1]) for r in day)}", f"{max(float(r[2]) for r in day)}", repr(tmean), repr(sum(w)), 48 - interp, interp, "reconstructed_complete" if interp else "observed_complete"])
    air = _write(tmp / "daily_air.csv", ["day_utc_minus7", "tmin_c", "tmax_c", "tmean_c", "total_weight", "n_observed", "n_interpolated", "reason"], air_rows)
    derived = _write(tmp / "derived_30min.csv", ["time_utc", "min_c", "max_c", "mean_c", "interpolated", "usable_for_model", "aggregation_weight"], half)
    (tmp / "receipt.json").write_text(json.dumps({"grid": {"intervals": n}, "interpolated_intervals": [{"time_utc": r[0]} for r in half if r[4] == "true"], "daily": {"rows": n_days}}))
    h, nb = ni.sha256_file(tmp / "receipt.json")
    receipt = {"path": str(tmp / "receipt.json"), "sha256": h, "bytes": nb}
    levels = {}
    for j, lv in enumerate(C.LEVELS):
        ends = [T0 + i * 1800 for i in range(n + 2)]                      # record ENDING at T0 .. T0 + n half hours (+1)
        header = ["startDateTime", "endDateTime", "VSWCMean", "VSWCFinalQF", "VSWCFinalQFSciRvw", "VSWCExpUncert"]
        recs = [[ni.format_utc(e - 1800), ni.format_utc(e), f"{IC_THETA[j] + 1e-5 * ((e - T0) // 1800) / 48 * (j + 1):.6f}" if e != T0 else f"{IC_THETA[j]}", "1" if (lv, e) in swc_fail else "0", "", "0.01"] for e in ends]
        # START-labelled value (the old screen's mistake) differs on purpose: the row that STARTS at T0 is the one ending at T0 + 1800
        files = []
        pieces = [[r for r, e in zip(recs, ends) if (lv, e) not in swc_missing]]                       # a missing record is ACTUALLY removed from the CSV (a real in-file gap)
        if split_swc:
            m = len(pieces[0]) // 2
            pieces = [pieces[0][:m], pieces[0][m:]]
        for pi, piece in enumerate(pieces):
            files.append(_write(tmp / f"swc_{lv}_{pi}.csv", header, piece))
        levels[lv] = files
    depth_path = tmp / "swc_depths"
    with open(depth_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["domainID", "siteID", "horizontalPosition.HOR", "verticalPosition.VER", "sensorDepth", "startDateTime", "endDateTime"])
        for lv, d in zip(C.LEVELS, DEPTHS):
            w.writerow(["D14", "JORN", "004", lv, f"{-d}", "2010-01-01T00:00:00Z", "NA"])
    rec = dd.source_receipt()
    k_local = FACTOR * KS_TOP
    return {
        "schema": C.SCHEMA, "name": "synthetic test", "provenance": {"note": "engineering test fixture, not field data"},
        "site": {"key": "JORN_synthetic", "swc_site": "JORN", "swc_hor": "004", "latitude_deg": 32.59052, "latitude_source": "test", "utc_offset_hours": -7, "air_height_note": "test"},
        "layers": {"edges_m": [0.0, 0.1, 0.2, 0.3], "theta_residual": [0.01] * 3, "theta_field_capacity": [0.39760410394534984, 0.39866650157195227, 0.39508647981211864],
                   "theta_saturation": [0.4643018867924528, 0.4651712600744756, 0.4586310940906557], "theta_evaporation_stress": 0.20170581117963332,
                   "interface_conductivity_m_s": [1.0566751696583226e-05, 9.797630862576748e-06]},
        "matrix": {"ks_top_m_s": KS_TOP, "local_conductivity_factor": FACTOR, "conductivity_m_s": k_local, "suction_m": 0.166, "suction_provenance": "test (EcoHyD reference value)", "vegetation_cover_sum": 0.0, "vegetation_a1": 0.013, "vegetation_a2": 0.067},
        "physics": {"diffusion_coefficient": 0.0010239461376024856, "evaporation_factor": 1.0, "transpiration_factor": 0, "bottom": "open", "gravity_drainage": True},
        "storm": {"max_dt_s": max_dt_s},
        "forcing": {"start_utc": START, "n_days": n_days, "files": {"daily_air": air, "derived_30min": derived, "rainfall_30min": rain, "receipt": receipt},
                    "expected": {"rain_rows": n, "total_rain_mm": str(total), "peak_rain_mm_per_h": str(peak), "interpolated_half_hours": len(interpolated_days)}},
        "swc": {"depth_table": {"path": str(depth_path), "sha256": ni.sha256_file(depth_path)[0]}, "levels": levels},
        "initial": {"end_utc": START, "expected_theta": list(IC_THETA), "expected_depths_m": list(DEPTHS)},
        "controls": {"backend": backend},
        "expected_sources": {k: {"package_sha256": v["package_sha256"], "files": v["files"]} for k, v in rec.items()},
    }


def write_config(tmp: Path, cfg: dict, name: str = "config.json") -> Path:
    p = tmp / name
    p.write_text(json.dumps(cfg, indent=1))
    return p


def mutated(cfg: dict, path: str, value) -> dict:
    out = copy.deepcopy(cfg)
    node = out
    keys = path.split(".")
    for k in keys[:-1]:
        node = node[int(k)] if isinstance(node, list) else node[k]
    last = keys[-1]
    if value is KeyError:
        del node[last]
    else:
        node[int(last) if isinstance(node, list) else last] = value
    return out
