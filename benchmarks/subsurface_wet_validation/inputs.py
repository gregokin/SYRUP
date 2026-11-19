"""Pinned-input preparation for the JORN 160-day field column: forcing packet, per-day rainfall schedules and PET, initial profile from END-labelled SWC records, null-safe END-matched targets and the RAW / t0-offset-aligned scores.
Nothing here steps the native column. Observations after initialisation are scored only; they never drive, stop or reset the model, and a missing or failed target is a null (never interpolated, never a reason to truncate)."""
from __future__ import annotations

import csv
import datetime as _dt
import json
import math
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.subsurface_validation import drydown_driver as dd
from benchmarks.subsurface_validation import neon_inputs as ni
from benchmarks.subsurface_wet_validation.config import DAY_S, HALF_HOUR_S, LEVELS

InputError = ni.InputError
RAIN_COLUMNS = ("startDateTime", "endDateTime", "precipBulk_mm", "original_finalQF", "original_finalQFSciRvw")
AIR_COLUMNS = ("day_utc_minus7", "tmin_c", "tmax_c", "tmean_c", "total_weight", "n_observed", "n_interpolated", "reason")
DERIVED_COLUMNS = ("time_utc", "min_c", "max_c", "mean_c", "interpolated", "usable_for_model", "aggregation_weight")
AIR_REASONS = ("observed_complete", "reconstructed_complete")
SUPPORT_NOTE = ("initial state = the three QC-passing raw SWC 30-minute means whose record ENDS at the run start (raw row startDateTime = end - 30 min). The earlier 57-day screen listed the START-labelled values (0.0709, 0.0921, 0.0659); "
                "those were a labelling error and are never the model initial condition. Layer water = piecewise-linear integral of the point profile (constant extension), not the point values.")


def verify_pins(cfg: dict) -> dict:
    """Re-hash every pinned file (forcing, SWC, depth table). Refuses before any model object exists."""
    n = 0
    for f in cfg["forcing"]["files"].values():
        ni.verify_file(f["path"], f["sha256"], f["bytes"])
        n += 1
    for lv in LEVELS:
        for f in cfg["swc"]["levels"][lv]:
            ni.verify_file(f["path"], f["sha256"], f["bytes"])
            n += 1
    got, _ = ni.sha256_file(cfg["swc"]["depth_table"]["path"])
    if got != cfg["swc"]["depth_table"]["sha256"]:
        raise InputError("the corrected depth table differs from its pin")
    return {"files_verified": n + 1}


def _rows(path: str, columns: tuple, what: str) -> list:
    with open(path, newline="") as fh:
        rdr = csv.DictReader(fh)
        missing = [c for c in columns if c not in (rdr.fieldnames or [])]
        if missing:
            raise InputError(f"{what}: missing columns {missing}")
        return list(rdr)


def load_rain(cfg: dict) -> dict:
    """The 30-minute observed rain of the packet: gap-free, exactly the declared rows, every row QA-pass (FinalQF 0, review blank/0), finite and >= 0; exact decimal total and peak equal the pins."""
    f, ex, t0 = cfg["forcing"]["files"]["rainfall_30min"], cfg["forcing"]["expected"], ni.parse_utc(cfg["forcing"]["start_utc"])
    rows = _rows(f["path"], RAIN_COLUMNS, "rainfall_30min")
    if len(rows) != ex["rain_rows"]:
        raise InputError(f"rainfall_30min has {len(rows)} rows, expected {ex['rain_rows']}")
    values = []
    for i, r in enumerate(rows):
        if ni.parse_utc(r["startDateTime"]) != t0 + i * HALF_HOUR_S or ni.parse_utc(r["endDateTime"]) != t0 + (i + 1) * HALF_HOUR_S:
            raise InputError(f"rainfall_30min row {i}: not the gap-free 30-minute grid from the declared start")
        if r["original_finalQF"] != "0" or r["original_finalQFSciRvw"] not in ("", "0"):
            raise InputError(f"rainfall_30min row {i}: not an observed QA-pass record (the packet must contain only passing rain)")
        v = Decimal(r["precipBulk_mm"])
        if not v.is_finite() or v < 0:
            raise InputError(f"rainfall_30min row {i}: precip must be finite and >= 0")
        values.append(v)
    total, peak = sum(values, Decimal(0)), max(values) * 2
    if total != Decimal(ex["total_rain_mm"]) or peak != Decimal(ex["peak_rain_mm_per_h"]):
        raise InputError(f"rain total/peak {total}/{peak} mm differ from the pins {ex['total_rain_mm']}/{ex['peak_rain_mm_per_h']}")
    return {"mm": np.array([float(v) for v in values]), "total_mm_decimal": str(total), "peak_mm_per_h_decimal": str(peak), "n_wet_rows": int(sum(v > 0 for v in values))}


def _count(text: str, what: str) -> int:
    """A nonnegative integer count written as plain digits; fractional, negative or non-numeric text is refused."""
    if not (isinstance(text, str) and text.isascii() and text.isdigit()):
        raise InputError(f"{what}: {text!r} is not a nonnegative integer count")
    return int(text)


def _bool(text: str, what: str) -> bool:
    if text not in ("true", "false"):
        raise InputError(f"{what}: {text!r} is not true/false")
    return text == "true"


def load_daily_air(cfg: dict) -> list:
    """The packet daily air table (UTC-7 days) bound to the pinned 30-minute table: consecutive dates from the run start, 48 = observed + interpolated half hours (integer counts), a FINITE POSITIVE `total_weight`
    (the sample-count aggregation weight of the 48 half hours, NOT elapsed seconds: it may be below 86400) and finite ordered extrema. Every day is re-derived from its 48 `derived_30min` rows (grid, usability,
    interpolation count/flags, weight sum, weighted mean, extrema min-of-minima / max-of-maxima) and must agree within FP64 roundoff (extrema exactly); the pinned receipt's counts must agree too. Nothing is refilled or rewritten."""
    f, n, ex = cfg["forcing"]["files"]["daily_air"], cfg["forcing"]["n_days"], cfg["forcing"]["expected"]
    rows = _rows(f["path"], AIR_COLUMNS, "daily_air")
    if len(rows) != n:
        raise InputError(f"daily_air has {len(rows)} rows, expected {n}")
    t0 = ni.parse_utc(cfg["forcing"]["start_utc"])
    half = _rows(cfg["forcing"]["files"]["derived_30min"]["path"], DERIVED_COLUMNS, "derived_30min")
    if len(half) != 48 * n:
        raise InputError(f"derived_30min has {len(half)} rows, expected {48 * n}")
    tz = _dt.timezone(_dt.timedelta(hours=cfg["site"]["utc_offset_hours"]))
    out, interp, interp_times = [], 0, set()
    for k, r in enumerate(rows):
        date = _dt.datetime.fromtimestamp(t0 + k * DAY_S, tz=tz).date()
        if r["day_utc_minus7"] != date.isoformat() or r["reason"] not in AIR_REASONS:
            raise InputError(f"daily_air row {k}: date/reason {r['day_utc_minus7']!r}/{r['reason']!r} is not {date.isoformat()!r} / a complete-day reason")
        no, ni_ = _count(r["n_observed"], f"daily_air row {k} n_observed"), _count(r["n_interpolated"], f"daily_air row {k} n_interpolated")
        tmin, tmax, tmean, weight = float(r["tmin_c"]), float(r["tmax_c"]), float(r["tmean_c"]), float(r["total_weight"])
        if no + ni_ != 48 or not all(math.isfinite(x) for x in (tmin, tmax, tmean, weight)) or weight <= 0.0 or not tmin <= tmean <= tmax or (ni_ == 0) != (r["reason"] == "observed_complete"):
            raise InputError(f"daily_air row {k}: inconsistent counts/weight/extrema (weight must be finite and > 0; counts must sum to 48)")
        h = half[48 * k:48 * (k + 1)]
        w = np.empty(48), np.empty(48)
        for i, q in enumerate(h):
            st = ni.parse_utc(q["time_utc"])
            if st != t0 + (48 * k + i) * HALF_HOUR_S or not _bool(q["usable_for_model"], f"derived_30min row {48 * k + i} usable_for_model"):
                raise InputError(f"derived_30min row {48 * k + i}: not the grid time or not usable for the model")
            w[0][i], w[1][i] = float(q["aggregation_weight"]), float(q["mean_c"])
            if _bool(q["interpolated"], f"derived_30min row {48 * k + i} interpolated"):
                interp_times.add(st)
        flags = sum(_bool(q["interpolated"], "interpolated") for q in h)
        wt, mean_c = w
        if not (np.all(np.isfinite(wt)) and np.all(wt > 0.0) and np.all(np.isfinite(mean_c))):
            raise InputError(f"derived_30min day {k}: weights must be finite and positive")
        lo, hi = min(float(q["min_c"]) for q in h), max(float(q["max_c"]) for q in h)
        if flags != ni_ or not math.isclose(float(wt.sum()), weight, rel_tol=1e-12, abs_tol=0.0) or not math.isclose(float((wt * mean_c).sum() / wt.sum()), tmean, rel_tol=1e-12, abs_tol=1e-12) or lo != tmin or hi != tmax:
            raise InputError(f"daily_air row {k} is not the aggregation of its 48 derived_30min rows (interpolated count, weight sum, weighted mean or extrema differ)")
        interp += ni_
        out.append({"date": date, "tmin_c": tmin, "tmean_c": tmean, "tmax_c": tmax, "n_interpolated": ni_, "reason": r["reason"], "weight": weight})
    if interp != ex["interpolated_half_hours"]:
        raise InputError(f"daily_air interpolated half hours {interp} differ from the pin {ex['interpolated_half_hours']}")
    rec = json.loads(Path(cfg["forcing"]["files"]["receipt"]["path"]).read_bytes())
    try:
        got = (rec["grid"]["intervals"], len(rec["interpolated_intervals"]), rec["daily"]["rows"], {ni.parse_utc(x["time_utc"]) for x in rec["interpolated_intervals"]})
    except (KeyError, TypeError) as exc:
        raise InputError(f"the packet receipt lacks the expected grid/interpolation/daily fields ({exc})") from None
    if got[:3] != (48 * n, interp, n) or got[3] != interp_times:
        raise InputError("the packet receipt's interval/interpolation/daily counts or times disagree with the tables")
    return out


def daily_pet_mm(cfg: dict, air: list) -> np.ndarray:
    """Reference PET (mm/day) from the ACTUAL `weather_pet` API (`ecohyd_hargreaves`), flat terrain factor 1, no canopy factor. Interpolated-air days are labelled `estimated`."""
    from maple_syrup import weather_pet as wp

    lat = cfg["site"]["latitude_deg"]
    out = []
    for d in air:
        rec = wp.DailyWeather(d["date"], d["tmin_c"], d["tmean_c"], d["tmax_c"], provenance="observed" if d["n_interpolated"] == 0 else "estimated")
        out.append(float(wp.potential_demand(rec, lat, method="ecohyd_hargreaves").pet_mm_day))
    return np.array(out)


def day_schedule(cfg: dict, rain_mm: np.ndarray, k: int):
    """The native `RainfallSchedule` of nominal day k, or None for a dry day: actual 30-minute piecewise rates (mm per row x 2 = mm/h) from the day's FIRST wet row to its LAST wet row end, with every in-between zero-rain row kept.
    Returns (schedule, info). Rain after the last wet row is not part of the storm; a following nominal day is a separate occurrence (history is carried by the driver)."""
    from maple_syrup.rainfall import RainfallProvenance, RainfallSchedule

    rows = rain_mm[48 * k:48 * (k + 1)]
    wet = np.flatnonzero(rows > 0.0)
    if wet.size == 0:
        return None, {"wet": False, "n_rows": 0, "duration_s": 0.0, "first_wet_row": None, "last_wet_row": None}
    a, b = int(wet[0]), int(wet[-1])
    seg = rows[a:b + 1]
    f = cfg["forcing"]["files"]["rainfall_30min"]
    sched = RainfallSchedule(edges_s=np.arange(seg.size + 1, dtype=np.float64) * HALF_HOUR_S, intensity_mm_per_h=seg * 2.0,
                             provenance=RainfallProvenance(kind="jorn160_packet_30min", path=f["path"], sha256=f["sha256"], size_bytes=f["bytes"], n_records=int(seg.size)))
    return sched, {"wet": True, "n_rows": int(seg.size), "duration_s": float(seg.size * HALF_HOUR_S), "first_wet_row": a, "last_wet_row": b, "observed_mm": float(sum(Decimal(str(x)) for x in seg))}


OBS_COLUMNS = ("startDateTime", "endDateTime", "VSWCMean", "VSWCFinalQF", "VSWCFinalQFSciRvw")


def read_obs_swc(path: str, name: str) -> tuple:
    """OBSERVATION-ONLY tolerant SWC reader (the strict `neon_inputs.read_swc` stays for everything else). Schema, per-row 30-minute interval duration, timestamps, strictly increasing order (so no duplicate END) are STRICT.
    A MISSING row is simply absent (a failed target, never a gap error); an unparsable/non-finite value or a non-integer/unparsable flag makes THAT record failed (not passing), never a stop and never filled.
    Returns sorted (end_s, value, passing)."""
    with open(path, newline="") as fh:
        rdr = csv.DictReader(fh)
        missing = [c for c in OBS_COLUMNS if c not in (rdr.fieldnames or [])]
        if missing:
            raise InputError(f"{name}: missing columns {missing}")
        rows = list(rdr)
    if not rows:
        raise InputError(f"{name}: no rows")
    end, val, ok = [], [], []
    for r in rows:
        s_, e_ = ni.parse_utc(r["startDateTime"]), ni.parse_utc(r["endDateTime"])
        if e_ - s_ != HALF_HOUR_S:
            raise InputError(f"{name}: a record is not a 30-minute interval")
        if end and e_ <= end[-1]:
            raise InputError(f"{name}: records are not strictly increasing in END time (duplicate or unsorted)")
        try:
            v = float(r["VSWCMean"]) if r["VSWCMean"] not in ("", "NA") else float("nan")
        except ValueError:
            v = float("nan")
        try:
            qf = ni._flag(r["VSWCFinalQF"], name)
            sci = r["VSWCFinalQFSciRvw"] == "" or ni._flag(r["VSWCFinalQFSciRvw"], name) == 0
        except InputError:
            qf, sci = -2, False
        end.append(e_)
        val.append(v)
        ok.append(math.isfinite(v) and qf == 0 and sci)
    return np.array(end, dtype=np.int64), np.array(val), np.array(ok, dtype=bool)


def read_levels(cfg: dict) -> dict:
    """Per level: END times, values and pass flags merged across the pinned monthly files with the tolerant observation reader (sorted; a duplicate END time across files is refused; absent times stay absent -> null targets)."""
    out = {}
    for lv in LEVELS:
        parts = [read_obs_swc(f["path"], lv) for f in cfg["swc"]["levels"][lv]]
        end, val, ok = (np.concatenate([p[i] for p in parts]) for i in range(3))
        order = np.argsort(end, kind="stable")
        end, val, ok = end[order], val[order], ok[order]
        if end.size > 1 and np.any(np.diff(end) <= 0):
            raise InputError(f"SWC level {lv}: duplicate END times across the pinned files")
        out[lv] = (end, val, ok)
    return out


def lookup(levels: dict, lv: str, end_s: int) -> tuple:
    """(value, passing) of the record ENDING at end_s; (nan, False) when absent."""
    end, val, ok = levels[lv]
    i = int(np.searchsorted(end, end_s))
    if i < end.size and end[i] == end_s:
        return float(val[i]), bool(ok[i])
    return float("nan"), False


def initial_and_targets(cfg: dict, levels: dict) -> dict:
    t0 = ni.parse_utc(cfg["initial"]["end_utc"])
    n = cfg["forcing"]["n_days"]
    theta0 = []
    for lv in LEVELS:
        v, ok = lookup(levels, lv, t0)
        if not ok:
            raise InputError(f"SWC level {lv}: no QC-passing record ENDING at the initial time {cfg['initial']['end_utc']}")
        theta0.append(v)
    theta0 = np.array(theta0)
    if not np.array_equal(theta0, cfg["initial"]["expected_theta"]):
        raise InputError(f"initial point theta {theta0.tolist()} differ from the pinned END-labelled values {cfg['initial']['expected_theta'].tolist()}")
    table = ni.load_depth_table(cfg["swc"]["depth_table"]["path"], cfg["swc"]["depth_table"]["sha256"])
    last = t0 + n * DAY_S
    site = cfg["site"]
    depths = np.array([ni.sensor_depth_m(table, site["swc_site"], site["swc_hor"], lv, t0) for lv in LEVELS])
    for t in (t0 + 1, last):                                                    # the corrected depth stays time-valid and unchanged through the run
        if not np.array_equal(depths, [ni.sensor_depth_m(table, site["swc_site"], site["swc_hor"], lv, t) for lv in LEVELS]):
            raise InputError("a corrected sensor depth changes inside the run")
    if not np.array_equal(depths, cfg["initial"]["expected_depths_m"]):
        raise InputError(f"corrected sensor depths {depths.tolist()} differ from the pins {cfg['initial']['expected_depths_m'].tolist()}")
    init = ni.integrate_initial_profile(depths, theta0, cfg["layers"]["edges_m"])
    dd.check_initial_state(init["layer_theta"], cfg["layers"]["theta_saturation"])
    obs = np.full((n, 3), np.nan)
    ends = t0 + (np.arange(n) + 1) * DAY_S
    for j, lv in enumerate(LEVELS):
        for k in range(n):
            v, ok = lookup(levels, lv, int(ends[k]))
            if ok:
                obs[k, j] = v
    return {"t0_s": t0, "end_times_s": ends, "theta0_points": theta0, "depths_m": depths, "initial": init, "obs": obs, "support_note": SUPPORT_NOTE,
            "probe_layers": [int(np.searchsorted(cfg["layers"]["edges_m"], d, side="right") - 1) for d in depths]}


def prepare_inputs(cfg: dict) -> dict:
    """Everything the run needs, without a native field step."""
    pins = verify_pins(cfg)
    rain, air = load_rain(cfg), load_daily_air(cfg)
    init = initial_and_targets(cfg, read_levels(cfg))
    pet = daily_pet_mm(cfg, air)
    schedules = [day_schedule(cfg, rain["mm"], k) for k in range(cfg["forcing"]["n_days"])]
    return {"pins": pins, "rain": rain, "air": air, "pet_mm_day": pet, "schedules": schedules, **init, "n_days": cfg["forcing"]["n_days"],
            "n_wet_days": sum(1 for s, _ in schedules if s is not None), "target_counts": [int(np.isfinite(init["obs"][:, j]).sum()) for j in range(3)]}


# ------------------------------------------------------------------ scoring (post hoc; no fit)
def _stats(pred: np.ndarray, obs: np.ndarray) -> dict:
    ok = np.isfinite(pred) & np.isfinite(obs)
    if not ok.any():
        return {"n": 0, "bias": None, "rmse": None, "mae": None, "mse": None}
    e = pred[ok] - obs[ok]
    return {"n": int(ok.sum()), "bias": float(e.mean()), "rmse": float(np.sqrt(np.mean(e**2))), "mae": float(np.mean(np.abs(e))), "mse": float(np.mean(e**2))}


def score_run(prep: dict, layer_theta: np.ndarray) -> dict:
    """`layer_theta` (days, 3): modelled layer means at each nominal close. Two scorings on the layer containing each probe: RAW (model - observation) and FIXED t0-OFFSET-ALIGNED changes
    ((model_k - model_0) - (obs_k - obs_0), model_0 the initial integrated layer mean, obs_0 the initial point value), against the persistence null (no change). No reset or re-offset after any storm.
    PRIMARY overall scores (`overall_*`) POOL every available valid pair of all depths (each observation weighted equally, as the pooled calibration did) and never become None merely because one depth has no
    post-initial target; per-depth scores are kept; the equal-depth-MSE average is only the separately labelled `equal_depth_rmse_*` (None unless every depth has a target)."""
    obs, pl = prep["obs"], prep["probe_layers"]
    pred = np.stack([layer_theta[:, li] for li in pl], axis=1)
    m0 = np.array([prep["initial"]["layer_theta"][li] for li in pl])
    o0 = prep["theta0_points"]
    out = {"levels": list(LEVELS), "initial_offset_model_minus_point": (m0 - o0).tolist(), "probe_layers": pl, "raw": {}, "offset_aligned": {}, "persistence_raw": {}, "persistence_offset_aligned": {}}
    pairs = {"raw": (pred, obs), "offset_aligned": (pred - m0, obs - o0), "persistence_raw": (np.broadcast_to(o0, obs.shape), obs), "persistence_offset_aligned": (np.zeros(obs.shape), obs - o0)}
    for key, (p, o) in pairs.items():
        for j, lv in enumerate(LEVELS):
            out[key][lv] = _stats(p[:, j], o[:, j])
        out[f"pooled_{key}"] = _stats(p.ravel(), o.ravel())
    for key in ("raw", "offset_aligned"):
        for tag, src in (("", key), ("persistence_", "persistence_" + key)):
            pooled = out[f"pooled_{src}"]
            out[f"overall_{tag}rmse_{key}"] = pooled["rmse"]
            mse = [out[src][lv]["mse"] for lv in LEVELS]
            out[f"equal_depth_{tag}rmse_{key}"] = None if any(m is None for m in mse) else float(np.sqrt(np.mean(mse)))
        out[f"overall_n_{key}"] = out[f"pooled_{key}"]["n"]
    return out


def jsonable(obj: Any) -> Any:
    return dd.jsonable(obj)
