"""NEON RELEASE-2026 input preparation for the observed dry-down experiments (stage 1). Pure host code (stdlib + NumPy); no physics, no network, no credential, no model run.

Everything is explicit and strict: pinned hashes, QA masks as supplied (nothing is filled or relabelled), corrected sensor depths from the pinned depth table, intervals joined on their END time, exact analytical profile
integration, and complete-day (all 48 half-hours) air-temperature aggregation. A series that fails a check raises `InputError`; no helper guesses.

Time convention: every `*_s` value is integer UTC epoch seconds. A 30-minute record is identified by its START and its END (END = START + 1800 s); "coincides" always means END equality.
"""
from __future__ import annotations

import csv
import datetime as _dt
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

HALF_HOUR_S = 1800
DAY_S = 86400
HALF_HOURS_PER_DAY = 48


class InputError(ValueError):
    """A data, pin, join or aggregation check failed."""


# ------------------------------------------------------------------ hashing / pins
def sha256_file(path: Any) -> tuple[str, int]:
    h, n = hashlib.sha256(), 0
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
            n += len(chunk)
    return h.hexdigest(), n


def verify_file(path: Any, sha256: str, nbytes: int) -> None:
    p = Path(path)
    if not p.is_file():
        raise InputError(f"missing input file {p}")
    got, n = sha256_file(p)
    if got != sha256 or n != nbytes:
        raise InputError(f"{p.name}: sha256/bytes differ from the pin ({got[:12]} {n} vs {sha256[:12]} {nbytes})")


def verify_inventory_files(inventory: dict, raw_root: Any, sections: tuple = ("MOAB", "JORN", "JORN_megapit_2013-03")) -> int:
    """Re-hash every selected raw file named by `phase3_observation_inventory.json` (paths relative to the JORN raw root); returns the count."""
    root, n = Path(raw_root), 0
    for sec in sections:
        for e in inventory["selected_files"][sec]:
            verify_file(root / e["path"], e["sha256"], e["bytes"])
            n += 1
    return n


def verify_manifest_files(manifest_path: Any, directory: Any) -> int:
    """Re-hash every file of a Root retrieval manifest (`manifest.json`, credential-free: name, path, bytes, sha256) under `directory`."""
    m = json.loads(Path(manifest_path).read_text())
    files = m.get("files") or m.get("entries")
    if not files:
        raise InputError(f"{manifest_path}: no file table")
    for e in files:
        verify_file(Path(directory) / e["path"], e["sha256"], e["bytes"])
    return len(files)


# ------------------------------------------------------------------ time
def parse_utc(text: str) -> int:
    if not (isinstance(text, str) and len(text) == 20 and text.endswith("Z")):
        raise InputError(f"not a NEON UTC time stamp: {text!r}")
    return int(_dt.datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=_dt.UTC).timestamp())


def format_utc(t_s: int) -> str:
    return _dt.datetime.fromtimestamp(int(t_s), tz=_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# ------------------------------------------------------------------ 30-minute series
@dataclass(frozen=True)
class Series:
    """One 30-minute NEON series exactly as supplied. `final_qf` is -1 when blank; `sci_ok` is True for a blank or 0 science-review flag. `passing` = finite value, FinalQF == 0, science review OK."""

    name: str
    start_s: np.ndarray
    end_s: np.ndarray
    value: np.ndarray
    vmin: np.ndarray | None
    vmax: np.ndarray | None
    numpts: np.ndarray | None
    uncert: np.ndarray | None
    final_qf: np.ndarray
    sci_ok: np.ndarray
    passing: np.ndarray

    def __len__(self) -> int:
        return len(self.start_s)


def _float(s: str) -> float:
    return float("nan") if s == "" or s == "NA" else float(s)


def _flag(s: str, what: str) -> int:
    """A NEON quality flag as an exact integer: blank -> -1 (kept distinct from 0); any non-integral or non-numeric text (for example 0.5) is REFUSED, never truncated to a passing 0."""
    if s == "":
        return -1
    try:
        v = float(s)
    except ValueError as exc:
        raise InputError(f"{what}: flag {s!r} is not numeric") from exc
    if not math.isfinite(v) or v != int(v):
        raise InputError(f"{what}: flag {s!r} is not an integer (a fractional flag is never read as 0)")
    return int(v)


def read_series(path: Any, *, name: str, value_col: str, qf_col: str, sci_col: str, min_col: str | None = None, max_col: str | None = None, numpts_col: str | None = None,
                uncert_col: str | None = None, expected_rows: int | None = None) -> Series:
    p = Path(path)
    with open(p, newline="") as fh:
        rdr = csv.DictReader(fh)
        need = [c for c in ("startDateTime", "endDateTime", value_col, qf_col, sci_col, min_col, max_col, numpts_col, uncert_col) if c]
        missing = [c for c in need if c not in (rdr.fieldnames or [])]
        if missing:
            raise InputError(f"{p.name}: missing columns {missing}")
        rows = list(rdr)
    if expected_rows is not None and len(rows) != expected_rows:
        raise InputError(f"{p.name}: {len(rows)} rows, expected {expected_rows}")
    if not rows:
        raise InputError(f"{p.name}: no rows")
    start = np.array([parse_utc(r["startDateTime"]) for r in rows], dtype=np.int64)
    end = np.array([parse_utc(r["endDateTime"]) for r in rows], dtype=np.int64)
    if np.any(end - start != HALF_HOUR_S) or np.any(np.diff(start) != HALF_HOUR_S):
        raise InputError(f"{p.name}: records are not a gap-free sequence of 30-minute intervals (a missing record would be a silent gap)")
    value = np.array([_float(r[value_col]) for r in rows])
    qf = np.array([_flag(r[qf_col], f"{p.name} {qf_col}") for r in rows], dtype=np.int64)
    # science review: a BLANK or an exact 0 passes; every other value (including an explicit -1 or -1.0, which is malformed, not "blank") fails and is kept as supplied
    sci_ok = np.array([r[sci_col] == "" or _flag(r[sci_col], f"{p.name} {sci_col}") == 0 for r in rows])
    opt = lambda col: np.array([_float(r[col]) for r in rows]) if col else None
    vmin, vmax, npts, unc = opt(min_col), opt(max_col), opt(numpts_col), opt(uncert_col)
    finite = np.isfinite(value)
    if vmin is not None:
        finite &= np.isfinite(vmin)
    if vmax is not None:
        finite &= np.isfinite(vmax)
    return Series(name, start, end, value, vmin, vmax, npts, unc, qf, sci_ok, finite & (qf == 0) & sci_ok)


def read_swc(path: Any, name: str, **kw: Any) -> Series:
    return read_series(path, name=name, value_col="VSWCMean", qf_col="VSWCFinalQF", sci_col="VSWCFinalQFSciRvw", uncert_col="VSWCExpUncert", **kw)


def read_precip(path: Any, name: str = "precip", **kw: Any) -> Series:
    return read_series(path, name=name, value_col="precipBulk", qf_col="finalQF", sci_col="finalQFSciRvw", **kw)


def read_air_single(path: Any, name: str = "air_single", **kw: Any) -> Series:
    """DP1.00002.001 single-aspirated air temperature (deg C): mean, the ACTUAL 30-minute minimum and maximum, and the sample count."""
    return read_series(path, name=name, value_col="tempSingleMean", qf_col="finalQF", sci_col="finalQFSciRvw", min_col="tempSingleMinimum", max_col="tempSingleMaximum", numpts_col="tempSingleNumPts", **kw)


def same_grid(*series: Series) -> None:
    for s in series[1:]:
        if len(s) != len(series[0]) or not np.array_equal(s.start_s, series[0].start_s):
            raise InputError(f"{s.name} is not on the grid of {series[0].name}")


def joint_passing_end_times(*series: Series) -> np.ndarray:
    """END times at which EVERY series passes (the three-probe joint availability)."""
    same_grid(*series)
    ok = np.ones(len(series[0]), dtype=bool)
    for s in series:
        ok &= s.passing
    return series[0].end_s[ok]


# ------------------------------------------------------------------ corrected depths
def load_depth_table(path: Any, expected_sha256: str) -> dict:
    """The pinned corrected depth table `swc_depthsV3` -> {(site, hor, ver): [(start_s, end_s | None, depth_m_positive), ...]}."""
    got, _ = sha256_file(path)
    if got != expected_sha256:
        raise InputError(f"depth table sha256 {got[:12]} differs from the pin {expected_sha256[:12]}")
    table: dict = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["sensorDepth"] in ("NA", ""):
                continue          # a record without a depth carries no usable position (it is never read as depth 0)
            end = None if r["endDateTime"] in ("NA", "") else parse_utc(r["endDateTime"])
            table.setdefault((r["siteID"], r["horizontalPosition.HOR"], r["verticalPosition.VER"]), []).append((parse_utc(r["startDateTime"]), end, -float(r["sensorDepth"])))
    return table


def sensor_depth_m(table: dict, site: str, hor: str, ver: str, t_s: int) -> float:
    rows = [d for (a, b, d) in table.get((site, hor, ver), ()) if a <= t_s and (b is None or t_s < b)]
    if len(rows) != 1:
        raise InputError(f"{site} HOR {hor} VER {ver}: {len(rows)} depth records valid at {format_utc(t_s)} (exactly one required)")
    if rows[0] <= 0.0:
        raise InputError("a sensor depth must be positive below the surface")
    return rows[0]


# ------------------------------------------------------------------ rain-free windows and the day clock
@dataclass(frozen=True)
class DryRun:
    start_s: int              # START of the first rain-free half-hour
    last_start_s: int         # START of the last rain-free half-hour
    run_hours: float
    exclusion_end_s: int      # run start + exclusion (the evaluable part begins here)
    evaluable_hours: float
    censored_at_start: bool   # the run touches the first record of the month file
    censored_at_end: bool


def find_dry_runs(rain: Series, *, min_hours: float, exclude_hours: float) -> list:
    """Contiguous half-hours in which the rain gauge PASSES and reports exactly 0 (a failing record breaks a run). Runs shorter than `min_hours` are dropped. An ANALYTICAL selection rule (proposal), not accepted science."""
    zero = rain.passing & (rain.value == 0.0)
    runs, i, n = [], 0, len(rain)
    while i < n:
        if not zero[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and zero[j + 1]:
            j += 1
        hours = (j - i + 1) * 0.5
        if hours >= min_hours:
            runs.append(DryRun(int(rain.start_s[i]), int(rain.start_s[j]), hours, int(rain.start_s[i]) + int(exclude_hours * 3600), hours - exclude_hours, i == 0, j == n - 1))
        i = j + 1
    return runs


def select_initial_state(joint_end_s: np.ndarray, exclusion_end_s: int, boundary_second_of_day: int | None = None) -> dict:
    """The two candidate initial times, kept DISTINCT: (a) the literal first joint passing END strictly after the exclusion end (the whole 30-minute interval then lies after the excluded 72 h); (b) when a fixed UTC boundary (seconds of day) is declared, the first joint passing END after the exclusion
    end that falls exactly on that boundary (the model's day clock then has the same time of day). Neither is invented: both are actual joint observation END times, or None."""
    after = joint_end_s[joint_end_s > exclusion_end_s]          # strictly after: an interval ending AT the exclusion end lies inside the excluded period
    first = int(after[0]) if len(after) else None
    boundary = None
    if boundary_second_of_day is not None:
        if not (isinstance(boundary_second_of_day, int) and 0 <= boundary_second_of_day < DAY_S and boundary_second_of_day % HALF_HOUR_S == 0):
            raise InputError("boundary_second_of_day must be an int multiple of 1800 s within a day")
        hits = after[(after % DAY_S) == boundary_second_of_day]
        boundary = int(hits[0]) if len(hits) else None
    return {"first_joint_at_or_after_exclusion_end_s": first, "first_boundary_coincident_joint_s": boundary}


def daily_end_times(anchor_end_s: int, n_days: int) -> np.ndarray:
    """Model day-end times: anchor + k x 86400 s (no partial step)."""
    if not (isinstance(n_days, int) and n_days >= 1):
        raise InputError("n_days must be a positive int")
    return anchor_end_s + DAY_S * np.arange(0, n_days + 1, dtype=np.int64)


def max_rainfree_days(rain: Series, anchor_end_s: int) -> int:
    """Whole model days after `anchor_end_s` whose EVERY half-hour of forcing is rain-free with a passing gauge (the suffix must stay rain-free at ALL forcing times; no model rain is ever injected)."""
    idx = np.where(rain.start_s == anchor_end_s)[0]
    if len(idx) != 1:
        raise InputError("the anchor END time is not the START of a rain record (it lies outside the rain file)")
    i, days = int(idx[0]), 0
    zero = rain.passing & (rain.value == 0.0)
    while i + HALF_HOURS_PER_DAY <= len(rain) and bool(np.all(zero[i:i + HALF_HOURS_PER_DAY])):
        days += 1
        i += HALF_HOURS_PER_DAY
    return days


# ------------------------------------------------------------------ observations at the day ends
def match_end_observations(levels: dict, target_end_s: np.ndarray) -> dict:
    """For each level (name -> Series) the observed value at each target END time that COINCIDES EXACTLY with a PASSING interval END (NaN otherwise: no nearest-time substitute, no fill). `joint` = all levels valid."""
    out: dict = {}
    valid = {}
    for name, s in levels.items():
        lookup = {int(e): k for k, e in enumerate(s.end_s)}
        vals = np.full(len(target_end_s), np.nan)
        for k, t in enumerate(target_end_s):
            j = lookup.get(int(t))
            if j is not None and s.passing[j]:
                vals[k] = s.value[j]
        out[name] = vals
        valid[name] = np.isfinite(vals)
    out["joint"] = np.logical_and.reduce(list(valid.values()))
    return out


def daily_median_descriptive(series: Series, window_starts_s: np.ndarray) -> np.ndarray:
    """Median of the passing values inside each 24 h window [t, t + 86400): DESCRIPTIVE only, never a substitute for the end-state target."""
    out = np.full(len(window_starts_s), np.nan)
    for k, t in enumerate(window_starts_s):
        m = series.passing & (series.start_s >= t) & (series.start_s < t + DAY_S)
        if m.any():
            out[k] = float(np.median(series.value[m]))
    return out


# ------------------------------------------------------------------ exact initial-profile integration
def profile_integral(z_m: float, depths_m: np.ndarray, theta: np.ndarray) -> float:
    """F(z) = integral_0^z theta(s) ds for the piecewise-linear profile through (depth, theta) with CONSTANT extension above the first and below the last sensor. Exact (trapezoids)."""
    d, th = np.asarray(depths_m, dtype=float), np.asarray(theta, dtype=float)
    if z_m <= d[0]:
        return float(th[0] * z_m)
    total = float(th[0] * d[0])
    for i in range(len(d) - 1):
        lo, hi = d[i], d[i + 1]
        if z_m >= hi:
            total += 0.5 * (th[i] + th[i + 1]) * (hi - lo)
        else:
            frac = (z_m - lo) / (hi - lo)
            total += (th[i] + 0.5 * (th[i + 1] - th[i]) * frac) * (z_m - lo)
            return total
    return total + float(th[-1] * (z_m - d[-1]))


def integrate_initial_profile(depths_m: Any, theta: Any, edges_m: Any) -> dict:
    """Layer water depths (m) and layer-mean theta from the piecewise-linear profile through the ACTUAL sensor depths (constant extension outside). Total water over [edges[0], edges[-1]] equals the analytical integral."""
    d, th, e = np.asarray(depths_m, dtype=float), np.asarray(theta, dtype=float), np.asarray(edges_m, dtype=float)
    if not (np.all(np.isfinite(d)) and np.all(np.isfinite(e))):
        raise InputError("sensor depths and layer edges must be finite")
    if d.ndim != 1 or d.shape != th.shape or len(d) < 1 or np.any(np.diff(d) <= 0.0) or d[0] <= 0.0:
        raise InputError("sensor depths must be strictly increasing positive depths with one theta each")
    if not np.all(np.isfinite(th)) or np.any(th < 0.0) or np.any(th > 1.0):
        raise InputError("probe theta must be finite and within [0, 1] (observations are never clipped)")
    if e.ndim != 1 or len(e) < 2 or e[0] != 0.0 or np.any(np.diff(e) <= 0.0):
        raise InputError("layer edges must be strictly increasing from 0")
    cum = np.array([profile_integral(z, d, th) for z in e])
    water = np.diff(cum)
    return {"layer_water_m": water, "layer_theta": water / np.diff(e), "total_water_m": float(cum[-1] - cum[0]), "edges_m": e}


# ------------------------------------------------------------------ complete-day weather
@dataclass(frozen=True)
class DayWeather:
    window_start_s: int
    window_end_s: int
    n_passing: int
    complete: bool
    reason: str
    tmin_c: float | None
    tmean_c: float | None
    tmax_c: float | None


def aggregate_daily_air(series: Series, window_starts_s: Any) -> list:
    """Aligned 24 h windows [t, t + 86400). A window is COMPLETE only when all 48 half-hour records exist and pass (FinalQF 0, science review OK): Tmin = minimum of the 30-minute minima, Tmax = maximum of the 30-minute
    maxima (not extrema of means), Tmean = sample-count-weighted mean of the 30-minute means. An incomplete window is reported with its reason; nothing is filled or interpolated."""
    if series.vmin is None or series.vmax is None or series.numpts is None:
        raise InputError("an air series with minimum, maximum and point counts is required")
    out = []
    for t0 in np.asarray(window_starts_s, dtype=np.int64):
        m = (series.start_s >= t0) & (series.start_s < t0 + DAY_S)
        n_rec, n_ok = int(m.sum()), int((m & series.passing).sum())
        if n_rec != HALF_HOURS_PER_DAY:
            out.append(DayWeather(int(t0), int(t0) + DAY_S, n_ok, False, f"only {n_rec} of 48 records in the window", None, None, None))
        elif n_ok != HALF_HOURS_PER_DAY:
            out.append(DayWeather(int(t0), int(t0) + DAY_S, n_ok, False, f"{HALF_HOURS_PER_DAY - n_ok} of 48 records fail QA", None, None, None))
        else:
            w, mean, lo, hi = series.numpts[m], series.value[m], series.vmin[m], series.vmax[m]
            if not (np.all(np.isfinite(w)) and np.all(w > 0.0) and np.all(np.isfinite(mean)) and np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))):
                out.append(DayWeather(int(t0), int(t0) + DAY_S, n_ok, False, "48 records pass QA but a point count is not finite and positive, or a value is not finite", None, None, None))
                continue
            if not np.all((lo <= mean) & (mean <= hi)):
                out.append(DayWeather(int(t0), int(t0) + DAY_S, n_ok, False, "48 records pass QA but a record violates minimum <= mean <= maximum", None, None, None))
                continue
            out.append(DayWeather(int(t0), int(t0) + DAY_S, n_ok, True, "complete", float(series.vmin[m].min()), float(np.sum(series.value[m] * w) / np.sum(w)), float(series.vmax[m].max())))
    return out


def day_date(window_start_s: int, utc_offset_hours: int = -7) -> _dt.date:
    """Declared date rule: the calendar date of the window MIDPOINT in a FIXED standard-time offset (default UTC-7, no daylight saving; day-of-year drives only the extraterrestrial radiation)."""
    tz = _dt.timezone(_dt.timedelta(hours=utc_offset_hours))
    return _dt.datetime.fromtimestamp(window_start_s + DAY_S // 2, tz=tz).date()


def hargreaves_daily_pet_mm(day: DayWeather, latitude_deg: float, utc_offset_hours: int = -7) -> float:
    """Reference PET (mm/day) from the ACTUAL `maple_syrup.weather_pet` API (`ecohyd_hargreaves`); no terrain factor is applied (factor 1: flat diagnostic, no aspect invented)."""
    from maple_syrup import weather_pet as wp

    if not day.complete:
        raise InputError(f"incomplete weather day ({day.reason}): no PET is computed")
    rec = wp.DailyWeather(day_date(day.window_start_s, utc_offset_hours), day.tmin_c, day.tmean_c, day.tmax_c, provenance="observed")
    return float(wp.potential_demand(rec, latitude_deg, method="ecohyd_hargreaves").pet_mm_day)


def pet_provenance_note(latitude_deg: float, source: str) -> dict:
    return {"latitude_deg": latitude_deg, "latitude_source": source, "terrain_factor": 1.0, "terrain_note": "flat diagnostic: no aspect or slope multiplier is applied", "method": "ecohyd_hargreaves"}
