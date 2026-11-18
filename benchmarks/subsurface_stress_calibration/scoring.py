"""Offset-aligned CHANGE scoring and the pooled objective of the evaporation-stress calibration. Pure NumPy post-processing: no column physics, no I/O, no input mutation.

For one window and one sensor the model series theta_m(t_k), k = 1..n (t0 EXCLUDED) is compared with the observed point series theta_o(t_k) through CHANGES from the common initial instant:

    residual_k = (theta_m(t_k) - theta_m(t_0)) - (theta_o(t_k) - theta_o(t_0))  =  aligned_model_k - theta_o(t_k),   aligned_model_k = theta_m(t_k) + offset,  offset = theta_o(t_0) - theta_m(t_0).

The offset is a constant per (window, sensor), fixed by the two initial values; it is never optimised and never touches the physical state. A missing observation (non-finite) after t0 is simply not counted. The pooled objective is
sum of every valid squared residual / total valid observations (equal weight per observation, NOT a mean of window RMSEs). An arbitrary constant added to the model initial value AND its series (an initial-state bias) leaves the score unchanged.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np


class ScoreError(ValueError):
    pass


def _series(x: Any, what: str) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    if a.ndim != 1 or a.size == 0:
        raise ScoreError(f"{what} must be a non-empty one-dimensional series (t0 excluded)")
    return a


def _scalar(x: Any, what: str) -> float:
    if isinstance(x, bool) or not np.isscalar(x) or not math.isfinite(float(x)):
        raise ScoreError(f"{what} must be a finite scalar, got {x!r}")
    return float(x)


def sensor_score(model: Any, model0: Any, obs: Any, obs0: Any) -> dict:
    """One (window, sensor). `model` and `obs` are the t >= 1 series (length n); `model0` and `obs0` the initial values. Inputs are not modified. Non-finite MODEL values are refused (a column never gives them);
    non-finite observations are the missing/QA-failed targets and are counted out."""
    m, o = _series(model, "model"), _series(obs, "obs")
    if m.shape != o.shape:
        raise ScoreError(f"model {m.shape} and obs {o.shape} differ in length")
    if not np.all(np.isfinite(m)):
        raise ScoreError("model series contains non-finite values")
    m0, o0 = _scalar(model0, "model0"), _scalar(obs0, "obs0")
    valid = np.isfinite(o)
    offset = o0 - m0
    aligned = m + offset
    resid = np.where(valid, aligned - o, np.nan)
    e = resid[valid]
    n = int(valid.sum())
    raw = (m - o)[valid]
    zero = (o0 - o)[valid]                                    # the zero-change baseline: model change 0 -> residual = -(obs - obs0)
    out = {"n": n, "n_missing": int(o.size - n), "offset": offset, "model0": m0, "obs0": o0}
    out.update({"sse": float(np.sum(e**2)), "rmse": float(np.sqrt(np.mean(e**2))), "bias_signed": float(e.mean()), "rmse_unshifted": float(np.sqrt(np.mean(raw**2))), "bias_unshifted": float(raw.mean()),
                "zero_change_sse": float(np.sum(zero**2)), "zero_change_rmse": float(np.sqrt(np.mean(zero**2)))} if n else
               {"sse": 0.0, "rmse": None, "bias_signed": None, "rmse_unshifted": None, "bias_unshifted": None, "zero_change_sse": 0.0, "zero_change_rmse": None})
    out["residual"] = resid                                    # NaN at missing observations
    out["aligned_model"] = aligned
    return out


def window_score(model_by_sensor: Any, model0: Any, obs_by_sensor: Any, obs0: Any) -> dict:
    """Three sensors: `model_by_sensor`, `obs_by_sensor` shape (3, n) (or any (S, n)); `model0`, `obs0` length S. A separate offset per sensor."""
    m, o = np.asarray(model_by_sensor, dtype=float), np.asarray(obs_by_sensor, dtype=float)
    if m.ndim != 2 or m.shape != o.shape:
        raise ScoreError("model_by_sensor and obs_by_sensor must be 2-D arrays of equal shape (sensors, days)")
    m0, o0 = np.asarray(model0, dtype=float), np.asarray(obs0, dtype=float)
    if m0.shape != (m.shape[0],) or o0.shape != (m.shape[0],):
        raise ScoreError("model0 and obs0 need one initial value per sensor")
    sensors = [sensor_score(m[i], m0[i], o[i], o0[i]) for i in range(m.shape[0])]
    return {"sensors": sensors, "n": sum(s["n"] for s in sensors), "sse": sum(s["sse"] for s in sensors), "zero_change_sse": sum(s["zero_change_sse"] for s in sensors)}


def pooled_objective(windows: dict) -> dict:
    """`windows`: name -> window_score(...). Pooled MSE = total SSE / total valid observations over every window and sensor; refuses when nothing is valid."""
    if not windows:
        raise ScoreError("no windows to pool")
    n = sum(w["n"] for w in windows.values())
    if n == 0:
        raise ScoreError("every observation is invalid: the pooled objective is undefined (refused)")
    sse = sum(w["sse"] for w in windows.values())
    zsse = sum(w["zero_change_sse"] for w in windows.values())
    return {"n": n, "sse": sse, "mse": sse / n, "rmse": math.sqrt(sse / n), "zero_change_mse": zsse / n, "zero_change_rmse": math.sqrt(zsse / n),
            "skill_vs_zero_change": None if zsse == 0.0 else 1.0 - sse / zsse}


def breakdown(windows: dict, sites: dict, depth_labels: Any) -> dict:
    """Counts, RMSE and signed bias per window / site / depth (sensor label) by pooling the valid residuals. `sites` maps window -> site. No window gets its own weight: every aggregate is sse/n of its members."""
    def agg(items):
        n = sum(i["n"] for i in items)
        if n == 0:
            return {"n": 0, "rmse": None, "bias_signed": None, "zero_change_rmse": None}
        return {"n": n, "rmse": math.sqrt(sum(i["sse"] for i in items) / n), "bias_signed": sum(i["bias_signed"] * i["n"] for i in items if i["n"]) / n,
                "zero_change_rmse": math.sqrt(sum(i["zero_change_sse"] for i in items) / n)}
    out = {"window": {}, "site": {}, "depth": {}, "window_depth": {}}
    for w, ws in windows.items():
        out["window"][w] = agg(ws["sensors"])
        for lab, s in zip(depth_labels, ws["sensors"]):
            out["window_depth"][f"{w}/{lab}"] = agg([s])
    for site in sorted(set(sites.values())):
        out["site"][site] = agg([s for w, ws in windows.items() if sites[w] == site for s in ws["sensors"]])
    for i, lab in enumerate(depth_labels):
        out["depth"][lab] = agg([ws["sensors"][i] for ws in windows.values()])
    return out
