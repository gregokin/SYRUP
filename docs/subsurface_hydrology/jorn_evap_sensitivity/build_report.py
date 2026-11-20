"""Builds the JORN 160-day SR33 1xK evaporation-factor / evaporation-stress sensitivity report: figures, CSV tables, editable Markdown, verification note, manifest and the Word document.

    PYTHONPATH=<tasks>/site_hydraulics_interface_comparison_20261117/report_tools MPLCONFIGDIR=/tmp/syrup-evap-report PYTHONDONTWRITEBYTECODE=1 /home/okin/MAPLE/.venv/bin/python -B build_report.py

DOCUMENTS ONLY. Reads Root's sealed results (every sealed file used is re-hashed against `root_run_seal.json` before and after), recomputes scores, period scores, budgets, endpoints and controls independently
from the daily rows, runs no model, fits nothing and uses no network or GPU. Declared input: EVAP_TASK (this task folder; default = the folder four levels above this file). Writes only inside its own folder.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import itertools
import json
import math
import os
import re
import shutil
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from docx import Document
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor, Twips

HERE = Path(__file__).resolve().parent
TASK = Path(os.environ.get("EVAP_TASK", str(HERE.parents[3])))
TASKS = TASK.parent
PRIOR = TASKS / "jorn_sr33_160day_20261120"
FIG, TAB, DATA = HERE / "figures", HERE / "tables", HERE / "data"
DOCX = HERE / "SYRUP_JORN_160day_SR33_1x_evaporation_sensitivity_20261120.docx"
KEYS = ["baseline", "e0p8", "e0p6", "e0p4", "s1p5", "s2", "s2p5", "s3"]
FAM = {"E": ["baseline", "e0p8", "e0p6", "e0p4"], "stress": ["baseline", "s1p5", "s2", "s2p5", "s3"]}
LAB = {"baseline": "baseline (E 1, stress x1)", "e0p8": "E 0.8", "e0p6": "E 0.6", "e0p4": "E 0.4", "s1p5": "stress x1.5", "s2": "stress x2", "s2p5": "stress x2.5", "s3": "stress x3"}
SHORT = {"baseline": "baseline", "e0p8": "E 0.8", "e0p6": "E 0.6", "e0p4": "E 0.4", "s1p5": "x1.5", "s2": "x2", "s2p5": "x2.5", "s3": "x3"}
COL = {"baseline": "#222222", "e0p8": "#2E8B57", "e0p6": "#E07B00", "e0p4": "#B22222", "s1p5": "#6BAED6", "s2": "#1F5FA8", "s2p5": "#7B3FA0", "s3": "#C2185B"}
LS = {"baseline": "-", "e0p8": "--", "e0p6": "-.", "e0p4": ":", "s1p5": "--", "s2": "-.", "s2p5": ":", "s3": (0, (5, 1))}
LV = ["501", "502", "503"]
DEPTH = {"501": "5.8 cm sensor (layer 0-10 cm)", "502": "15.8 cm sensor (layer 10-20 cm)", "503": "25.8 cm sensor (layer 20-30 cm)"}
SYR, MAP = "4e38a472e268a56491b788d5b1271cc3c3da23bb1cc3221a8c59d90f3cc864d2", "fbc504bbcd1ec0c02e6a96089134a06187532455ef7ba5e6f146801c8313c02e"
S0 = 0.20170581117963332
for d in (FIG, TAB, DATA):
    d.mkdir(exist_ok=True)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def csvw(path, header, rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


def J(p):
    return json.loads(Path(p).read_text())


def seal_check():
    seal = J(TASK / "root_run_seal.json")["files"]
    assert len(seal) == 218
    for p, rec in seal.items():
        b = Path(p).read_bytes()
        assert hashlib.sha256(b).hexdigest() == rec["sha256"] and len(b) == rec["bytes"], p
    return seal


def restart_seal_check():
    rs = J(TASK / "root_restart_seal.json")
    assert rs["schema"] == "syrup.jorn_evap_restart.seal/1"
    for p, rec in rs["files"].items():
        b = Path(p).read_bytes()
        assert hashlib.sha256(b).hexdigest() == rec["sha256"] and len(b) == rec["bytes"], p
    assert str(TASK / "root_restart_verification.json") in rs["files"]
    return rs["files"]


def load():
    seal = seal_check()
    S = lambda k, r: TASK / "study" / k / r
    R = {"seal": seal, "audit": J(TASK / "root_result_audit.json"), "per": J(TASK / "root_period_scores.json"), "inp": J(TASK / "root_independent_inputs.json"), "param": J(TASK / "root_parameter_verification.json"),
         "pre": J(TASK / "root_preexecution_acceptance.json")}
    for k in KEYS:
        R[k] = {"res": J(S(k, "actual_numba/result.json")), "cfg": J(S(k, "case_numba.json")), "tm": J(S(k, "actual_numba/timing.json")), "np": J(S(k, "actual_numpy/result.json")), "d900": J(S(k, "actual_numba_dt900/result.json")),
                "wall": J(S(k, "actual_numba_runs/result.json")), "dec": J(S(k, "root_decimal_verification.json")), "src": {v: (J(S(k, f"actual_{v}_runs/sources_before.json")), J(S(k, f"actual_{v}_runs/sources_after.json"))) for v in ("numba", "numpy", "numba_dt900")}}
        for rel in ("actual_numba/result.json", "actual_numba/timing.json", "case_numba.json", "actual_numpy/result.json", "actual_numba_dt900/result.json", "root_decimal_verification.json", "actual_numba_runs/result.json"):
            assert str(S(k, rel)) in seal, (k, rel)
    R["prior_k1"] = J(PRIOR / "study/k1/actual_numba/result.json")
    R["restart"] = J(TASK / "root_restart_verification.json")
    R["rseal"] = restart_seal_check()
    return R


def prep(R):
    r0 = R["baseline"]["res"]["rows"]
    R["obs"] = np.array([[np.nan if v is None else v for v in r["obs_theta"]] for r in r0])
    for k in KEYS:
        rows = R[k]["res"]["rows"]
        R[k]["theta"] = np.array([r["layer_theta"] for r in rows])
        other = np.array([[np.nan if v is None else v for v in r["obs_theta"]] for r in rows])
        assert np.array_equal(np.nan_to_num(other, nan=-1), np.nan_to_num(R["obs"], nan=-1))
        assert R[k]["res"]["plan"]["initial"]["layer_theta"] == R["baseline"]["res"]["plan"]["initial"]["layer_theta"]
    R["theta0"] = np.array(R["baseline"]["res"]["plan"]["initial"]["layer_theta"])
    R["obs0"] = np.array(R["baseline"]["res"]["plan"]["initial"]["point_theta"])
    R["close"] = [dt.date.fromisoformat(r["date"]) + dt.timedelta(days=1) for r in r0]
    R["start"] = dt.date.fromisoformat(r0[0]["date"])
    R["fc"] = np.array(R["baseline"]["cfg"]["layers"]["theta_field_capacity"])
    R["sat"] = np.array(R["baseline"]["cfg"]["layers"]["theta_saturation"])
    R["early"] = np.array([r["date"] < "2024-03-01" for r in r0])
    return R


def stats(e):
    return {"n": int(e.size), "rmse": 100 * math.sqrt(float(np.mean(e**2))), "bias": 100 * float(e.mean())}


def err(R, k, j, kind, mask=None):
    th, ob = R[k]["theta"][:, j], R["obs"][:, j]
    ok = np.isfinite(ob) if mask is None else (np.isfinite(ob) & mask)
    if kind == "raw":
        return th[ok] - ob[ok]
    if kind == "aligned":
        return (th[ok] - R["theta0"][j]) - (ob[ok] - R["obs0"][j])
    return -(ob[ok] - R["obs0"][j])


def scores(R, k):
    out = {}
    for kind in ("raw", "aligned", "persistence"):
        allv = [err(R, k, j, kind) for j in range(3)]
        out[kind] = {v: stats(allv[j]) for j, v in enumerate(LV)} | {"pooled": stats(np.concatenate(allv))}
    return out


def periods(R, k):
    out = {}
    for name, mask in (("before_March1", R["early"]), ("March1_onward", ~R["early"])):
        d = {}
        for kind in ("raw", "aligned"):
            per = [err(R, k, j, kind, mask) for j in range(3)]
            d[kind] = {"501": stats(per[0]), "502": stats(per[1]), "503": stats(per[2]), "upper_two": stats(np.concatenate(per[:2])), "all_three": stats(np.concatenate(per))}
        out[name] = d
    return out


def budget(R, k):
    rows = R[k]["res"]["rows"]
    s = lambda key: 1000 * sum(r[key] for r in rows)
    perc = np.array([r["percolation_m3"] for r in rows]) * 1000
    dif = np.array([r["diffusion_m3"] for r in rows]) * 1000
    th = R[k]["theta"]
    jan = next(i for i, r in enumerate(rows) if r["date"] == "2024-01-23")
    return {"rain": s("rain_m3"), "intake": s("intake_m3"), "runoff": s("runoff_pool_m3") + s("surface_return_m3"), "drain": s("drainage_m3"), "evap": s("evaporation_m3"), "unmet": s("evaporation_unmet_m3"), "final": 1000 * rows[-1]["storage_m3"],
            "grav": perc.sum(0).tolist(), "diff_down": dif.sum(0).tolist(), "peak": (th.max(0) * 100).tolist(), "jan_top": float((th[jan, 0] - th[jan - 1, 0]) * 100), "jan_close_deep": float(th[jan, 2] * 100),
            "final_theta": (th[-1] * 100).tolist(), "th_series": th}


def endpoints(R, k):
    rows = R["baseline"]["res"]["rows"]
    out = []
    for j, v in enumerate(LV):
        i = int(np.flatnonzero(np.isfinite(R["obs"][:, j]))[-1])
        out.append({"level": v, "index": i, "date": rows[i]["date"], "close": rows[i]["nominal_close_utc"], "obs": float(R["obs"][i, j] * 100), "model": float(R[k]["theta"][i, j] * 100)})
    return out


def review(R):
    """Independent recomputation from the daily rows against Root's audit, period scores, controls and plan; every assertion aborts the build."""
    A = {r["key"]: r for r in R["audit"]["records"]}
    P = {r["key"]: r for r in R["per"]["records"]}
    assert [r["key"] for r in R["audit"]["records"]] == KEYS and R["audit"]["status"] == "PASS" and R["per"]["status"] == "PASS" and R["param"]["status"] == "PASS"
    obs_inp = R["inp"]["observations"]
    for j, v in enumerate(LV):
        want = [np.nan if x is None else x for x in obs_inp[v]["day_end_values"]]
        assert np.array_equal(np.nan_to_num(np.array(want, dtype=float), nan=-1), np.nan_to_num(R["obs"][:, j], nan=-1)), v
    assert [int(np.isfinite(R["obs"][:, j]).sum()) for j in range(3)] == [56, 147, 79]
    S, B, PS, EP = {}, {}, {}, {}
    for k in KEYS:
        res, rows = R[k]["res"], R[k]["res"]["rows"]
        S[k], B[k], PS[k], EP[k] = scores(R, k), budget(R, k), periods(R, k), endpoints(R, k)
        rec = A[k]
        for v in LV:
            for kind, rk in (("raw", "raw"), ("aligned", "offset_aligned"), ("persistence", "persistence_raw")):
                assert math.isclose(S[k][kind][v]["rmse"], rec["scores"][v][rk]["rmse_pp"], rel_tol=1e-9, abs_tol=1e-12) and S[k][kind][v]["n"] == rec["scores"][v][rk]["n"]
        assert math.isclose(S[k]["raw"]["pooled"]["rmse"], rec["pooled"]["raw"]["rmse_pp"], rel_tol=1e-9) and math.isclose(S[k]["aligned"]["pooled"]["rmse"], rec["pooled"]["offset_aligned"]["rmse_pp"], rel_tol=1e-9)
        b, bud = B[k], rec["budgets_mm"]
        assert math.isclose(b["evap"], bud["evaporation"], rel_tol=1e-9) and math.isclose(b["drain"], bud["bottom_drainage"], rel_tol=1e-9) and math.isclose(b["final"], bud["final_storage"], rel_tol=1e-9)
        assert all(math.isclose(x, y, rel_tol=1e-9) for x, y in zip(b["grav"], bud["gravity_transfers"])) and all(math.isclose(x, y, rel_tol=1e-9, abs_tol=1e-12) for x, y in zip(b["diff_down"], bud["net_diffusive_transfers_down_positive"]))
        assert abs(22.9424 + b["intake"] - b["evap"] - b["drain"] - b["final"]) < 1e-9 and math.isclose(b["intake"], 25.088, rel_tol=1e-12) and math.isclose(b["rain"], 25.088, rel_tol=1e-12) and b["runoff"] == 0.0
        assert res["complete"] and res["days_done"] == 160 and res["physical_end_t_s"] == 13923000.0 and res["added_rain_hours"] == 27.5 and rows[-1]["nominal_close_utc"] == "2024-06-26T07:00:00Z"
        assert all(r["n_ponded_steps"] == 0 and r["n_onsets"] == 0 and r["room_rejected_m3"] == 0 for r in rows) and res["ledger"]["n_matrix_steps"] == 55 and res["ledger"]["n_soil_steps"] == 215
        # period scores vs Root (nominal-day label before / from 2024-03-01)
        for pname in ("before_March1", "March1_onward"):
            for sub in ("upper_two", "all_three", "501", "502", "503"):
                for kind in ("raw", "aligned"):
                    got, want = PS[k][pname][kind][sub], P[k]["periods"][pname][sub][kind]
                    assert got["n"] == want["n"] and math.isclose(got["rmse"], want["rmse_pp"], rel_tol=1e-9, abs_tol=1e-12) and math.isclose(got["bias"], want["bias_pp"], rel_tol=1e-9, abs_tol=1e-12), (k, pname, sub, kind)
        assert [PS[k]["before_March1"]["raw"]["upper_two"]["n"], PS[k]["March1_onward"]["raw"]["upper_two"]["n"]] == [84, 119]
        for e, w in zip(EP[k], P[k]["endpoints"]):
            assert e["level"] == w["level"] and e["close"] == w["close_utc"] and math.isclose(e["obs"], 100 * w["observed_theta"], abs_tol=1e-9) and math.isclose(e["model"], 100 * w["model_theta"], rel_tol=1e-12)
        assert [e["close"] for e in EP[k]] == ["2024-03-15T07:00:00Z", "2024-06-25T07:00:00Z", "2024-06-26T07:00:00Z"]
        # controls
        assert R[k]["np"]["rows"] == rows and R[k]["np"]["ledger"] == res["ledger"] and all(v == 0.0 for v in rec["backend_max_absolute_difference"].values())
        d9 = np.array([r["layer_theta"] for r in R[k]["d900"]["rows"]])
        dd = (np.abs(d9 - R[k]["theta"]).max(0) * 100).tolist()
        assert all(math.isclose(x, y, rel_tol=1e-6) for x, y in zip(dd, rec["dt900_max_theta_difference_pp_by_layer"]))
        B[k]["d900_pp"] = dd
        for v, (a, c) in R[k]["src"].items():
            for blk in (a, c):
                got = {x["module"]: (x["actual_sha256"], x["actual_files"]) for x in blk}
                assert got == {"maple_syrup": (SYR, 180), "maple": (MAP, 412)}, (k, v, got)
        assert R[k]["dec"]["status"] == "PASS" and len(R[k]["dec"]["cases"]) == 3 and R[k]["wall"]["returncode"] == 0 and res["config_sha256"] == sha(TASK / "study" / k / "case_numba.json")
        B[k]["dec_max_err_m"] = max(c["max_water_error_m"] for c in R[k]["dec"]["cases"])
        B[k]["dec_over_bound"] = max(c["max_error_over_declared_daily_bound"] for c in R[k]["dec"]["cases"])
        cfg = R[k]["cfg"]
        assert cfg["layers"]["theta_evaporation_stress"] == A[k]["stress"] and cfg["physics"]["evaporation_factor"] == A[k]["E"] and math.isclose(A[k]["stress"] / S0, A[k]["stress_multiplier"], rel_tol=1e-12)
        assert max(max(r["layer_theta"]) for r in rows) < 0.1100                      # every layer stays at or below its field capacity at every daily close
    # the baseline reproduces the previously accepted SR33 1xK case exactly
    kp = R["prior_k1"]
    assert kp["rows"] == R["baseline"]["res"]["rows"] and kp["ledger"] == R["baseline"]["res"]["ledger"]
    ref = R["audit"]["reference_SR33_K1_E1"]
    assert math.isclose(ref["pooled_aligned_rmse_pp"], S["baseline"]["aligned"]["pooled"]["rmse"], rel_tol=1e-9) and math.isclose(ref["E_mm"], B["baseline"]["evap"], rel_tol=1e-9)
    # statements the report makes, checked against the data
    for k in KEYS[1:]:
        assert B[k]["evap"] < B["baseline"]["evap"] and B[k]["drain"] > B["baseline"]["drain"]
        assert S[k]["aligned"]["pooled"]["rmse"] > S["baseline"]["aligned"]["pooled"]["rmse"] and S[k]["raw"]["pooled"]["rmse"] > S["baseline"]["raw"]["pooled"]["rmse"]
    for f in (0.8, 0.6, 0.4):
        k = {0.8: "e0p8", 0.6: "e0p6", 0.4: "e0p4"}[f]
        assert B[k]["evap"] / B["baseline"]["evap"] > f                          # actual E falls less than proportionally with the factor (moisture and drainage feedback)
    early = {k: PS[k]["before_March1"] for k in KEYS}
    best_raw = min(KEYS, key=lambda k: early[k]["raw"]["upper_two"]["rmse"])
    best_al = min(KEYS, key=lambda k: early[k]["aligned"]["upper_two"]["rmse"])
    assert best_raw == "e0p8" and best_al == "e0p6" and min(KEYS, key=lambda k: S[k]["aligned"]["pooled"]["rmse"]) == "baseline"
    # equivalence: effective coefficient E/r^2
    for k in KEYS:
        r = A[k]["stress_multiplier"]
        A[k]["eff"] = A[k]["E"] / (r * r)
    assert all(R["fc"][0] < A[k]["stress"] for k in KEYS) and max(A[k]["stress"] for k in KEYS) > R["sat"].max() and A["s2p5"]["stress"] > R["sat"].max() and A["s2"]["stress"] < R["sat"].min()
    rv = R["restart"]
    assert rv["status"] == "PASS" and rv["case"] == "s3" and rv["split_day"] == 80 and rv["fresh_process_resume"] is True and rv["stress_multiplier"] == 3
    assert {"rows", "ledger", "cumulative_closure", "matrix_regime", "scores", "physical_end_t_s", "added_rain_hours", "nominal_elapsed_s", "complete", "days_done"} <= set(rv["exact_equal_fields"]) and "not production checkpoint" in rv["scope"]
    res_r = J(TASK / "study/s3/actual_numba_resumed/result.json")
    part = J(TASK / "study/s3/actual_numba_midpoint/result_partial_day80.json")
    assert res_r["rows"] == R["s3"]["res"]["rows"] and res_r["ledger"] == R["s3"]["res"]["ledger"] and res_r["scores"] == R["s3"]["res"]["scores"] and res_r["complete"] and part["days_done"] == 80 and not part["complete"]
    R["A"], R["P"] = A, P
    return S, B, PS, EP


# ------------------------------------------------------------------------------------------------ figures
def style(ax):
    ax.grid(alpha=0.25)
    ax.tick_params(labelsize=8)


def rain_panel(ax, R, idx):
    rows = R["baseline"]["res"]["rows"]
    ax.bar([R["close"][i] - dt.timedelta(days=0.5) for i in idx], [rows[i]["observed_rain_mm"] for i in idx], width=0.85, color="#1F5FA8")
    ax.set_ylabel("rain\n(mm/day)", fontsize=8)
    ax.set_title("Observed rain per nominal day", fontsize=9, loc="left")
    style(ax)


def fig_traj(R, fam, aligned, name, title, idx=None):
    idx = list(range(len(R["close"]))) if idx is None else list(idx)
    x = [R["close"][i] for i in idx]
    fig, axs = plt.subplots(4, 1, figsize=(7.2, 8.2), sharex=True, gridspec_kw={"height_ratios": [0.75, 1.4, 1.4, 1.4]})
    rain_panel(axs[0], R, idx)
    for j in range(3):
        ax = axs[j + 1]
        for k in FAM[fam]:
            y = R[k]["theta"][idx, j] * 100 - (R["theta0"][j] * 100 if aligned else 0)
            ax.plot(x, y, color=COL[k], ls=LS[k], lw=1.4 if k != "baseline" else 1.6, label=LAB[k])
        o = R["obs"][idx, j] * 100 - (R["obs0"][j] * 100 if aligned else 0)
        ax.scatter(x, o, facecolors="none", edgecolors="black", s=14, lw=0.7, label="sensor point value" + (" (change since day 0)" if aligned else ""))
        ax.axhline(R["fc"][j] * 100 - (R["theta0"][j] * 100 if aligned else 0), color="#B22222", lw=0.9, ls="--", alpha=0.8)
        if aligned:
            ax.axhline(0, color="#777777", lw=0.6)
        ax.set_ylabel("change since day 0\n(percentage points)" if aligned else "water content\n(vol. %)", fontsize=8)
        ax.set_title(DEPTH[LV[j]], fontsize=9, loc="left")
        style(ax)
    h, l = axs[1].get_legend_handles_labels()
    h.append(plt.Line2D([], [], color="#B22222", ls="--", lw=0.9))
    l.append("field capacity (SR33, 33 kPa)")
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=7, frameon=False)
    for ax in axs[1:]:
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0.085, 1, 0.97))
    fig.savefig(FIG / name, dpi=150)
    plt.close(fig)


def fig_scores(S, PS):
    fig, axs = plt.subplots(1, 3, figsize=(7.6, 4.1), gridspec_kw={"width_ratios": [1, 1.35, 1.1]})
    xs = np.arange(len(KEYS))
    lab = [SHORT[k] for k in KEYS]
    ax = axs[0]
    w = 0.38
    ax.bar(xs - w / 2, [S[k]["aligned"]["pooled"]["rmse"] for k in KEYS], w, color=[COL[k] for k in KEYS], label="aligned")
    ax.bar(xs + w / 2, [S[k]["raw"]["pooled"]["rmse"] for k in KEYS], w, color=[COL[k] for k in KEYS], alpha=0.45, edgecolor="black", lw=0.5, label="RAW (lighter)")
    ax.axhline(S["baseline"]["persistence"]["pooled"]["rmse"], color="#444444", ls=":", lw=1)
    ax.set_xticks(xs)
    ax.set_xticklabels(lab, fontsize=6.5, rotation=60, ha="right")
    ax.set_ylabel("pooled RMSE (percentage points)", fontsize=8)
    ax.set_title("(a) all 282 values", fontsize=8.5)
    ax.legend(fontsize=6, frameon=False, loc="upper left")
    style(ax)
    ax = axs[1]
    for i, k in enumerate(KEYS):
        for j, v in enumerate(LV):
            ax.bar(j * 9 + i, S[k]["aligned"][v]["rmse"], 0.85, color=COL[k])
    for j, v in enumerate(LV):
        ax.hlines(S["baseline"]["persistence"][v]["rmse"], j * 9 - 0.5, j * 9 + 7.5, color="black", ls=":", lw=1.3)
    ax.set_xticks([3.5, 12.5, 21.5])
    ax.set_xticklabels(["5.8 cm", "15.8 cm", "25.8 cm"], fontsize=7.5)
    ax.set_ylabel("aligned RMSE (percentage points)", fontsize=8)
    ax.set_title("(b) by depth; dotted = no change", fontsize=8.5)
    style(ax)
    ax = axs[2]
    ax.bar(xs - w / 2, [PS[k]["before_March1"]["raw"]["upper_two"]["rmse"] for k in KEYS], w, color=[COL[k] for k in KEYS], label="before Mar 1 (n = 84)")
    ax.bar(xs + w / 2, [PS[k]["March1_onward"]["raw"]["upper_two"]["rmse"] for k in KEYS], w, color=[COL[k] for k in KEYS], alpha=0.45, edgecolor="black", lw=0.5, label="from Mar 1 (n = 119)")
    ax.set_xticks(xs)
    ax.set_xticklabels(lab, fontsize=6.5, rotation=60, ha="right")
    ax.set_ylabel("upper-two RAW RMSE (percentage points)", fontsize=8)
    ax.set_title("(c) 5.8 + 15.8 cm sensors", fontsize=8.5)
    ax.legend(fontsize=6, frameon=False, loc="upper left")
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "fig5_scores.png", dpi=150)
    plt.close(fig)


def fig_degeneracy(R, B, S):
    fig, axs = plt.subplots(1, 3, figsize=(7.6, 3.4))
    for ax, key, lab_ in ((axs[0], "evap", "actual E (mm)"), (axs[1], "drain", "bottom drainage (mm)"), (axs[2], "rmse", "pooled aligned RMSE (pct. points)")):
        for k in KEYS:
            eff = R["A"][k]["eff"]
            y = S[k]["aligned"]["pooled"]["rmse"] if key == "rmse" else B[k][key]
            ax.scatter(eff, y, color=COL[k], marker="o" if k in ("baseline", "e0p8", "e0p6", "e0p4") else "s", s=34, zorder=3, edgecolors="black", lw=0.5)
            ax.annotate(SHORT[k], (eff, y), textcoords="offset points", xytext=(3, 3 if key != "drain" else -9), fontsize=6)
        ax.set_xlabel("effective coefficient E / r²", fontsize=8)
        ax.set_ylabel(lab_, fontsize=8)
        style(ax)
    axs[1].set_title("circles: E family, squares: stress family", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig6_common_coefficient.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ tables
def g(x, n=3):
    return f"{x:.{n}f}"


def tables(R, S, B, PS, EP):
    A = R["A"]
    T = {}
    T["params"] = (["Case", "Evaporation factor E", "Stress threshold (vol. %)", "Stress / baseline r", "Effective coefficient E / r²", "Stress vs top-layer saturation (46.43 vol. %)"],
                   [[LAB[k], g(A[k]["E"], 2), g(100 * A[k]["stress"], 2), g(A[k]["stress_multiplier"], 1), g(A[k]["eff"], 4), "above" if A[k]["stress"] > R["sat"][0] else "below"] for k in KEYS])
    pet = R["baseline"]["res"]["plan"]["pet_mm_day"]["sum"]
    for name, keys in (("budget_E", FAM["E"]), ("budget_stress", FAM["stress"])):
        rows = [["Rain = intake (mm)"] + [g(B[k]["intake"]) for k in keys], ["Runoff / ponding (mm)"] + ["0"] * len(keys),
                ["Potential soil demand E x PET (mm)"] + [g(A[k]["E"] * pet, 1) for k in keys], ["Actual evaporation (mm)"] + [g(B[k]["evap"], 2) for k in keys],
                ["Actual E / baseline actual E"] + [g(B[k]["evap"] / B["baseline"]["evap"], 3) for k in keys], ["Bottom drainage (mm)"] + [g(B[k]["drain"], 3) for k in keys],
                ["Final storage (mm; initial 22.9424)"] + [g(B[k]["final"], 2) for k in keys],
                ["Water crossing the upper interface (0-10 to 10-20 cm) (mm)"] + [g(B[k]["grav"][0], 2) for k in keys], ["Water crossing the lower interface (10-20 to 20-30 cm) (mm)"] + [g(B[k]["grav"][1], 2) for k in keys],
                ["Net upward diffusive exchange, top-to-middle (mm)"] + [g(-B[k]["diff_down"][0], 2) for k in keys]]
        T[name] = (["Quantity"] + [LAB[k] for k in keys], rows)
    cols = ["5.8 cm (n = 56)", "15.8 cm (n = 147)", "25.8 cm (n = 79)", "All pooled (n = 282)"]
    vs = LV + ["pooled"]
    for kind, nm in (("aligned", "scores_aligned"), ("raw", "scores_raw")):
        rows = [[LAB[k]] + [g(S[k][kind][v]["rmse"]) for v in vs] for k in KEYS] + [["No change (day-0 sensor value held)"] + [g(S["baseline"]["persistence"][v]["rmse"]) for v in vs]]
        T[nm] = (["Setting"] + cols, rows)
    pc = lambda k, pn, kind, sub: PS[k][pn][kind][sub]
    T["period_top"] = (["Setting", "Before Mar 1 RAW", "Before Mar 1 aligned", "From Mar 1 RAW", "From Mar 1 aligned"],
                       [[LAB[k]] + [g(pc(k, pn, kind, "upper_two")["rmse"]) for pn in ("before_March1", "March1_onward") for kind in ("raw", "aligned")] for k in KEYS])
    T["period_all"] = (["Setting", "Before Mar 1 RAW", "Before Mar 1 aligned", "From Mar 1 RAW", "From Mar 1 aligned"],
                       [[LAB[k]] + [g(pc(k, pn, kind, "all_three")["rmse"]) for pn in ("before_March1", "March1_onward") for kind in ("raw", "aligned")] for k in KEYS])
    T["period_bias"] = (["RAW bias (model minus sensor, percentage points)", "Upper two, before Mar 1", "Upper two, from Mar 1", "All three, before Mar 1", "All three, from Mar 1"],
                        [[LAB[k]] + [f"{pc(k, pn, 'raw', sub)['bias']:+.2f}" for sub in ("upper_two", "all_three") for pn in ("before_March1", "March1_onward")] for k in KEYS])
    e0 = EP["baseline"]
    T["endpoint"] = (["Setting (vol. %)"] + [f"{DEPTH[e['level']].split(' (')[0]}, {e['close'][:10]}" for e in e0],
                     [["Observed (last valid sensor value)"] + [g(e["obs"], 2) for e in e0]] + [[LAB[k]] + [g(e["model"], 2) for e in EP[k]] for k in KEYS])
    rows, rows2 = [], []
    for k in KEYS:
        tm = R[k]["tm"]
        cl = R[k]["res"]["cumulative_closure"]
        rows.append([LAB[k], g(tm["integration_s"], 4), g(tm["operator_prepare_s"], 2), g(tm["source_checks_and_input_preparation_s"], 2), g(R[k]["wall"]["seconds"], 2)])
        rows2.append([LAB[k], f"{abs(cl['soil_residual_m3']):.1e} / {cl['soil_bound_m3']:.2e}", f"{max(B[k]['d900_pp']):.5f}", f"{B[k]['dec_max_err_m']:.1e}", "0 (exact)"])
    T["timing"] = (["Run", "Integration (s)", "Operator prep + JIT (s)", "Source checks + input prep (s)", "Full fresh command (s)"], rows)
    T["checks"] = (["Run", "Closure residual / native bound (m³)", "900 s step: max change (percentage points)", "90-digit check: max error (m)", "NumPy vs Numba max difference"], rows2)
    return T
def shade(cell, color):
    tcPr = cell._tc.get_or_add_tcPr()
    sh = OxmlElement("w:shd")
    sh.set(qn("w:val"), "clear")
    sh.set(qn("w:color"), "auto")
    sh.set(qn("w:fill"), color)
    tcPr.append(sh)


def add_field(par, instr):
    r = par.add_run()
    for t, txt in (("begin", None), (None, instr), ("end", None)):
        if t:
            e = OxmlElement("w:fldChar")
            e.set(qn("w:fldCharType"), t)
        else:
            e = OxmlElement("w:instrText")
            e.set(qn("xml:space"), "preserve")
            e.text = txt
        r._r.append(e)


def runs(par, text):
    for i, part in enumerate(re.split(r"\*\*(.+?)\*\*", text)):
        if part:
            par.add_run(part).bold = bool(i % 2)


def setup_styles(doc):
    st = doc.styles
    n = st["Normal"]
    n.font.name, n.font.size = "Calibri", Pt(10.5)
    n.paragraph_format.space_after = Pt(6)
    n.paragraph_format.line_spacing = 1.08
    h = st["Heading 1"]
    h.font.name, h.font.size, h.font.bold = "Calibri", Pt(14), True
    h.font.color.rgb = RGBColor.from_string("1F3A5F")
    h.paragraph_format.space_before, h.paragraph_format.space_after = Pt(10), Pt(4)
    h.paragraph_format.keep_with_next = True
    for nm, base in (("RepTitle", "Title"), ("RepSubtitle", "Subtitle"), ("RepCaption", "Normal"), ("RepTable", "Normal")):
        s = st.add_style(nm, WD_STYLE_TYPE.PARAGRAPH)
        s.base_style = st[base]
    t = st["RepTitle"]
    t.font.name, t.font.size, t.font.bold = "Calibri", Pt(19), True
    t.font.color.rgb = RGBColor.from_string("1F3A5F")
    st["RepSubtitle"].font.size, st["RepSubtitle"].font.italic = Pt(10.5), True
    c = st["RepCaption"]
    c.font.size, c.font.italic = Pt(9), True
    c.paragraph_format.space_after = Pt(8)
    c.paragraph_format.keep_together = True
    tt = st["RepTable"]
    tt.font.size = Pt(9)
    tt.paragraph_format.space_after = Pt(0)
    tt.paragraph_format.line_spacing = 1.0


def add_table(doc, hd, rows, width_in, caption):
    assert len(hd) <= 7, hd
    cp = doc.add_paragraph(style="RepCaption")
    runs(cp, caption)
    cp.paragraph_format.keep_with_next = True
    cp.paragraph_format.space_after = Pt(3)
    t = doc.add_table(rows=1, cols=len(hd))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    lay = OxmlElement("w:tblLayout")
    lay.set(qn("w:type"), "fixed")
    t._tbl.tblPr.append(lay)
    wts = [min(max(len(str(h)) * 0.28, max(len(str(r[i])) for r in rows) * 0.8, 6), 26) for i, h in enumerate(hd)]
    if len(hd) == 2:
        wts = [1.0, 2.6]
    tw = [round(width_in * 1440 * w / sum(wts)) for w in wts]
    ws = [Twips(x) for x in tw]
    for gc, x in zip(t._tbl.tblGrid.findall(qn("w:gridCol")), tw):
        gc.set(qn("w:w"), str(x))
    for i, h in enumerate(hd):
        c = t.rows[0].cells[i]
        c.text = ""
        c.paragraphs[0].style = "RepTable"
        c.paragraphs[0].add_run(str(h)).bold = True
        shade(c, "D9E2F3")
        c.width = ws[i]
    th = OxmlElement("w:tblHeader")
    th.set(qn("w:val"), "true")
    t.rows[0]._tr.get_or_add_trPr().append(th)
    t.rows[0]._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
    for ri, row in enumerate(rows):
        cells = t.add_row().cells
        t.rows[-1]._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
        for i, v in enumerate(row):
            cells[i].text = ""
            cells[i].paragraphs[0].style = "RepTable"
            cells[i].paragraphs[0].add_run(str(v))
            cells[i].width = ws[i]
            if ri % 2:
                shade(cells[i], "F3F6FA")
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


# ------------------------------------------------------------------------------------------------ text


# ------------------------------------------------------------------------------------------------ text
REFS = [(1, "Tietjen, B., Zehe, E., & Jeltsch, F. (2009). Simulating plant water availability in dry lands under climate change: a generic model of two soil layers. Water Resources Research 45, W01418.", "https://doi.org/10.1029/2007WR006589"),
        (2, "Saxton, K. E., & Rawls, W. J. (2006). Soil water characteristic estimates by texture and organic matter for hydrologic solutions. Soil Science Society of America Journal 70, 1569-1578.", "https://doi.org/10.2136/sssaj2005.0117"),
        (3, "Hargreaves, G. H., & Samani, Z. A. (1985). Reference crop evapotranspiration from temperature. Applied Engineering in Agriculture 1, 96-99.", "https://doi.org/10.13031/2013.26773"),
        (4, "NEON (RELEASE-2026). Soil water content and water salinity, DP1.00094.001 (Jornada, JORN, plot 004, sensors 501-503).", "https://data.neonscience.org/data-products/DP1.00094.001/RELEASE-2026"),
        (5, "NEON (RELEASE-2026). Precipitation, DP1.00045.001.", "https://data.neonscience.org/data-products/DP1.00045.001/RELEASE-2026"),
        (6, "NEON (RELEASE-2026). Single aspirated air temperature, DP1.00002.001 (air temperature only; no soil temperature is used by the evaporation demand).", "https://data.neonscience.org/data-products/DP1.00002.001/RELEASE-2026"),
        (7, "NEON (RELEASE-2026). Soil physical and chemical properties, Megapit, DP1.00096.001 (Jornada, 2013-03 pit 1): texture, organic carbon and bulk density behind the layer properties. A spatial proxy for the 2024 plot 004 sensors, not demonstrably co-located.", "https://data.neonscience.org/data-products/DP1.00096.001/RELEASE-2026"),
        (8, "SYRUP project. Joint fit of the diffusion multiplier d and the evaporation-stress threshold at E = 1 (report of 2026-11-18/19), docs/subsurface_hydrology/joint_d_stress_calibration/report.md; and the SR33 field-capacity / all-K/Ks study of 2026-11-20 (SYRUP_JORN_160day_SR33_K_sensitivity_20261120.docx).", "")]


def blocks(R, S, B, PS, EP):
    A = R["A"]
    a = lambda k, kind="aligned", v="pooled": S[k][kind][v]["rmse"]
    top = lambda k, pn, kind: PS[k][pn][kind]["upper_two"]["rmse"]
    bias = lambda k, pn, sub="upper_two": PS[k][pn]["raw"][sub]["bias"]
    rows = R["baseline"]["res"]["rows"]
    jan = next(i for i, r in enumerate(rows) if r["date"] == "2024-01-23")
    ob = R["obs"]
    assert np.isnan(ob[jan, 2])
    i_dp = int(next(i for i in range(jan + 1, len(rows)) if np.isfinite(ob[i, 2])))
    dp_close, dp_obs = rows[i_dp]["nominal_close_utc"], float(ob[i_dp, 2] * 100)
    dp_models = [float(R[k]["theta"][i_dp, 2] * 100) for k in KEYS]
    dmax = float(np.nanmax(ob[:, 2]) * 100)
    assert max(dp_models) - min(dp_models) < 0.01 and max(b_["jan_close_deep"] for b_ in B.values()) - min(b_["jan_close_deep"] for b_ in B.values()) < 0.001
    dp_txt = f"about {float(np.mean(dp_models)):.2f} vol. %"
    off = R["theta0"] - R["obs0"]
    fc = R["fc"] * 100
    pet = R["baseline"]["res"]["plan"]["pet_mm_day"]["sum"]
    ordered = sorted(KEYS, key=lambda k: -A[k]["eff"])
    assert all(a(x) < a(y) for x, y in itertools.pairwise(ordered)) and all(B[x]["evap"] > B[y]["evap"] for x, y in itertools.pairwise(ordered))
    best_depth = {v: min(KEYS, key=lambda k: a(k, "aligned", v)) for v in LV}
    assert best_depth["501"] == "e0p8" and best_depth["502"] == "baseline" and best_depth["503"] == "baseline"
    ps = {v: S["baseline"]["persistence"][v]["rmse"] for v in LV}
    return [
        ("title", "Evaporation factor and evaporation stress at the SR33 1xK baseline: JORN, January-June 2024"),
        ("sub", "Two sweeps of the frozen 160-day JORN column: evaporation factor E = 1, 0.8, 0.6, 0.4 at the baseline stress, and E = 1 with the stress threshold multiplied by 1, 1.5, 2, 2.5, 3. Eight unique cases (the baseline is shared). A sensitivity of the native SYRUP column, not a recalibration."),
        ("h1", "1. Summary"),
        ("bul", [f"**The baseline has the lowest full-period, all-depth pooled error, but not at every depth or period.** Pooled aligned RMSE over all 160 days and 282 sensor values (percentage points) is {a('baseline'):.3f} at the baseline and {a('e0p8'):.3f}, {a('e0p6'):.3f}, {a('e0p4'):.3f} for E 0.8, 0.6, 0.4 and {a('s1p5'):.3f}, {a('s2'):.3f}, {a('s2p5'):.3f}, {a('s3'):.3f} for stress x1.5, x2, x2.5, x3 (no change: {S['baseline']['persistence']['pooled']['rmse']:.3f}). The baseline's stress and d were jointly fitted at E = 1 with no transpiration; lowering E or raising the stress moves the pooled fit away from it, although the shallowest sensor (aligned {a('e0p8', 'aligned', '501'):.3f} at E 0.8 against {a('baseline', 'aligned', '501'):.3f}) and the two shallower sensors before March 1 (below) are better at some settings. The lowest full-period error is not evidence that E = 1 is the best physical value: the model has no transpiration and the data after March 1 may include it.",
                 f"**The two families are one knob.** Every top-layer water content at the evaporation step is at or below field capacity ({fc[0]:.2f} vol. %), which is below every stress value, so the quadratic branch (water content / stress)² always applies and a stress multiplier r acts exactly like an evaporation factor E / r². Ordered by that effective coefficient from {A[ordered[0]]['eff']:.0f} (the baseline) down to {A[ordered[-1]]['eff']:.3f} (stress x3), pooled error rises and actual evaporation falls monotonically across all eight cases (Figure 6). The sweeps cannot identify E and the stress separately.",
                 f"**Less evaporation does not give proportionally less actual evaporation, and the retained water leaves through the bottom.** Actual E is {B['baseline']['evap']:.2f} mm at the baseline, {B['e0p8']['evap']:.2f}, {B['e0p6']['evap']:.2f}, {B['e0p4']['evap']:.2f} mm for E 0.8, 0.6, 0.4 (ratios {B['e0p8']['evap'] / B['baseline']['evap']:.3f}, {B['e0p6']['evap'] / B['baseline']['evap']:.3f}, {B['e0p4']['evap'] / B['baseline']['evap']:.3f}, all above the factor) because a wetter top layer evaporates more. Final storage rises ({B['baseline']['final']:.2f} to {B['e0p4']['final']:.2f} mm at E 0.4) but bottom drainage also grows from {B['baseline']['drain']:.2f} to {B['e0p4']['drain']:.2f} mm (to {B['s3']['drain']:.2f} mm at stress x3). All {B['baseline']['intake']:.3f} mm of rain still infiltrates in every case, with no runoff.",
                 f"**Before and after March 1 the answer differs.** For the two shallower sensors before March 1, the lowest RAW RMSE is E 0.8 ({top('e0p8', 'before_March1', 'raw'):.3f} against {top('baseline', 'before_March1', 'raw'):.3f}) and the lowest aligned RMSE is E 0.6 ({top('e0p6', 'before_March1', 'aligned'):.3f} against {top('baseline', 'before_March1', 'aligned'):.3f}); from March 1 the baseline is best and every reduction in evaporation makes the model wetter than the sensors. These are different choices from the full-period result and are diagnostics, not a recommendation.",
                 f"**The deep-layer discrepancy remains in every case.** The model 20-30 cm layer fills to about its field capacity ({fc[2]:.2f} vol. %) at the January 23 storm in all eight cases, whereas the 25.8 cm sensor never exceeds {dmax:.2f} vol. %; its first value after that storm, at {dp_close[:10]}, is {dp_obs:.2f} against {dp_txt} in the model. The cause is unresolved; neither a sensor fault nor a data filter is assumed."]),
        ("h1", "2. What was run"),
        ("p", f"The base is the accepted SR33 1xK run of the previous study [8]: one 1 m² column of three 10 cm layers with an open bottom, SR33 (33 kPa) field capacities {fc[0]:.2f}, {fc[1]:.2f} and {fc[2]:.2f} vol. % [2], all conductivities at 1x, d = 0.0010239461376024856, Tietjen-type exchange [1], observed NEON rain of 25.088 mm [5], Hargreaves-Samani potential evaporation (777.3 mm over the period) from NEON air temperature [6] [3], initial water 22.9424 mm from the NEON soil sensors [4], matrix-only infiltration, daily hydrology, and no transpiration, growth, plants, shading, preferential flow or routing (surface water is exported with zero residence). Evaporation and transpiration are off during rain, and no new sub-daily dry-day step is added. The clock is the earlier daily-weather approximation: 160 nominal days plus 27.5 added storm hours, ending at the 2024-06-26 07:00 UTC close. The baseline rows and ledger reproduce the accepted SR33 1xK case exactly."),
        ("p", "Each case file differs from the base only in the requested evaporation factor or stress threshold, its name and an explicit provenance record. Layer properties come from the 2013 Jornada megapit [7], a spatial proxy for the 2024 plot 004 sensors, not demonstrably co-located. Pooled scores use 282 sensor values (56, 147 and 79 at 5.8, 15.8 and 25.8 cm), with gaps left as gaps."),
        ("tab", "params", "Table 1. The eight cases. The stress values at x2.5 and x3 exceed saturation: they are hypothetical normalisation scales, accepted without changing capacity, saturation, wilting point or field capacity."),
        ("h1", "3. What the stress threshold is"),
        ("p", "In the model, the demand for soil evaporation is the potential rate times E. While the top layer holds at least the stress water content, the full demand is allowed; below it the demand is multiplied by (water content / stress)², so evaporation falls with the square of the dryness ratio, and a separate residual content (1 vol. %) stops any withdrawal. The stress threshold is therefore a proxy for the rising resistance of drying soil to evaporation. It is not a measured soil property, not a retention point (suction, field capacity or saturation) and not a separate water store."),
        ("p", f"**Where the value came from.** Tietjen et al. [1] (section 2.3) define a threshold water content at which stomatal closure begins for combined evapotranspiration, associated in their Table 2 with -1500 kPa. The later EcoHyD code used the wilting-point water content in the same quadratic form, and its Tabernas input of 0.18 is a case value, not a universal one. SYRUP uses the form for soil evaporation only, with plant transpiration requested separately by LEAF. The baseline here, {100 * S0:.2f} vol. %, is neither of those: it was fitted jointly with d, at E = 1 and under the earlier 6 kPa field capacity, to the 53 valid sensor values of the rain-free JORN (April 8-18 and April 26-30, 2025) and MOAB (May 22-28, 2025) windows [8], with transpiration set to zero. It is not the Saxton-Rawls wilting point (3.95-4.63 vol. %) [2]."),
        ("p", "**What that does and does not mean.** A rain-free window is not proof of a transpiration-free window: if plants were active in April-May, the fitted d and stress could absorb the missing transpiration, because the fit is to total observed soil drying and not to independently measured bare-soil evaporation. This report does not claim that transpiration was measured, or that plants were active, only that the conditional inference is possible; the user's assumed onset of transpiration in March is context for reading the late-period bias; it is an assumption, no phenology was measured and none is implemented. No transpiration is switched on in these runs."),
        ("p", f"**Why two sweeps give one answer.** At the evaporation step the top layer never holds more than its field capacity ({fc[0]:.2f} vol. %), well below even the baseline stress ({100 * S0:.2f} vol. %), because gravity moves any excess down first. The quadratic branch therefore always applies, and demand = E x PET x (water content / (r x baseline stress))² = (E / r²) x PET x (water content / baseline stress)². The effective coefficients are 0.444 (x1.5), 0.25 (x2), 0.16 (x2.5) and 0.111 (x3). The native step test of the setup stage confirmed that a stress multiplier r and an evaporation factor E / r² give the same actual evaporation and the same layer water, with an exact water budget and the same limit by available water. A stress above saturation, 50.4 and 60.5 vol. % here, therefore does not mean more stored water and does not change the drying law: at a given water content the demand is lowered by the constant factor 1 / r² while the (water content)² dependence is unchanged, so only the amplitude changes."),
        ("h1", "4. Water: what lower evaporation does"),
        ("tab", "budget_E", "Table 2a. E family. The two interface rows are separate internal pathway counts and can include the same water: do not add them to each other or to losses and storage."),
        ("tab", "budget_stress", "Table 2b. Stress family. Same quantities and the same caution about the interface rows."),
        ("p", f"The baseline takes up all the rain and loses {B['baseline']['evap']:.1f} mm to evaporation and {B['baseline']['drain']:.2f} mm through the bottom. With less evaporative demand the profile keeps more water between storms, so more of the later rain exceeds field capacity in these runs and moves down: the upper-interface count rises from {B['baseline']['grav'][0]:.2f} to {B['e0p4']['grav'][0]:.2f} mm at E 0.4 and {B['s3']['grav'][0]:.2f} mm at stress x3, and the bottom drainage from {B['baseline']['drain']:.2f} to {B['e0p4']['drain']:.2f} and {B['s3']['drain']:.2f} mm. Summed over the 160 days, actual evaporation is {B['s3']['evap'] / pet * 100:.1f}% of the summed potential PET at stress x3 and {B['baseline']['evap'] / pet * 100:.1f}% at the baseline (160-day totals, not values at the end date); it does not scale linearly with E because the drier top layer evaporates less per unit demand. More stored water at the end ({B['s3']['final']:.1f} mm at stress x3) is therefore not a sign of a better model: part of it is deep water that the sensors do not show (below)."),
        ("h1", "5. Comparison with the sensors"),
        ("p", f"**RAW** compares model layer means with sensor point values; **aligned** subtracts one constant day-0 offset per depth ({off[0] * 100:+.2f}, {off[1] * 100:+.2f} and {off[2] * 100:+.2f} points at 5.8, 15.8 and 25.8 cm) because a 10 cm mean is not a point reading. The offsets are the same in every case, and model water is never altered by the alignment. Figures 1 and 3 show the E family, Figures 2 and 4 the stress family; the dashed red line is the SR33 field capacity (the stress values, 20-61 vol. %, are far above the plotted range)."),
        ("fig", "fig1_raw_E.png", "Figure 1. RAW water content, E family: baseline and E 0.8, 0.6, 0.4 (colours and dash styles in the legend), sensor point values (black circles; gaps are not interpolated) and the field capacity of each layer (dashed red). The top panel shows observed rain."),
        ("fig", "fig2_raw_stress.png", "Figure 2. RAW water content, stress family: baseline and stress x1.5, x2, x2.5, x3. Same layout as Figure 1."),
        ("tab", "scores_aligned", "Table 3a. ALIGNED RMSE in percentage points (100 x m³/m³) by depth (n = number of sensor values) and pooled. No change = holding the day-0 sensor value."),
        ("tab", "scores_raw", "Table 3b. RAW RMSE in percentage points, same rows and columns."),
        ("fig", "fig3_aligned_E.png", "Figure 3. The E family as changes since day 0 (the aligned view). The dashed red line is the field capacity minus the layer's day-0 content."),
        ("fig", "fig4_aligned_stress.png", "Figure 4. The stress family as changes since day 0."),
        ("p", f"By depth, the shallowest sensor is best at E 0.8 ({a('e0p8', 'aligned', '501'):.3f} against {a('baseline', 'aligned', '501'):.3f} at the baseline and {ps['501']:.3f} for no change), whereas the middle and deep sensors are best at the baseline ({a('baseline', 'aligned', '502'):.3f} against {ps['502']:.3f} no change; {a('baseline', 'aligned', '503'):.3f} against {ps['503']:.3f}). The middle sensor dominates the pooled score (n = 147), and the deep layer is worse than no change in every case. The final table below pairs each sensor's last valid value with the model at that same close; the shallow sensor has no value after the March 15 close, and the middle sensor none at the June 26 close."),
        ("fig", "fig5_scores.png", "Figure 5. (a) Pooled RMSE, dark = aligned, lighter = RAW; the dotted line is the pooled no-change value. (b) Aligned RMSE by depth; dotted lines are the per-depth no-change values. (c) RAW RMSE for the 5.8 and 15.8 cm sensors before March 1 and from March 1. Categories are equally spaced cases, not a continuous fit."),
        ("fig", "fig6_common_coefficient.png", "Figure 6. Actual evaporation, bottom drainage and pooled aligned RMSE against the common effective coefficient E / r². The two families fall on one curve."),
        ("h1", "6. Before and after March 1"),
        ("p", f"Periods use the nominal local-day label (before 2024-03-01 against from 2024-03-01); each observation is at the daily close that follows its nominal day. No new calendar or transpiration rule is introduced. For the two shallower sensors the counts are 84 before and 119 from March 1; for all three sensors, 104 and 178. Before March 1 the best RAW case for those sensors is E 0.8 ({top('e0p8', 'before_March1', 'raw'):.3f}) and the best aligned case E 0.6 ({top('e0p6', 'before_March1', 'aligned'):.3f}), against {top('baseline', 'before_March1', 'raw'):.3f} and {top('baseline', 'before_March1', 'aligned'):.3f} at the baseline: different cases for different scorings, and neither equals the best full-period case. From March 1 the baseline is best ({top('baseline', 'March1_onward', 'raw'):.3f} RAW) and every reduction raises the error (E 0.4: {top('e0p4', 'March1_onward', 'raw'):.3f}; stress x3: {top('s3', 'March1_onward', 'raw'):.3f}) because the model, which has no transpiration, is already wetter than the sensors (RAW bias {bias('baseline', 'March1_onward'):+.2f} points for the upper two) and less evaporation widens that."),
        ("tab", "period_top", "Table 4a. RMSE (percentage points) for the 5.8 and 15.8 cm sensors, before March 1 (n = 84) and from March 1 (n = 119)."),
        ("tab", "period_all", "Table 4b. RMSE (percentage points) for all three sensors, before March 1 (n = 104) and from March 1 (n = 178)."),
        ("tab", "period_bias", "Table 4c. RAW bias, model minus sensor (percentage points), by period and sensor group."),
        ("p", "A higher late no-transpiration soil water can reflect the transpiration the model lacks; a smaller error from lower evaporation before March 1 could equally reflect an over-strong early evaporation or the shallow sensors' support. These data cannot separate them, so no setting is declared best and no parameter is adopted."),
        ("h1", "7. Same-time endpoint comparison and the deep layer"),
        ("tab", "endpoint", "Table 5. Last valid sensor value at each depth and every case at the SAME close. The shallow sensor has no value after the 2024-03-15 close and the middle sensor none at the 2024-06-26 close; the model's final inventories (Tables 2a and 2b) are valid model end states and imply no observation."),
        ("p", f"The model 20-30 cm layer is wetted to about its field capacity ({fc[2]:.2f} vol. %) by the January 23 storm in all eight cases (daily close about {B['baseline']['jan_close_deep']:.2f} vol. % in every case), because the SR33 field capacity is low enough for the gravity step to fill it. The 25.8 cm sensor shows no such wetting: its maximum is {dmax:.2f} vol. %, and its first value after the storm ({dp_close[:10]}) is {dp_obs:.2f} against {dp_txt}. A whole 20-30 cm bucket that is wetted should be reflected by a probe at 25.8 cm, so the discrepancy stands. Candidate causes include the transport approximation of the cascade, the 2013 soil proxy and a sensor or support effect; none is established, the sensor is not declared faulty and no value was removed. The January filling is essentially unchanged across the eight cases, so neither the evaporation factor nor the stress repairs it."),
        ("h1", "8. Verification and run time"),
        ("bul", ["Before consuming any result I re-hashed all 218 files of Root's seal and checked them again afterwards. From the daily rows I recomputed the per-depth and pooled RAW, aligned and no-change scores, the period scores and counts, the endpoint values, the water budgets and closure, and compared them with Root's audit and period files; all agree (relative 1e-9).",
                 f"NumPy and Numba are identical in every tracked output for all eight cases; a 900 s storm step changes layer water by at most {max(max(B[k]['d900_pp']) for k in KEYS):.5f} percentage points. Root's independent 90-digit calculation (open gravity cascade, evaporation and pair relaxation, no production step call) agrees to {max(B[k]['dec_max_err_m'] for k in KEYS):.1e} m or better. Source packages SYRUP 180 files (4e38a472...) and MAPLE 412 files (fbc504bb...) are exact before and after every run. The baseline reproduces the accepted SR33 1xK case exactly. The finer 900 s controls refine the storm steps only; dry interevent evaporation remains one daily step, so no time-resolution qualification of dry-day evaporation is claimed.",
                 f"Restart: Root's receipt root_restart_verification.json ({R['restart']['status']}, {R['restart']['utc'][:19].replace('T', ' ')} UTC) splits the stress x3 case at nominal day {R['restart']['split_day']}, resumes it in a fresh process from the driver state and reproduces rows, ledger, cumulative closure, matrix regime, scores, physical end time, added rain hours, nominal elapsed time and completion exactly. It is a driver-level component state reconstruction only, not a native checkpoint or coupled-model acceptance, and no new run was made. Its late files are sealed separately (root_restart_seal.json, {len(R['rseal'])} files; the 218-file seal is unchanged). GPU, scaling and field-validation claims are not made."]),
        ("tab", "timing", "Table 6a. Timing, separated (separate fresh runs on a shared host with other jobs; not a controlled speed comparison)."),
        ("tab", "checks", "Table 6b. Verification per case: closure, 900 s step, 90-digit control and NumPy/Numba comparison."),
        ("p", "Integration takes about 0.04 s in each case; the fresh-process time (about 7 s) is dominated by one-time source verification, input preparation and compilation and says nothing about solver speed. One 1 m² CPU column."),
        ("h1", "9. Limits"),
        ("bul", ["A prescribed sensitivity: no calibration, no best parameter, no default change and no adoption. d and the baseline stress are conditional on E = 1, no transpiration and the old field capacity; E and the stress are not separately identified.",
                 "No transpiration, plants or shading; the observed drying after March 1 may include transpiration that the model lacks. This is a conditional inference, not a measurement or a claim of plant activity.",
                 "The above-field-capacity gravity cascade is not rate-limited by any conductivity, and the deep-layer wetting in January is common to all cases. Persistent Green-Ampt history has no automatic reset or drying recovery; here all rain is unponded and below the matrix K, so it does not control intake.",
                 "Point sensors against 10 cm layer means, a daily clock with no new sub-daily dry-day step, a 2013 soil proxy, and the 030 against 020 air-temperature height inherited from the original run. No significance test."]),
        ("h1", "References"), ("refs", REFS),
        ("h1", "Appendix. Identity (compact)"),
        ("p", "Sources: SYRUP package 180 files, SHA-256 " + SYR + "; MAPLE package 412 files, SHA-256 " + MAP + ". Base case: jorn_sr33_160day_20261120/study/k1/case_numba.json, SHA-256 f024054f5198ec8358c95f7a33683e6228084e6af246f9790cde586d437f3d2a. Case-file hashes: " + "; ".join(f"{SHORT[k]} {sha(TASK / 'study' / k / 'case_numba.json')[:16]}..." for k in KEYS)
         + ". The hash of every sealed file is in tables/provenance_seal.csv and artifact_manifest.json; daily trajectories, scores and period scores of all cases are in the CSV tables. Raw NEON files are not copied.")]
def to_docx(Bk, T, path):
    doc = Document()
    setup_styles(doc)
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Inches(8.5), Inches(11)
    sec.left_margin = sec.right_margin = Inches(0.9)
    sec.top_margin = sec.bottom_margin = Inches(0.8)
    fp = sec.footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    fr = fp.add_run("SYRUP JORN 160-day: SR33 1xK evaporation and stress sensitivity · not a recalibration · page ")
    fr.font.size = Pt(8)
    add_field(fp, "PAGE")
    for b in Bk:
        k = b[0]
        if k == "title":
            doc.add_paragraph(b[1], style="RepTitle")
        elif k == "sub":
            doc.add_paragraph(b[1], style="RepSubtitle")
        elif k == "h1":
            doc.add_heading(b[1], level=1)
        elif k == "p":
            runs(doc.add_paragraph(), b[1])
        elif k in ("bul", "bul2"):
            for x in b[1]:
                runs(doc.add_paragraph(style="List Bullet"), x)
        elif k == "tab":
            hd, rows = T[b[1]]
            add_table(doc, hd, rows, 6.7, b[2])
        elif k == "fig":
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.keep_with_next = True
            p.add_run().add_picture(str(FIG / b[1]), width=Inches(6.0 if b[1] != "fig5_scores.png" else 6.7))
            runs(doc.add_paragraph(style="RepCaption"), b[2])
        elif k == "refs":
            for n, c, u in b[1]:
                p = doc.add_paragraph()
                p.paragraph_format.left_indent = Inches(0.25)
                p.paragraph_format.first_line_indent = Inches(-0.25)
                p.paragraph_format.space_after = Pt(2)
                p.add_run(f"[{n}] ").bold = True
                p.add_run(f"{c} {u}".strip()).font.size = Pt(9)
    doc.core_properties.title = "SYRUP JORN 160-day: SR33 1xK evaporation and stress sensitivity"
    doc.core_properties.author = "SYRUP project"
    doc.save(path)


def verify_docx(path, Bk):
    z = zipfile.ZipFile(path)
    names = z.namelist()
    media = [n for n in names if n.startswith("word/media/")]
    xml = z.read("word/document.xml").decode()
    ntab, nhdr = xml.count("<w:tbl>"), xml.count("<w:tblHeader")
    nfig, ntabs = sum(1 for b in Bk if b[0] == "fig"), sum(1 for b in Bk if b[0] == "tab")
    pagef = "PAGE" in "".join(z.read(n).decode(errors="ignore") for n in names if "footer" in n)
    text = re.sub(r"<[^>]+>", " ", xml)
    assert z.testzip() is None and len(media) == nfig and ntab == ntabs == nhdr and pagef
    assert not re.search(r"TODO|TBD|XXX|placeholder|DRAFT|\bnan\b", text)
    return {"zip_ok": True, "media": len(media), "tables": ntab, "header_rows_repeat": nhdr, "paragraphs": xml.count("<w:p>") + xml.count("<w:p "), "words": len(text.split()), "bytes": Path(path).stat().st_size, "sha256": sha(path)}


def to_markdown(Bk, T):
    out = []
    for b in Bk:
        k = b[0]
        if k == "title":
            out.append(f"# {b[1]}\n")
        elif k == "sub":
            out.append(f"*{b[1]}*\n")
        elif k == "h1":
            out.append("## " + b[1] + "\n")
        elif k == "p":
            out.append(b[1] + "\n")
        elif k in ("bul", "bul2"):
            out.append("\n".join(f"- {x}" for x in b[1]) + "\n")
        elif k == "tab":
            hd, rows = T[b[1]]
            out.append(f"**{b[2]}**\n\n| " + " | ".join(hd) + " |\n|" + "---|" * len(hd) + "\n" + "\n".join("| " + " | ".join(str(c) for c in r) + " |" for r in rows) + "\n")
        elif k == "fig":
            out.append(f"![{b[2]}](figures/{b[1]})\n")
        elif k == "refs":
            out.append("\n".join(f"[{n}] {c} {u}".strip() for n, c, u in b[1]) + "\n")
    return "\n".join(out)




def main():
    R = prep(load())
    S, B, PS, EP = review(R)
    for p in list(TAB.glob("*.csv")) + list(DATA.glob("*")):
        p.unlink()
    fig_traj(R, "E", False, "fig1_raw_E.png", "RAW water content at each nominal daily close: evaporation-factor family")
    fig_traj(R, "stress", False, "fig2_raw_stress.png", "RAW water content at each nominal daily close: stress-threshold family")
    fig_traj(R, "E", True, "fig3_aligned_E.png", "Changes since day 0 (aligned view): evaporation-factor family")
    fig_traj(R, "stress", True, "fig4_aligned_stress.png", "Changes since day 0 (aligned view): stress-threshold family")
    fig_scores(S, PS)
    fig_degeneracy(R, B, S)
    T = tables(R, S, B, PS, EP)
    for name, (hd, rows) in T.items():
        csvw(TAB / f"{name}.csv", hd, rows)
    csvw(TAB / "provenance_seal.csv", ["file", "sha256", "bytes"], [[k, v["sha256"], v["bytes"]] for k, v in R["seal"].items()])
    csvw(TAB / "case_hashes.csv", ["case", "config_sha256_in_result", "case_numba_json_sha256"], [[k, R[k]["res"]["config_sha256"], sha(TASK / "study" / k / "case_numba.json")] for k in KEYS])
    hdr = ["day_index", "date_utc_minus7", "nominal_close_utc", "physical_t_s", "rain_mm", "pet_mm"] + [f"theta_l{j + 1}_{k}" for k in KEYS for j in range(3)] + [f"evap_mm_{k}" for k in KEYS] + ["obs_501", "obs_502", "obs_503"]
    rows = []
    for i, r0 in enumerate(R["baseline"]["res"]["rows"]):
        rows.append([r0["day_index"], r0["date"], r0["nominal_close_utc"], r0["physical_t_s"], 1000 * r0["rain_m3"], r0["pet_mm_day"]] + [R[k]["res"]["rows"][i]["layer_theta"][j] for k in KEYS for j in range(3)]
                    + [1000 * R[k]["res"]["rows"][i]["evaporation_m3"] for k in KEYS] + ["" if v is None else v for v in r0["obs_theta"]])
    csvw(TAB / "daily_trajectories.csv", hdr, rows)
    prow = []
    for k in KEYS:
        for pn in ("before_March1", "March1_onward"):
            for sub in ("upper_two", "all_three", "501", "502", "503"):
                for kind in ("raw", "aligned"):
                    s_ = PS[k][pn][kind][sub]
                    prow.append([k, pn, sub, kind, s_["n"], s_["rmse"], s_["bias"]])
    csvw(TAB / "period_scores_all.csv", ["case", "period", "sensor_group", "scoring", "n", "rmse_pp", "bias_pp"], prow)
    for k in KEYS:
        shutil.copyfile(TASK / "study" / k / "case_numba.json", DATA / f"case_numba_{k}.json")
        shutil.copyfile(TASK / "study" / k / "root_decimal_verification.json", DATA / f"root_decimal_verification_{k}.json")
        shutil.copyfile(TASK / "study" / k / "actual_numba/timing.json", DATA / f"timing_numba_{k}.json")
    for name in ("root_result_audit.json", "root_period_scores.json", "root_independent_inputs.json", "root_parameter_verification.json", "report_pending_notes.md", "root_restart_verification.json", "root_restart_seal.json"):
        shutil.copyfile(TASK / name, DATA / name)
    blk = blocks(R, S, B, PS, EP)
    to_docx(blk, T, DOCX)
    info = verify_docx(DOCX, blk)
    seal_check()                                                    # re-hash after consuming
    restart_seal_check()
    (HERE / "report.md").write_text(to_markdown(blk, T), newline="\n")
    (HERE / "verification.md").write_text("# Verification record\n\nBefore building, all 218 files of `root_run_seal.json` were re-hashed; they were re-hashed again after the document was built. From the daily rows of all eight cases the builder recomputed per-depth and pooled RAW, aligned and no-change RMSE/bias/counts, the "
                                          "before/after-March-1 period scores (nominal day label < / >= 2024-03-01) and counts, the same-timestamp endpoint values, water budgets and closure, the zero runoff/ponding/onset/room-rejection checks, NumPy = Numba row equality, the 900 s maximum differences, "
                                          "the before/after source pins of all 24 runs (SYRUP 4e38a472.../180, MAPLE fbc504bb.../412), Root's 90-digit controls, the sensor values against Root's independent raw-file audit, exact reproduction of the accepted SR33 1xK baseline, "
                                          "and the monotonic ordering of actual E and pooled error with the effective coefficient E/r². Static DOCX checks: zip integrity, one media file per figure, table count equals repeated-header count, footer page field, no placeholder text. "
                                          "No Office renderer was available, so rendered pagination and legibility were not checked (figures were viewed as PNG).\n\nRoot's restart receipt (stress x3, split at nominal day 80, fresh-process resume) was read, its seal (`root_restart_seal.json`, 11 files, separate from the immutable 218-file seal) re-hashed before and after the build, and the resumed rows, ledger and scores compared with the continuous run: exact. It is a driver-level component reconstruction only, not a native checkpoint or coupled acceptance.\n\n```\n" + json.dumps(info, indent=1) + "\n```\n", newline="\n")
    man_files = {}
    for sub in (FIG, TAB, DATA):
        for p in sorted(sub.iterdir()):
            if p.is_file() and p.suffix in (".png", ".csv", ".json", ".md"):
                man_files[f"{sub.name}/{p.name}"] = sha(p)
    man = {"docx": {"file": DOCX.name, **info}, "files": man_files, "builder": sha(Path(__file__)), "report_md": sha(HERE / "report.md"), "verification_md": sha(HERE / "verification.md"), "seal_files_rechecked": len(R["seal"]), "restart_seal_files_rechecked": len(R["rseal"]),
           "excluded": "cache directories and ignored files (.ruff_cache, __pycache__, temp) are never listed"}
    (HERE / "artifact_manifest.json").write_text(json.dumps(man, indent=1) + "\n", newline="\n")
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
