"""Builds the JORN 160-day SR33 field-capacity x all-K/Ks (0.5, 1, 2, 4) report: figures, CSV tables, editable Markdown, verification note, manifest and the Word document.

    PYTHONPATH=<tasks>/site_hydraulics_interface_comparison_20261117/report_tools MPLCONFIGDIR=/tmp/syrup-sr33-report PYTHONDONTWRITEBYTECODE=1 /home/okin/MAPLE/.venv/bin/python -B build_report.py

DOCUMENTS ONLY. Reads Root's sealed results (every sealed file used is re-hashed against `root_run_seal.json` before and after), recomputes the scores/budgets/controls independently from the daily rows, runs no model, fits nothing and uses no network or GPU.
Declared inputs: SR33_TASK (this task folder; default = the folder four levels above this file). Writes only inside its own folder.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
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
TASK = Path(os.environ.get("SR33_TASK", str(HERE.parents[3])))
TASKS = TASK.parent
OLD1 = TASKS / "jorn_160_day_run_20261119"
OLDK = TASKS / "jorn_conductivity_multipliers_20261120"
FIG, TAB, DATA = HERE / "figures", HERE / "tables", HERE / "data"
DOCX = HERE / "SYRUP_JORN_160day_SR33_K_sensitivity_20261120.docx"
KEYS = {0.5: "k0p5", 1: "k1", 2: "k2", 4: "k4"}
NEW = [0.5, 1, 2, 4]
OLDF = [1, 2, 4]                                                      # old (6 kPa-hybrid) field capacity, matched K
OLDDIR = {1: OLD1, 2: OLDK / "x2", 4: OLDK / "x4"}
LV = ["501", "502", "503"]
DEPTH = {"501": "5.8 cm sensor (layer 0-10 cm)", "502": "15.8 cm sensor (layer 10-20 cm)", "503": "25.8 cm sensor (layer 20-30 cm)"}
COL = {0.5: "#6BAED6", 1: "#2E8B57", 2: "#1F5FA8", 4: "#7B3FA0"}
LS = {0.5: "-", 1: "-", 2: "--", 4: "-."}
OLDCOL = "#C0392B"
SYR_NEW, MAP = "4e38a472e268a56491b788d5b1271cc3c3da23bb1cc3221a8c59d90f3cc864d2", "fbc504bbcd1ec0c02e6a96089134a06187532455ef7ba5e6f146801c8313c02e"
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


def nm(f, old=False):
    return f"old FC {f:g}x" if old else f"SR33 {f:g}x"


# ------------------------------------------------------------------------------------------------ sealed inputs
def seal_check():
    seal = J(TASK / "root_run_seal.json")["files"]
    assert len(seal) == 120
    for p, rec in seal.items():
        b = Path(p).read_bytes()
        assert hashlib.sha256(b).hexdigest() == rec["sha256"] and len(b) == rec["bytes"], p
    return seal


def load():
    seal = seal_check()
    S = lambda k, r: TASK / "study" / KEYS[k] / r
    R = {"seal": seal, "audit": J(TASK / "root_result_audit.json"), "inp": J(TASK / "root_independent_inputs.json"), "param": J(TASK / "root_parameter_verification.json"),
         "restart": J(TASK / "root_restart_verification.json"), "plan": J(TASK / "root_run_plan.json")}
    for f in NEW:
        runs = "actual_numba_runs_v2" if f == 0.5 else "actual_numba_runs"
        R[f] = {"res": J(S(f, "actual_numba/result.json")), "cfg": J(S(f, "case_numba.json")), "tm": J(S(f, "actual_numba/timing.json")), "np": J(S(f, "actual_numpy/result.json")), "d900": J(S(f, "actual_numba_dt900/result.json")),
                "wall": J(S(f, runs + "/result.json")), "src_before": J(S(f, runs + "/sources_before.json")), "src_after": J(S(f, runs + "/sources_after.json")), "dec": J(S(f, "root_decimal_verification.json")),
                "rt": J(S(f, "runtime_numba.json")), "runs": runs}
    for f in OLDF:
        d = OLDDIR[f]
        R[("old", f)] = {"res": J(d / "actual_numba/result.json"), "cfg": J(d / "case_numba.json"), "tm": J(d / "actual_numba/timing.json")}
    for p in (*(OLDDIR[f] / n for f in OLDF for n in ("actual_numba/result.json", "actual_numba/timing.json", "case_numba.json")),):
        assert str(p) in seal, p
    return R


def prep(R):
    r1 = R[1]["res"]["rows"]
    R["obs"] = np.array([[np.nan if v is None else v for v in r["obs_theta"]] for r in r1])
    for key in [*NEW, *(("old", f) for f in OLDF)]:
        rows = R[key]["res"]["rows"]
        R[key]["theta"] = np.array([r["layer_theta"] for r in rows])
        other = np.array([[np.nan if v is None else v for v in r["obs_theta"]] for r in rows])
        assert np.array_equal(np.nan_to_num(other, nan=-1), np.nan_to_num(R["obs"], nan=-1))
        assert R[key]["res"]["plan"]["initial"]["layer_theta"] == R[1]["res"]["plan"]["initial"]["layer_theta"]
    R["theta0"] = np.array(R[1]["res"]["plan"]["initial"]["layer_theta"])
    R["obs0"] = np.array(R[1]["res"]["plan"]["initial"]["point_theta"])
    R["close"] = [dt.date.fromisoformat(r["date"]) + dt.timedelta(days=1) for r in r1]
    R["start"] = dt.date.fromisoformat(r1[0]["date"])
    R["fc"] = np.array(R[1]["cfg"]["layers"]["theta_field_capacity"])
    R["fc_old"] = np.array(R[("old", 1)]["cfg"]["layers"]["theta_field_capacity"])
    return R


def stats(e):
    return {"n": int(e.size), "rmse": 100 * math.sqrt(float(np.mean(e**2))), "bias": 100 * float(e.mean())}


def scores(R, key):
    th, ob = R[key]["theta"], R["obs"]
    out, A, Rw, P = {}, [], [], []
    for j, v in enumerate(LV):
        ok = np.isfinite(ob[:, j])
        raw = th[ok, j] - ob[ok, j]
        al = (th[ok, j] - R["theta0"][j]) - (ob[ok, j] - R["obs0"][j])
        ps = -(ob[ok, j] - R["obs0"][j])
        out[v] = {"raw": stats(raw), "aligned": stats(al), "persistence": stats(ps)}
        A.append(al), Rw.append(raw), P.append(ps)
    out["pooled"] = {"raw": stats(np.concatenate(Rw)), "aligned": stats(np.concatenate(A)), "persistence": stats(np.concatenate(P))}
    return out


def budget(R, key):
    rows = R[key]["res"]["rows"]
    s = lambda k: 1000 * sum(r[k] for r in rows)
    perc = np.array([r["percolation_m3"] for r in rows]) * 1000
    dif = np.array([r["diffusion_m3"] for r in rows]) * 1000
    storm = np.array([r["storm_rows"] > 0 for r in rows])
    th = R[key]["theta"]
    jan = next(i for i, r in enumerate(rows) if r["date"] == "2024-01-23")
    pk = int(np.argmax(th[:, 2]))
    return {"rain": s("rain_m3"), "intake": s("intake_m3"), "runoff": s("runoff_pool_m3") + s("surface_return_m3"), "drain": s("drainage_m3"), "evap": s("evaporation_m3"), "final": 1000 * rows[-1]["storage_m3"],
            "grav": perc.sum(0).tolist(), "grav_storm_days": perc[storm].sum(0).tolist(), "n_storm_days": int(storm.sum()), "diff_down": dif.sum(0).tolist(), "peak_close": (th.max(0) * 100).tolist(),
            "peak_close_date": [rows[int(np.argmax(th[:, j]))]["date"] for j in range(3)], "final_theta": (th[-1] * 100).tolist(), "jan_top": float((th[jan, 0] - th[jan - 1, 0]) * 100),
            "jan_mid72": float((th[jan + 2, 1] - th[jan - 1, 1]) * 100), "deep_peak_gain_mm": float(rows[pk]["water_m"][2] * 1000 - R["theta0"][2] * 100), "deep_peak_date": rows[pk]["date"],
            "top_mid_up": float(-dif[:, 0].sum())}


def endpoints(R):
    """Last finite sensor value of each depth, its row/close time, and every model run AT THAT SAME close."""
    rows = R[1]["res"]["rows"]
    out = []
    for j, v in enumerate(LV):
        i = int(np.flatnonzero(np.isfinite(R["obs"][:, j]))[-1])
        out.append({"level": v, "index": i, "date": rows[i]["date"], "close": rows[i]["nominal_close_utc"], "obs": float(R["obs"][i, j] * 100),
                    "model": {k: float(R[k]["theta"][i, j] * 100) for k in [*NEW, ("old", 1)]}})
    return out


def review(R):
    """Independent re-computation from the daily rows of every number the report uses; compares with Root's audit, controls and plan."""
    A = {r["factor"]: r for r in R["audit"]["records"]}
    obs_inp = R["inp"]["observations"]
    for j, v in enumerate(LV):                                         # observations in the rows = Root's raw-CSV audit
        want = [np.nan if x is None else x for x in obs_inp[v]["day_end_values"]]
        assert np.array_equal(np.nan_to_num(np.array(want, dtype=float), nan=-1), np.nan_to_num(R["obs"][:, j], nan=-1)), v
    n = [int(np.isfinite(R["obs"][:, j]).sum()) for j in range(3)]
    assert n == [56, 147, 79] and sum(n) == 282
    S, B = {}, {}
    for key in [*NEW, *(("old", f) for f in OLDF)]:
        S[key], B[key] = scores(R, key), budget(R, key)
    for f in NEW:
        rec, res, rows = A[f], R[f]["res"], R[f]["res"]["rows"]
        for v in LV:
            for kind, rk in (("raw", "raw"), ("aligned", "offset_aligned"), ("persistence", "persistence_raw")):
                assert math.isclose(S[f][v][kind]["rmse"], rec["scores"][v][rk]["rmse_pp"], rel_tol=1e-9, abs_tol=1e-12) and S[f][v][kind]["n"] == rec["scores"][v][rk]["n"]
        assert math.isclose(S[f]["pooled"]["raw"]["rmse"], rec["pooled"]["raw"]["rmse_pp"], rel_tol=1e-9) and math.isclose(S[f]["pooled"]["aligned"]["rmse"], rec["pooled"]["offset_aligned"]["rmse_pp"], rel_tol=1e-9)
        b = B[f]
        bud = rec["budgets_mm"]
        assert math.isclose(b["evap"], bud["evaporation"], rel_tol=1e-9) and math.isclose(b["drain"], bud["bottom_drainage"], rel_tol=1e-9) and math.isclose(b["final"], bud["final_storage"], rel_tol=1e-9)
        assert all(math.isclose(x, y, rel_tol=1e-9) for x, y in zip(b["grav"], bud["gravity_transfers"])) and all(math.isclose(x, y, rel_tol=1e-9, abs_tol=1e-12) for x, y in zip(b["diff_down"], bud["net_diffusive_transfers_down_positive"]))
        assert abs(22.9424 + b["intake"] - b["evap"] - b["drain"] - b["final"]) < 1e-9 and math.isclose(b["intake"], b["rain"], rel_tol=1e-12) and b["runoff"] == 0.0
        assert math.isclose(b["rain"], 25.088, rel_tol=1e-12)
        assert res["complete"] and res["days_done"] == 160 and res["physical_end_t_s"] == 13923000.0 and res["added_rain_hours"] == 27.5 and rows[-1]["nominal_close_utc"] == "2024-06-26T07:00:00Z"
        assert all(r["n_ponded_steps"] == 0 and r["n_onsets"] == 0 and r["runoff_pool_m3"] == 0 and r["surface_return_m3"] == 0 and r["room_rejected_m3"] == 0 for r in rows)
        assert res["ledger"]["n_matrix_steps"] == 55 and res["ledger"]["n_soil_steps"] == 215 and res["matrix_regime"]["n_ponded_steps"] == 0
        # NumPy = Numba (tracked physical outputs) and the 900 s control
        assert R[f]["np"]["rows"] == rows and R[f]["np"]["ledger"] == res["ledger"]
        d9 = np.array([r["layer_theta"] for r in R[f]["d900"]["rows"]])
        dd = (np.abs(d9 - R[f]["theta"]).max(0) * 100).tolist()
        assert all(math.isclose(x, y, rel_tol=1e-6) for x, y in zip(dd, rec["dt900_max_theta_difference_pp_by_layer"])), (dd, rec["dt900_max_theta_difference_pp_by_layer"])
        B[f]["d900_pp"] = dd
        # sources before/after and Root's 90-digit control
        for blk in ("src_before", "src_after"):
            got = {x["module"]: (x["actual_sha256"] if "actual_sha256" in x else x["sha256"], x.get("actual_files", x.get("files"))) for x in R[f][blk]}
            assert got == {"maple_syrup": (SYR_NEW, 180), "maple": (MAP, 412)}, (f, blk, got)
        assert R[f]["dec"]["status"] == "PASS" and len(R[f]["dec"]["cases"]) == 3 and R[f]["wall"]["returncode"] == 0 and R[f]["res"]["config_sha256"] == sha(TASK / "study" / KEYS[f] / "case_numba.json")
        B[f]["dec_max_err_m"] = max(c["max_water_error_m"] for c in R[f]["dec"]["cases"])
        B[f]["dec_over_bound"] = max(c["max_error_over_declared_daily_bound"] for c in R[f]["dec"]["cases"])
        assert max(max(r["layer_theta"]) for r in rows) < 0.1100
    ep = endpoints(R)                                                    # real-data endpoint assertions (same-timestamp pairing; no June 26 shallow/middle observation)
    want = [("2024-03-14", "2024-03-15T07:00:00Z", 5.63, 4.428192760762708), ("2024-06-24", "2024-06-25T07:00:00Z", 5.17, 6.336857779544249), ("2024-06-25", "2024-06-26T07:00:00Z", 4.64, 8.642131054844985)]
    for e, (d, c, o, m) in zip(ep, want):
        assert (e["date"], e["close"]) == (d, c) and math.isclose(e["obs"], o, abs_tol=1e-9) and math.isclose(e["model"][1], m, rel_tol=1e-12), (e, d)
    assert np.isnan(R["obs"][-1, 0]) and np.isnan(R["obs"][-1, 1]) and np.isfinite(R["obs"][-1, 2]) and R[1]["res"]["rows"][-1]["nominal_close_utc"] == "2024-06-26T07:00:00Z"
    assert R["audit"]["status"] == "PASS" and R["restart"]["status"] == "PASS" and R["param"]["status"] == "PASS"
    assert R["plan"]["k0p5"]["numba"]["runner_config"].endswith("actual_numba_retry_config.json") and R[0.5]["runs"] == "actual_numba_runs_v2"
    # old runs: same observations / initial water; stored old config relations
    for f in OLDF:
        assert R[("old", f)]["res"]["complete"] and R[("old", f)]["cfg"]["layers"]["theta_field_capacity"] == R["fc_old"].tolist()
    o = S[("old", 1)]["pooled"]["aligned"]["rmse"]
    assert math.isclose(o, R["audit"]["original_FC1x"]["pooled_aligned_rmse_pp"], rel_tol=1e-9)
    assert all(S[f]["pooled"]["aligned"]["rmse"] > o for f in NEW) and min(NEW, key=lambda f: S[f]["pooled"]["aligned"]["rmse"]) == 4
    assert min(NEW, key=lambda f: S[f]["pooled"]["raw"]["rmse"]) == 4 if S[4]["pooled"]["raw"]["rmse"] < S[2]["pooled"]["raw"]["rmse"] else True
    return S, B


# ------------------------------------------------------------------------------------------------ figures
def style(ax):
    ax.grid(alpha=0.25)
    ax.tick_params(labelsize=8)


def rain_panel(ax, R, idx=None):
    rows = R[1]["res"]["rows"]
    idx = range(len(rows)) if idx is None else idx
    ax.bar([R["close"][i] - dt.timedelta(days=0.5) for i in idx], [rows[i]["observed_rain_mm"] for i in idx], width=0.85, color="#1F5FA8")
    ax.set_ylabel("rain\n(mm/day)", fontsize=8)
    ax.set_title("Observed rain per nominal day", fontsize=9, loc="left")
    style(ax)


def monthaxis(ax):
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))


def model_lines(ax, R, j, x, idx, aligned):
    for f in NEW:
        y = R[f]["theta"][idx, j] * 100 - (R["theta0"][j] * 100 if aligned else 0)
        ax.plot(x, y, color=COL[f], ls=LS[f], lw=1.3, label=nm(f))
    y = R[("old", 1)]["theta"][idx, j] * 100 - (R["theta0"][j] * 100 if aligned else 0)
    ax.plot(x, y, color=OLDCOL, ls=":", lw=1.2, label="old FC (6 kPa hybrid) 1x")


def fig_trajectory(R, name, aligned, idx=None, title=None, zoom=False):
    idx = list(range(len(R["close"]))) if idx is None else list(idx)
    x = [R["close"][i] for i in idx]
    fig, axs = plt.subplots(4, 1, figsize=(7.2, 8.2), sharex=True, gridspec_kw={"height_ratios": [0.75, 1.4, 1.4, 1.4]})
    rain_panel(axs[0], R, idx)
    for j in range(3):
        ax = axs[j + 1]
        model_lines(ax, R, j, x, idx, aligned)
        o = R["obs"][idx, j] * 100 - (R["obs0"][j] * 100 if aligned else 0)
        ax.scatter(x, o, facecolors="none", edgecolors="black", s=14 if not zoom else 22, lw=0.7, label="sensor point value" + (" (change since day 0)" if aligned else ""))
        fc = R["fc"][j] * 100 - (R["theta0"][j] * 100 if aligned else 0)
        ax.axhline(fc, color="#B22222", lw=1.0, ls="--", label="new field capacity (SR33, 33 kPa)")
        if aligned:
            ax.axhline(0, color="#777777", lw=0.6)
        if not aligned and not zoom:
            ax.scatter([R["start"]], [R["theta0"][j] * 100], marker="^", color="#2E8B57", s=34, zorder=5, label="initial model layer mean")
            ax.scatter([R["start"]], [R["obs0"][j] * 100], marker="^", facecolors="none", edgecolors="black", s=34, zorder=5, label="initial sensor value")
        ax.set_ylabel("change since day 0\n(percentage points)" if aligned else "water content\n(vol. %)", fontsize=8)
        ax.set_title(DEPTH[LV[j]], fontsize=9, loc="left")
        style(ax)
    if zoom:
        axs[-1].xaxis.set_major_locator(mdates.DayLocator(interval=2))
        axs[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    else:
        monthaxis(axs[-1])
    h, l = axs[1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=7, frameon=False)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0.085, 1, 0.97))
    fig.savefig(FIG / name, dpi=150)
    plt.close(fig)


def fig_scores(S, B):
    keys = [("old", 1), 0.5, 1, 2, 4, ("old", 2), ("old", 4)]
    lab = ["old 1x", "SR33 0.5x", "SR33 1x", "SR33 2x", "SR33 4x", "old 2x", "old 4x"]
    col = [OLDCOL, COL[0.5], COL[1], COL[2], COL[4], OLDCOL, OLDCOL]
    hat = ["//", "", "", "", "", "//", "//"]
    fig, axs = plt.subplots(1, 3, figsize=(7.6, 4.0), gridspec_kw={"width_ratios": [1, 1.25, 1.1]})
    xs = np.arange(len(keys))
    ax = axs[0]
    w = 0.38
    ax.bar(xs - w / 2, [S[k]["pooled"]["aligned"]["rmse"] for k in keys], w, color=col, hatch=hat, edgecolor="white", label="aligned")
    ax.bar(xs + w / 2, [S[k]["pooled"]["raw"]["rmse"] for k in keys], w, color=col, hatch=hat, edgecolor="black", lw=0.6, alpha=0.45, label="RAW (lighter)")
    ax.axhline(S[1]["pooled"]["persistence"]["rmse"], color="#444444", ls=":", lw=1)
    ax.set_xticks(xs)
    ax.set_xticklabels(lab, fontsize=6.5, rotation=60, ha="right")
    ax.set_ylabel("pooled RMSE (percentage points)", fontsize=8)
    ax.set_title("(a) all 282 values", fontsize=8.5)
    ax.legend(fontsize=6, frameon=False)
    style(ax)
    ax = axs[1]
    sel = [("old", 1), 0.5, 1, 2, 4]
    for i, k in enumerate(sel):
        for j, v in enumerate(LV):
            ax.bar(j * 6 + i, S[k][v]["aligned"]["rmse"], 0.85, color=OLDCOL if isinstance(k, tuple) else COL[k], hatch="//" if isinstance(k, tuple) else "", edgecolor="white")
    for j, v in enumerate(LV):
        ax.hlines(S[1][v]["persistence"]["rmse"], j * 6 - 0.5, j * 6 + 4.5, color="black", ls=":", lw=1.3)
    ax.set_xticks([2, 8, 14])
    ax.set_xticklabels(["5.8 cm", "15.8 cm", "25.8 cm"], fontsize=7.5)
    ax.set_ylabel("aligned RMSE (percentage points)", fontsize=8)
    ax.set_title("(b) by depth; dotted = no change", fontsize=8.5)
    style(ax)
    ax = axs[2]
    series = [("evap", "evaporation", "#E8A33D"), ("g0", "gravity step, upper interface", "#2C7FB8"), ("g1", "gravity step, lower interface", "#7FCDBB"), ("drain", "bottom drainage", "#D95F0E")]
    w = 0.2
    for si, (p, labl, c) in enumerate(series):
        v = [B[k]["evap"] if p == "evap" else (B[k]["grav"][0] if p == "g0" else (B[k]["grav"][1] if p == "g1" else B[k]["drain"])) for k in keys]
        ax.bar(xs + (si - 1.5) * w, v, w, color=c, label=labl)
    ax.set_xticks(xs)
    ax.set_xticklabels(lab, fontsize=6.5, rotation=60, ha="right")
    ax.set_ylabel("mm over 160 days (separate, not additive)", fontsize=8)
    ax.set_title("(c) water pathways", fontsize=8.5)
    ax.set_ylim(0, 52)
    ax.legend(fontsize=6, frameon=False, loc="upper left")
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_scores_and_budgets.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ tables
def g(x, n=3):
    return f"{x:.{n}f}"


def tables(R, S, B):
    T = {}
    new, old = R[1]["cfg"]["provenance"], R[("old", 1)]["cfg"]["provenance"]
    fu = new["field_capacity_update"]
    names = ["0-10 cm (A)", "10-20 cm (A+Bw mix)", "20-30 cm (Bw)"]
    rows_s, rows_r = [], []
    for i in range(3):
        p, q = new["layer_properties"][i], old["layer_properties"][i]
        rows_s.append([names[i], g(100 * fu["primary_texture_fractions_sand_silt_clay"][0][i], 1) + " / " + g(100 * fu["primary_texture_fractions_sand_silt_clay"][1][i], 1) + " / " + g(100 * fu["primary_texture_fractions_sand_silt_clay"][2][i], 1),
                       g(fu["organic_matter_percent"][i], 2), g(p["raw_density_g_cm3"], 3), g(p["effective_density_g_cm3"], 3)])
        rows_r.append([names[i], g(100 * p["theta_saturation"], 2), g(100 * p["theta1500"], 2), g(100 * p["theta_field_capacity"], 2), g(100 * q["theta_field_capacity"], 2), g(p["Ks_mm_h"], 1)])
        assert q["theta_field_capacity"] == R["fc_old"][i]
    T["soil"] = (["Layer", "Sand / silt / clay (%)", "Organic matter (%)", "Raw bulk density (g/cm³)", "Effective bulk density used (g/cm³)"], rows_s)
    T["ret"] = (["Layer", "Saturation (vol. %)", "Wilting point (vol. %)", "NEW field capacity, 33 kPa (vol. %)", "OLD field capacity, 6 kPa hybrid (vol. %)", "Ks, 1x (mm/h)"], rows_r)
    P = lambda f: R[f]["cfg"]
    rows = [[lab] + [g(fn(f), 1) for f in NEW] for lab, fn in (("Matrix K, top layer (direct input)", lambda f: P(f)["matrix"]["conductivity_m_s"] * 3.6e6), ("Interface K, 0-10 / 10-20 cm", lambda f: P(f)["layers"]["interface_conductivity_m_s"][0] * 3.6e6),
                                                                ("Interface K, 10-20 / 20-30 cm", lambda f: P(f)["layers"]["interface_conductivity_m_s"][1] * 3.6e6), ("Ks record, layer 0-10 cm", lambda f: P(f)["provenance"]["layer_properties"][0]["Ks_mm_h"]),
                                                                ("Ks record, layer 10-20 cm", lambda f: P(f)["provenance"]["layer_properties"][1]["Ks_mm_h"]), ("Ks record, layer 20-30 cm", lambda f: P(f)["provenance"]["layer_properties"][2]["Ks_mm_h"]))]
    for rec in R["param"]["rows"]:
        assert math.isclose(rec["matrix_K_mm_h"], P(rec["factor"])["matrix"]["conductivity_m_s"] * 3.6e6, rel_tol=1e-12) and rec["FC"] == R["fc"].tolist()
    T["params"] = (["Conductivity (mm/h)"] + [nm(f) for f in NEW], rows)
    m = R[1]["cfg"]
    T["fixed"] = (["Fixed in every run", "Value"], [
        ["Residual water content / evaporation stress threshold", f"{100 * m['layers']['theta_residual'][0]:.0f} vol. % / {100 * m['layers']['theta_evaporation_stress']:.3f} vol. % (fitted earlier under the OLD field capacity)"],
        ["Tietjen diffusion coefficient d (dimensionless) / evaporation factor", f"{m['physics']['diffusion_coefficient']:.10f} / {m['physics']['evaporation_factor']:g}"],
        ["Matrix suction / local conductivity factor", f"{m['matrix']['suction_m'] * 1000:.0f} mm (unmeasured reference) / {m['matrix']['local_conductivity_factor']:.6f}"],
        ["Layers, bottom, gravity", "three 10 cm layers, 1 m², open bottom, gravity drainage on"],
        ["Initial storage (all runs)", f"{R[1]['res']['plan']['initial']['total_water_m'] * 1000:.4f} mm; layer means {', '.join(g(100 * t, 2) for t in R['theta0'])} vol. %; sensor points {', '.join(g(100 * t, 2) for t in R['obs0'])} vol. %"],
        ["Forcing", "NEON rain 25.088 mm (peak 10.034 mm/h; 26 half-hour rows with rain on 11 days), Hargreaves PET 1.26-9.37 mm/day (sum 777.3 mm), 160 nominal days from 2024-01-18 07:00 UTC"],
        ["Excluded", "transpiration, growth, preferential flow, routing, sediment; surface water exported with zero residence"]])
    order = [("old", 1), 0.5, 1, 2, 4, ("old", 2), ("old", 4)]
    cols = ["5.8 cm (n = 56)", "15.8 cm (n = 147)", "25.8 cm (n = 79)", "All pooled (n = 282)"]
    vs = LV + ["pooled"]
    for kind, title in (("aligned", "scores_aligned"), ("raw", "scores_raw")):
        rows = [[nm(k[1], True) if isinstance(k, tuple) else nm(k)] + [g(S[k][v][kind]["rmse"]) for v in vs] for k in order]
        rows.append(["No change (day-0 sensor value held)"] + [g(S[1][v]["persistence"]["rmse"]) for v in vs])
        assert all(S[k][v][kind]["n"] == n for k in order for v, n in zip(vs, (56, 147, 79, 282)))
        T[title] = (["Setting"] + cols, rows)
    T["endpoint"] = (["Sensor", "Last valid sensor close (UTC)", "Observed (vol. %)"] + [nm(f) + " at that close (vol. %)" for f in NEW], [[DEPTH[e["level"]].split(" (")[0], e["close"].replace("T", " ").replace(":00Z", " UTC"), g(e["obs"], 2)] + [g(e["model"][f], 2) for f in NEW] for e in endpoints(R)])
    T["bias"] = (["RAW bias, model minus sensor (percentage points)"] + cols[:3] + [cols[3]], [[nm(k[1], True) if isinstance(k, tuple) else nm(k)] + [f"{S[k][v]['raw']['bias']:+.2f}" for v in vs] for k in (("old", 1), 0.5, 1, 2, 4)])
    ks = [("old", 1), 0.5, 1, 2, 4]
    rows = [["Rain = intake (mm)"] + [g(B[k]["intake"]) for k in ks], ["Runoff / ponding (mm)"] + ["0"] * 5,
            ["Water crossing the upper interface (0-10 to 10-20 cm) by the gravity step (mm)"] + [g(B[k]["grav"][0], 2) for k in ks], ["Water crossing the lower interface (10-20 to 20-30 cm) by the gravity step (mm)"] + [g(B[k]["grav"][1], 2) for k in ks],
            ["Bottom drainage (mm)"] + [g(B[k]["drain"], 3) for k in ks], ["Net diffusive exchange, top-to-middle, upward (mm)"] + [g(B[k]["top_mid_up"], 2) for k in ks],
            ["Net diffusive exchange, middle-to-deep, upward (mm)"] + [g(-B[k]["diff_down"][1], 2) for k in ks], ["Evaporation (mm)"] + [g(B[k]["evap"], 3) for k in ks], ["Final storage (mm; initial 22.9424)"] + [g(B[k]["final"], 3) for k in ks],
            ["Highest daily-close value, top / middle / deep (vol. %)"] + [" / ".join(g(x, 2) for x in B[k]["peak_close"]) for k in ks], ["Top-layer rise at the Jan 23 storm close (points)"] + [g(B[k]["jan_top"], 2) for k in ks],
            ["Middle layer, 72 h change (points)"] + [g(B[k]["jan_mid72"], 2) for k in ks], ["Deep layer, peak gain (mm)"] + [g(B[k]["deep_peak_gain_mm"], 2) for k in ks],
            ["Model end state 2024-06-26 07:00 UTC, top / middle / deep (vol. %); no observation implied"] + [" / ".join(g(x, 2) for x in B[k]["final_theta"]) for k in ks]]
    T["budget"] = (["Quantity"] + ["old FC 1x", "SR33 0.5x", "SR33 1x", "SR33 2x", "SR33 4x"], rows)
    rows, rows2 = [], []
    for f in OLDF:
        rows.append([f"{f:g}x", g(S[("old", f)]["pooled"]["aligned"]["rmse"]), g(S[f]["pooled"]["aligned"]["rmse"]), g(S[("old", f)]["pooled"]["raw"]["rmse"]), g(S[f]["pooled"]["raw"]["rmse"])])
        rows2.append([f"{f:g}x", g(B[("old", f)]["evap"], 2), g(B[f]["evap"], 2), g(B[f]["drain"], 2) + " (old: 0)", g(B[("old", f)]["final"], 2), g(B[f]["final"], 2)])
    T["matched"] = (["K multiplier", "Aligned RMSE, old FC", "Aligned RMSE, SR33", "RAW RMSE, old FC", "RAW RMSE, SR33"], rows)
    T["matched_budget"] = (["K multiplier", "Evaporation, old FC (mm)", "Evaporation, SR33 (mm)", "Bottom drainage, SR33 (mm)", "Final storage, old FC (mm)", "Final storage, SR33 (mm)"], rows2)
    rows, rows2 = [], []
    for f in NEW:
        tm = R[f]["tm"]
        cl = R[f]["res"]["cumulative_closure"]
        rows.append([nm(f), g(tm["integration_s"], 4), g(tm["operator_prepare_s"], 2), g(tm["source_checks_and_input_preparation_s"], 2), g(R[f]["wall"]["seconds"], 2)])
        rows2.append([nm(f), f"{abs(cl['soil_residual_m3']):.1e} / {cl['soil_bound_m3']:.2e}", f"{max(B[f]['d900_pp']):.5f}", f"{B[f]['dec_max_err_m']:.1e}", "0 (exact)"])
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
REFS = [(1, "Tietjen, B., Zehe, E., & Jeltsch, F. (2009). Simulating plant water availability in dry lands under climate change: a generic model of two soil layers. Water Resources Research 45, W01418.", "https://doi.org/10.1029/2007WR006589"),
        (2, "Saxton, K. E., & Rawls, W. J. (2006). Soil water characteristic estimates by texture and organic matter for hydrologic solutions. Soil Science Society of America Journal 70, 1569-1578.", "https://doi.org/10.2136/sssaj2005.0117"),
        (3, "Hargreaves, G. H., & Samani, Z. A. (1985). Reference crop evapotranspiration from temperature. Applied Engineering in Agriculture 1, 96-99.", "https://doi.org/10.13031/2013.26773"),
        (4, "NEON (RELEASE-2026). Soil water content and water salinity, DP1.00094.001 (Jornada, JORN, plot 004, sensors 501-503).", "https://data.neonscience.org/data-products/DP1.00094.001/RELEASE-2026"),
        (5, "NEON (RELEASE-2026). Precipitation, DP1.00045.001.", "https://data.neonscience.org/data-products/DP1.00045.001/RELEASE-2026"),
        (6, "NEON (RELEASE-2026). Single aspirated air temperature, DP1.00002.001.", "https://data.neonscience.org/data-products/DP1.00002.001/RELEASE-2026"),
        (8, "NEON (RELEASE-2026). Soil physical and chemical properties, Megapit, DP1.00096.001 (Jornada, 2013-03 pit 1: perbiogeosample, perhorizon and perbulksample tables; primary texture, organic carbon and bulk density as used in the field-capacity audit of 2026-11-20).", "https://data.neonscience.org/data-products/DP1.00096.001/RELEASE-2026"),
        (7, "Earlier reports of this study, by file name only: SYRUP_JORN_160day_rainfall_comparison_20261119.docx (original run, old field capacity), SYRUP_JORN_160day_K_Ks_multipliers_2x_3x_4x_20261120.docx and SYRUP_JORN_field_capacity_worked_audit_20261120.docx.", "")]


def blocks(R, S, B):
    P = lambda k, kind="aligned": S[k]["pooled"][kind]["rmse"]
    old1 = ("old", 1)
    rows = R[1]["res"]["rows"]
    jan = next(i for i, r in enumerate(rows) if r["date"] == "2024-01-23")
    ob = R["obs"]
    o_top, o_mid72 = (ob[jan, 0] - ob[jan - 1, 0]) * 100, (ob[jan + 2, 1] - ob[jan - 1, 1]) * 100
    ep = endpoints(R)
    o_max_deep = float(np.nanmax(ob[:, 2]) * 100)
    assert abs(o_top - 3.06) < 1e-9 and abs(o_mid72 - 1.06) < 1e-9 and np.isnan(ob[jan, 2])
    off = R["theta0"] - R["obs0"]
    fc = R["fc"] * 100
    ps = [S[1][v]["persistence"]["rmse"] for v in LV]
    a = lambda k, v: S[k][v]["aligned"]["rmse"]
    assert a(4, "501") < ps[0] and all(a(f, "501") > ps[0] for f in (0.5, 1, 2)) and a(4, "502") < ps[1] and a(old1, "501") > ps[0] and all(a(f, "503") > ps[2] for f in NEW)
    assert P(4) < P(1, "persistence") and P(1) > P(1, "persistence") and P(2) < P(1, "persistence")
    d_hi = lambda f: B[f]["peak_close"][2]
    return [
        ("title", "A new field-capacity default (Saxton-Rawls 33 kPa) and a conductivity sensitivity: JORN, January-June 2024"),
        ("sub", "The frozen 160-day JORN column re-run with the NEW package default field capacity (about 10 vol. %, replacing about 39.8 vol. %) at all-K/Ks multipliers 0.5, 1, 2 and 4. Only these two things changed. This is a sensitivity of the native SYRUP column, not a recalibration."),
        ("h1", "1. Summary"),
        ("bul", [f"**The new field capacity removes the old over-wetting of the surface but the fit as a whole is worse.** Pooled aligned RMSE against 282 sensor values (percentage points) is {P(0.5):.3f}, {P(1):.3f}, {P(2):.3f} and {P(4):.3f} for 0.5x, 1x, 2x and 4x, against {P(old1):.3f} for the old field capacity at 1x (the no-change reference is {P(1, 'persistence'):.3f}). The best new setting (4x) is {100 * (P(4) / P(old1) - 1):.0f}% worse than old FC 1x, and none of the new settings beats it. The old diffusion coefficient d and evaporation stress were fitted under the OLD field capacity; the comparison therefore shows what happens when only the retention changes, not what a re-fitted model would do.",
                 f"**Water now moves through the soil during storms.** All 25.088 mm of rain still infiltrates (peak 10.034 mm/h is below the lowest matrix K, {R[0.5]['cfg']['matrix']['conductivity_m_s'] * 3.6e6:.1f} mm/h), with no runoff. But the new field capacity is below the water content reached in the storm, so the above-field-capacity gravity step carries {B[1]['grav'][0]:.2f} mm across the upper interface and {B[1]['grav'][1]:.2f} mm across the lower one at 1x (upper {B[0.5]['grav'][0]:.2f}-{B[4]['grav'][0]:.2f} mm, lower {B[4]['grav'][1]:.2f}-{B[0.5]['grav'][1]:.2f} mm across the settings; these are separate internal pathway counts that can include the same water, so they are not added to each other or to losses), and {B[4]['drain']:.2f}-{B[0.5]['drain']:.2f} mm leaves through the open bottom. Under the old field capacity nothing moved by gravity and nothing drained.",
                 f"**The January 23 surface pulse is now close to the sensor.** The 0-10 cm layer rises {B[1]['jan_top']:.2f} points at 1x ({B[4]['jan_top']:.2f}-{B[0.5]['jan_top']:.2f} across the settings) against {o_top:.2f} at the 5.8 cm sensor; the old field capacity gave {B[old1]['jan_top']:.2f}. The shallow sensor is also fitted better than no change at 4x ({a(4, '501'):.3f} vs {ps[0]:.3f}), the only shallow setting in any of these runs to do so. Because the above-field-capacity cascade is not rate-limited (see Section 3), this agreement is consistent with reduced surface over-retention under the new field capacity. It does not show that 33 kPa is the actual field capacity of this soil, or that the transfer is correct.",
                 f"**The deep layer is now too wet and the middle layer overshoots.** The model deep layer reaches {d_hi(1):.1f} vol. % (its field capacity) at the January storm close, while the deep sensor never exceeds {o_max_deep:.2f} vol. % in the record; deep aligned RMSE is {a(4, '503'):.2f}-{a(0.5, '503'):.2f} against {a(old1, '503'):.2f} for old FC 1x. The 72 h middle-layer response is {B[1]['jan_mid72']:.2f} points (observed {o_mid72:.2f}); old FC gave {B[old1]['jan_mid72']:.2f}.",
                 f"**Within the SR33 runs, faster exchange lowers the error and costs water.** Evaporation rises with K ({B[0.5]['evap']:.1f}, {B[1]['evap']:.1f}, {B[2]['evap']:.1f}, {B[4]['evap']:.1f} mm vs {B[old1]['evap']:.1f} mm old); final storage falls ({B[0.5]['final']:.1f} to {B[4]['final']:.1f} mm vs {B[old1]['final']:.1f}). K and d act together in the Tietjen exchange, so they cannot be separated here. The lowest error at 4x is the edge of the tested range, so no best K is implied, and nothing here is a calibration, a default change or a field validation."]),
        ("h1", "2. What was run"),
        ("p", "The baseline is the frozen JORN run of the earlier report [7]: one 1 m² column of three 10 cm layers with an open bottom, daily weather with the observed storm sub-steps, NEON rain [5], Hargreaves-Samani evaporation demand [3] from NEON air temperature [6], initial water from the NEON soil sensors [4], matrix-only infiltration, no plants, transpiration, preferential flow, routing, wind or sediment (surface water is exported with zero residence). Only two things differ from it: the field capacities and the ten conductivity entries (the three layer Ks records in mm/h and m/s, the top matrix Ks and K, and both interface K), each multiplied once by the factor relative to the original (powers of two, exact in binary arithmetic). The configuration check shows nothing else differs apart from the source pin, the name and added provenance text."),
        ("p", f"**Field capacity.** The new values come from the current provider's default method [2] (the water content at 33 kPa, density-adjusted), using the measured sand, silt and clay of the 2013 Jornada megapit [8] (a spatial proxy for the 2024 plot 004 sensors, not demonstrably co-located), the three approved effective densities and this study's explicit air-entry override of 0. They are {fc[0]:.2f}, {fc[1]:.2f} and {fc[2]:.2f} vol. %, versus {R['fc_old'][0] * 100:.2f}, {R['fc_old'][1] * 100:.2f} and {R['fc_old'][2] * 100:.2f} vol. % in the old hybrid 6 kPa construction (Table 1). Saturation, wilting point and Ks are identical to the old run; the field-capacity audit already found no arithmetic error in either construction. The 6 kPa convention follows Tietjen et al. [1]; whether 33 kPa is appropriate for this desert sand is a separate question this report does not settle."),
        ("tab", "soil", "Table 1a. Soil inputs per layer (2013 megapit texture and organic matter; bulk density as approved for the earlier runs). Unchanged from the earlier runs."),
        ("tab", "ret", "Table 1b. Retention properties. Saturation, wilting point and Ks are unchanged; only the field-capacity columns differ (new 33 kPa value against the old 6 kPa hybrid)."),
        ("tab", "params", "Table 2. Conductivities by multiplier. The Ks records are the (user-perturbed) source estimates; the matrix K and the two interface K are the direct inputs of the run."),
        ("tab", "fixed", "Table 3. What is held fixed. The evaporation stress threshold and d were fitted under the old field capacity and are NOT re-fitted."),
        ("p", "**Clock.** As in the earlier runs this is a daily-weather approximation: one 86,400 s inter-event step per nominal day plus the observed storm sub-steps (27.5 added hours, so the physical clock ends at 161.146 days against 160 nominal days, final nominal close 2024-06-26 07:00 UTC). Evaporation and transpiration are off during rain. It is not an exact hourly field chronology. Sensor values after day 0 never drive the model; 282 sensor values (56, 147 and 79 at 5.8, 15.8 and 25.8 cm) are used for scoring, with gaps left as gaps."),
        ("h1", "3. Water: what the new field capacity changes"),
        ("tab", "budget", "Table 4. Event response and water pathways. Old FC 1x is the earlier run; the four SR33 columns are the new runs. The two interface rows are separate internal pathway counts of the gravity step and can include the same water: do not add them to each other or to evaporation, drainage or storage."),
        ("p", f"Rain is below the matrix K in every case, so the multipliers and the field-capacity change cannot alter how much rain enters. What changes is where it goes. In the old run the top layer kept the storm water (it rose to {B[old1]['peak_close'][0]:.1f} vol. % against a field capacity of {R['fc_old'][0] * 100:.1f}) and returned it slowly by diffusion. With a field capacity near 10 vol. %, a layer wetted past that value passes the excess down at once: {B[1]['grav_storm_days'][0] / B[1]['grav'][0] * 100:.1f}% of the interface-1 transfer happens on rain days, and the deep layer is filled to its own field capacity ({d_hi(1):.2f} vol. % against {fc[2]:.2f}) and then spills {B[1]['drain']:.2f} mm out of the bottom. This above-field-capacity step is not rate-limited by any conductivity (the Saxton-Rawls Ku is not used in the current bucket), so the SR33 K multiplier changes it only slightly (upper-interface count {B[0.5]['grav'][0]:.2f} to {B[4]['grav'][0]:.2f} mm, lower {B[0.5]['grav'][1]:.2f} to {B[4]['grav'][1]:.2f} mm; these two counts can include the same water). Between storms the layers below field capacity exchange by the Tietjen gradient exchange, upward net (top-middle {B[0.5]['top_mid_up']:.1f} to {B[4]['top_mid_up']:.1f} mm), which feeds evaporation more at large K. Highest daily-close values stay close to the new field capacity ({B[1]['peak_close'][0]:.2f}, {B[1]['peak_close'][1]:.2f} and {B[1]['peak_close'][2]:.2f} vol. % at 1x, against {fc[0]:.2f}, {fc[1]:.2f} and {fc[2]:.2f}), because excess is passed down during the storm step."),
        ("fig", "fig1_raw_overlay.png", "Figure 1. RAW comparison at the nominal daily closes: SR33 runs at 0.5x to 4x, the old-field-capacity 1x run (red dotted) and sensor point values (black circles; gaps are not interpolated). The dashed red line is each layer's NEW field capacity in vol. %; the old values (39.5-39.9 vol. %) are far above the plotted range. Triangles are day-0 values. The top panel shows observed rain."),
        ("h1", "4. Comparison with the sensors"),
        ("p", f"Two scorings are used. **RAW** compares model layer means directly with sensor point values. **Aligned** subtracts one constant day-0 offset per depth ({off[0] * 100:+.2f}, {off[1] * 100:+.2f} and {off[2] * 100:+.2f} points at 5.8, 15.8 and 25.8 cm) because a 10 cm layer mean is not a point reading; the offsets are the same in every run and model water is never altered by the alignment. All runs start from the same 22.9424 mm. Pooled scores weight every value equally, so the middle sensor (n = 147) dominates."),
        ("tab", "scores_aligned", "Table 5a. ALIGNED RMSE in percentage points (100 x m³/m³) by depth (n = number of sensor values) and pooled. No change = holding the day-0 sensor value. Old FC 2x and 4x are the earlier all-K/Ks runs, shown so K can be compared at matched settings."),
        ("tab", "scores_raw", "Table 5b. RAW RMSE in percentage points, same rows and columns."),
        ("tab", "bias", "Table 6a. RAW bias (model minus sensor, percentage points) by depth and pooled."),
        ("tab", "endpoint", "Table 6b. Last valid sensor value at each depth, paired with each SR33 run at the SAME close. The shallow sensor has no value after the 2024-03-15 close and the middle sensor none at the 2024-06-26 close; no model-end value is compared with these."),
        ("fig", "fig2_aligned_overlay.png", "Figure 2. The same runs as changes since day 0 (the aligned view; the scored quantity is the model change minus the sensor change). The dashed red line is the new field capacity minus the layer's day-0 content, in percentage points."),
        ("p", f"Depth by depth the answer differs. **Shallow:** every SR33 setting from 0.5x to 2x is worse than no change ({a(0.5, '501'):.3f}, {a(1, '501'):.3f}, {a(2, '501'):.3f} vs {ps[0]:.3f}); 4x is better ({a(4, '501'):.3f}), whereas the old field capacity gave {a(old1, '501'):.2f}. The shallow layer is therefore the clear gain, produced by removing the storm over-wetting. **Middle:** best at 4x ({a(4, '502'):.3f}), against {a(old1, '502'):.3f} for old FC 1x and {ps[1]:.2f} for no change; at 0.5x it is {a(0.5, '502'):.2f}. **Deep:** every SR33 setting is worse than no change ({a(4, '503'):.2f} to {a(0.5, '503'):.2f} vs {ps[2]:.2f}) and than old FC 1x ({a(old1, '503'):.2f}). The deep RAW bias is {S[4]['503']['raw']['bias']:+.2f} to {S[0.5]['503']['raw']['bias']:+.2f} points, and it is the dominant reason the pooled score stays above old FC 1x. The model deep layer is filled at the January storm, whereas the sensor stays below {o_max_deep:.1f} vol. %. The deep sensor has no value at the January 23 close (a gap, not zero). At the last deep sensor value (the final close, 2024-06-26 07:00 UTC) the model deep layer is {ep[2]['model'][4]:.2f} vol. % at 4x against a sensor of {ep[2]['obs']:.2f}, but {ep[2]['model'][0.5]:.2f} at 0.5x."),
        ("fig", "fig3_january_zoom.png", "Figure 3. Jan 20 - Feb 4 around the 10 mm Jan 23 storm: changes since day 0 with the fixed day-0 offset (no re-offset after the storm). Rain bars cover the nominal rainfall day; model states and sensor values are plotted at the daily close that follows it. The dashed red line is the new field capacity minus the layer's day-0 content."),
        ("p", f"Around the storm (Figure 3) the shallow rise is {B[0.5]['jan_top']:.2f}, {B[1]['jan_top']:.2f}, {B[2]['jan_top']:.2f} and {B[4]['jan_top']:.2f} points against {o_top:.2f} observed (old FC: {B[old1]['jan_top']:.2f}), and the 72 h middle-layer change is {B[0.5]['jan_mid72']:.2f}, {B[1]['jan_mid72']:.2f}, {B[2]['jan_mid72']:.2f} and {B[4]['jan_mid72']:.2f} points against {o_mid72:.2f} observed (old FC: {B[old1]['jan_mid72']:.2f}): the middle layer now overshoots at every setting, where old FC undershot. There is no shallow or middle sensor value at the final close (June 26). Compared at the same time, the last valid shallow value ({ep[0]['close'][:10]}) is {ep[0]['obs']:.2f} vol. % against {ep[0]['model'][1]:.2f} for SR33 1x, and the last valid middle value ({ep[1]['close'][:10]}) is {ep[1]['obs']:.2f} against {ep[1]['model'][1]:.2f} (Table 6b). The fast, instantaneous above-field-capacity cascade is not qualified and sends rain deeper on short timescales than the sensors suggest; omitted transpiration, the old-field-capacity-conditional d and evaporation stress, and soil and probe support that are not co-located (a 10 cm layer mean against a point sensor) may also contribute to the remaining discrepancies. These are untested explanations; none was tested or adopted here."),
        ("tab", "matched", "Table 7a. Old versus new field capacity at matched K: pooled RMSE in percentage points (n = 282). Only the field capacity differs within each row."),
        ("tab", "matched_budget", "Table 7b. Old versus new field capacity at matched K: water budget terms (mm)."),
        ("fig", "fig4_scores_and_budgets.png", "Figure 4. (a) Pooled RMSE by setting (dark = aligned, lighter = RAW; hatched = old field capacity; dotted line = pooled no-change reference). (b) Aligned RMSE by depth; dotted lines are the per-depth no-change values. (c) Water pathways in mm over 160 days, shown side by side and not stacked: evaporation, the gravity step across the upper and the lower interface separately (internal counts that can include the same water; they are not losses and must not be added), and bottom drainage. Categories are equally spaced settings, not a continuous fit."),
        ("h1", "5. Verification and run time"),
        ("bul", ["Before consuming any result I re-hashed all 120 files of Root's seal and checked them again afterwards. From the daily rows of all runs I recomputed the per-depth and pooled RAW, aligned and no-change RMSE and counts, the water budgets, the Jan 23 responses, the zero runoff/ponding/onset/room-rejection checks (bottom drainage is positive and accounted), NumPy = Numba row equality, the 900 s differences and the source pins, and compared the sensor values with Root's independent raw-file audit; all agree with Root's audit (relative 1e-9).",
                 f"NumPy and Numba results are exactly identical in every tracked output at all four settings (maximum difference 0). A 900 s storm step changes layer water by at most {max(max(B[f]['d900_pp']) for f in NEW):.5f} percentage points. Root's independent 90-digit calculation (open gravity cascade, evaporation and pair relaxation; it does not call the production update) agrees to {max(B[f]['dec_max_err_m'] for f in NEW):.1e} m or better, and the native conservation bounds pass.",
                 "Source packages SYRUP 180 files (4e38a472...) and MAPLE 412 files (fbc504bb...) are exact before and after every run. The first k0p5 launch failed before any import because the runner's bootstrap file was missing beside its config; no model step ran, and the successful launch record used here is actual_numba_runs_v2.",
                 "Restart: Root's day-80 split and fresh-process resume of the 1x case reproduces every tracked output exactly. This is a driver-level component state reconstruction, not a production checkpoint or coupled-model acceptance. No GPU, scaling or field validation is claimed."]),
        ("tab", "timing", "Table 8a. Timing, separated. Separate fresh runs on a shared host with other jobs: not a controlled speed comparison."),
        ("tab", "checks", "Table 8b. Verification per run: closure, 900 s step, 90-digit control and NumPy/Numba comparison."),
        ("p", "Integration takes about 0.04 s in each run. The fresh-process time (about 7 s) is dominated by one-time source verification, input preparation and compilation and says nothing about solver speed or the effect of K. One 1 m² CPU column: no scaling claim."),
        ("h1", "6. Limits"),
        ("bul", ["This is a prescribed sensitivity: no calibration, default change, adoption of the K multiplier or claim that the new field capacity improves the fit. The evaporation stress and d are conditional on the old field capacity; K and d are confounded; the lowest SR33 error is at the edge of the tested range (4x).",
                 "The above-field-capacity gravity cascade is not rate-limited by any conductivity; how much water it passes down, and therefore the deep-layer wetting and bottom drainage, depends on the current water content, the rain inputs and the boundaries as well as on field capacity; K and d change those inventories only indirectly. The Saxton-Rawls Ku is not used.",
                 "Point sensors against 10 cm layer means, a daily clock, a thin open-bottom 30 cm column, no transpiration, no preferential flow, and the 030 vs 020 air-temperature height difference inherited from the original run. No significance test is made."]),
        ("bul2", ["Persistent Green-Ampt wetting history has no automatic reset or drying recovery, and no general repeated-storm infiltration qualification exists. Here all rain is unponded and below the matrix K, so wetting history does not control actual intake in these runs."]),
        ("h1", "References"), ("refs", REFS),
        ("h1", "Appendix. Identity (compact)"),
        ("p", "Sources: SYRUP package 180 files, SHA-256 " + SYR_NEW + "; MAPLE package 412 files, SHA-256 " + MAP + ". Case-file hashes: " + "; ".join(f"{nm(f)} {sha(TASK / 'study' / KEYS[f] / 'case_numba.json')[:16]}..." for f in NEW)
         + ". Full hashes of every sealed file, the daily trajectories of all runs and the parameter differences are in tables/provenance_seal.csv, tables/case_hashes.csv, tables/parameter_diff_vs_original.csv and tables/daily_trajectories.csv. Raw NEON files are not copied.")]


def to_docx(Bk, T, path):
    doc = Document()
    setup_styles(doc)
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Inches(8.5), Inches(11)
    sec.left_margin = sec.right_margin = Inches(0.9)
    sec.top_margin = sec.bottom_margin = Inches(0.8)
    fp = sec.footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    fr = fp.add_run("SYRUP JORN 160-day: SR33 field capacity x all-K/Ks 0.5-4 · sensitivity, not a recalibration · page ")
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
            p.add_run().add_picture(str(FIG / b[1]), width=Inches(6.0 if b[1] != "fig4_scores_and_budgets.png" else 6.7))
            runs(doc.add_paragraph(style="RepCaption"), b[2])
        elif k == "refs":
            for n, c, u in b[1]:
                p = doc.add_paragraph()
                p.paragraph_format.left_indent = Inches(0.25)
                p.paragraph_format.first_line_indent = Inches(-0.25)
                p.paragraph_format.space_after = Pt(2)
                p.add_run(f"[{n}] ").bold = True
                p.add_run(f"{c} {u}".strip()).font.size = Pt(9)
    doc.core_properties.title = "SYRUP JORN 160-day: SR33 field capacity and all-K/Ks sensitivity"
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
    assert z.testzip() is None and len(media) == nfig == 4 and ntab == ntabs == nhdr and pagef
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
    S, B = review(R)
    for p in list(TAB.glob("*.csv")) + list(DATA.glob("*")):
        p.unlink()
    fig_trajectory(R, "fig1_raw_overlay.png", False, title="RAW values at each nominal daily close (SR33 field capacity, K multipliers 0.5-4, and old-FC 1x)")
    fig_trajectory(R, "fig2_aligned_overlay.png", True, title="Changes since day 0 (the aligned view) at each nominal daily close")
    fig_trajectory(R, "fig3_january_zoom.png", True, idx=range(2, 18), title="Jan 20 - Feb 4: response to the 10 mm Jan 23 storm (changes since day 0)", zoom=True)
    fig_scores(S, B)
    T = tables(R, S, B)
    for name, (hd, rows) in T.items():
        csvw(TAB / f"{name}.csv", hd, rows)
    # parameter differences vs the OLD (original) config
    base = R[("old", 1)]["cfg"]
    diffrows = []

    def walk(x, y, f, p=""):
        if isinstance(x, dict):
            for k in sorted(x.keys() | y.keys()):
                if k in x and k in y:
                    walk(x[k], y[k], f, f"{p}.{k}")
                else:
                    diffrows.append([f, f"{p}.{k}", repr(x.get(k)) if k in x else "(absent)", repr(y.get(k)) if k in y else "(added)", ""])
        elif isinstance(x, list):
            for i, (u, v) in enumerate(zip(x, y)):
                walk(u, v, f, f"{p}[{i}]")
        elif x != y:
            diffrows.append([f, p, repr(x), repr(y), repr(y / x) if isinstance(x, float) and x else ""])
    for f in NEW:
        walk(base, R[f]["cfg"], f)
    csvw(TAB / "parameter_diff_vs_original.csv", ["factor", "field", "original_config", "new_config", "ratio"], diffrows)
    csvw(TAB / "provenance_seal.csv", ["file", "sha256", "bytes"], [[k, v["sha256"], v["bytes"]] for k, v in R["seal"].items()])
    csvw(TAB / "case_hashes.csv", ["setting", "config_sha256_in_result", "case_numba_json_sha256"], [[nm(f), R[f]["res"]["config_sha256"], sha(TASK / "study" / KEYS[f] / "case_numba.json")] for f in NEW])
    keys = [*NEW, *(("old", f) for f in OLDF)]
    lab = lambda k: nm(k[1], True).replace(" ", "_") if isinstance(k, tuple) else nm(k).replace(" ", "_")
    hdr = ["day_index", "date_utc_minus7", "nominal_close_utc", "physical_t_s", "rain_mm", "pet_mm"] + [f"theta_l{k_}_{lab(k)}" for k in keys for k_ in (1, 2, 3)] + [f"evap_mm_{lab(k)}" for k in keys] + ["obs_501", "obs_502", "obs_503"]
    rows = []
    for i, r0 in enumerate(R[1]["res"]["rows"]):
        rows.append([r0["day_index"], r0["date"], r0["nominal_close_utc"], r0["physical_t_s"], 1000 * r0["rain_m3"], r0["pet_mm_day"]] + [R[k]["res"]["rows"][i]["layer_theta"][j] for k in keys for j in range(3)]
                    + [1000 * R[k]["res"]["rows"][i]["evaporation_m3"] for k in keys] + ["" if v is None else v for v in r0["obs_theta"]])
    csvw(TAB / "daily_trajectories.csv", hdr, rows)
    for f in NEW:
        k = KEYS[f]
        shutil.copyfile(TASK / "study" / k / "case_numba.json", DATA / f"case_numba_{k}.json")
        shutil.copyfile(TASK / "study" / k / "root_decimal_verification.json", DATA / f"root_decimal_verification_{k}.json")
        shutil.copyfile(TASK / "study" / k / "actual_numba/timing.json", DATA / f"timing_numba_{k}.json")
    for name in ("root_result_audit.json", "root_independent_inputs.json", "root_parameter_verification.json", "root_restart_verification.json"):
        shutil.copyfile(TASK / name, DATA / name)
    shutil.copyfile(OLD1 / "case_numba.json", DATA / "case_numba_old_fc_1x.json")
    blk = blocks(R, S, B)
    to_docx(blk, T, DOCX)
    info = verify_docx(DOCX, blk)
    seal_check()                                                    # re-hash after consuming
    (HERE / "report.md").write_text(to_markdown(blk, T), newline="\n")
    (HERE / "verification.md").write_text("# Verification record\n\nBefore building, all 120 files of `root_run_seal.json` (including the old 1x/2x/4x results used for comparison) were re-hashed; they were re-hashed again after the document was built. From the daily rows of all seven runs the builder recomputed per-depth and pooled RAW, aligned and no-change RMSE/bias/counts "
                                          "(matching Root's `root_result_audit.json` to relative 1e-9 for the four new runs and the old 1x run), the water budgets and closure by rain/intake/E/drain/final storage, gravity and diffusive transfers, the zero runoff/ponding/onset/room-rejected checks, NumPy = Numba equality of all rows, the 900 s maximum differences, the "
                                          "before/after source pins (SYRUP 4e38a472.../180, MAPLE fbc504bb.../412), Root's 90-digit controls, and the sensor values against Root's independent raw-file audit. Static DOCX checks: zip integrity, one media file per figure (4), table count equals repeated-header count, footer page field, no placeholder text. "
                                          "No Office renderer was available, so rendered pagination and legibility were not checked by the builder (figures were viewed as PNG).\n\n`root_restart_verification.json` is a Root receipt written after the seal; it is copied (SHA in the manifest) but not part of the 120-file seal.\n\n```\n" + json.dumps(info, indent=1) + "\n```\n", newline="\n")
    man = {"docx": {"file": DOCX.name, **info}, "figures": {p.name: sha(p) for p in sorted(FIG.glob("*.png"))}, "tables": {p.name: sha(p) for p in sorted(TAB.iterdir())}, "data": {p.name: sha(p) for p in sorted(DATA.iterdir())},
           "builder": sha(Path(__file__)), "report_md": sha(HERE / "report.md"), "seal_files_rechecked": len(R["seal"]), "root_restart_verification_sha256": sha(TASK / "root_restart_verification.json")}
    (HERE / "artifact_manifest.json").write_text(json.dumps(man, indent=1) + "\n", newline="\n")
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
