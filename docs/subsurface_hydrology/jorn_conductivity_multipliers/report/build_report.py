"""Builds the JORN 160-day all-K/Ks 1x / 1.2x / 2x / 3x / 4x comparison report: figures, CSV tables, editable Markdown, verification note, manifest and the Word document.

    PYTHONPATH=<tasks>/site_hydraulics_interface_comparison_20261117/report_tools MPLCONFIGDIR=/tmp/syrup-kmult-report PYTHONDONTWRITEBYTECODE=1 /home/okin/MAPLE/.venv/bin/python -B build_report.py

DOCUMENTS ONLY. Reads Root's sealed results (every file used is re-hashed against `root_run_seal.json` first). Declared inputs: JORN_KM_TASK (this task folder), JORN_BASE_TASK (original 1x case), JORN_P20_TASK (1.2x case).
Not portable without them. It runs no model, fits nothing, uses no network or GPU and writes only inside its own folder.
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
from docx.shared import Inches, Pt, RGBColor

HERE = Path(__file__).resolve().parent
TASK = Path(os.environ.get("JORN_KM_TASK", str(HERE.parent)))
TASKS = TASK.parent
DIRS = {1: Path(os.environ.get("JORN_BASE_TASK", str(TASKS / "jorn_160_day_run_20261119"))), 1.2: Path(os.environ.get("JORN_P20_TASK", str(TASKS / "jorn_conductivity_plus20_20261120"))), 2: TASK / "x2", 3: TASK / "x3", 4: TASK / "x4"}
FACT = [1, 1.2, 2, 3, 4]
NAME = {1: "1x (original)", 1.2: "1.2x", 2: "2x", 3: "3x", 4: "4x"}
COL = {1: "#2E8B57", 1.2: "#888888", 2: "#1F5FA8", 3: "#E07B00", 4: "#7B3FA0"}
LS = {1: "-", 1.2: ":", 2: "--", 3: "-.", 4: (0, (5, 1))}
FIG, TAB, DATA = HERE / "figures", HERE / "tables", HERE / "data"
DOCX = HERE / "SYRUP_JORN_160day_K_Ks_multipliers_2x_3x_4x_20261120.docx"
LV = ["501", "502", "503"]
DEPTH = {"501": "5.8 cm sensor (layer 0-10 cm)", "502": "15.8 cm sensor (layer 10-20 cm)", "503": "25.8 cm sensor (layer 20-30 cm)"}
for d in (FIG, TAB, DATA):
    d.mkdir(exist_ok=True)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def csvw(path, header, rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


# ------------------------------------------------------------------------------------------------ sealed inputs
def used_files():
    out = []
    for f in FACT:
        d = DIRS[f]
        names = ["case_numba.json", "actual_numba/result.json", "actual_numba/timing.json", "actual_numba_runs/result.json", "root_result_audit.json"]
        if f in (2, 3, 4):
            names += ["parameter_changes.json", "root_decimal_verification.json", "actual_numpy/result.json", "actual_numba_dt900/result.json", "actual_numba_dt900/timing.json", "actual_numpy/timing.json", "actual_numba_runs/sources_before.json",
                      "actual_numba_runs/sources_after.json", "actual_numba_dt900_runs/result.json", "actual_numpy_runs/result.json", "frozen_helper_manifest.json"]
        out += [(f, d / n) for n in names]
    out += [(None, TASK / "comparison.json"), (None, TASK / "root_parameter_validation.json")]
    return out


def load():
    seal = json.loads((TASK / "root_run_seal.json").read_text())
    pend = []
    for f, p in used_files():
        key = str(p)
        assert key in seal, f"{key} is not in root_run_seal.json"
        assert sha(p) == seal[key], f"{key} differs from root_run_seal.json"
        pend.append((f, p))
    seal2 = json.loads((DIRS[1.2] / "root_run_seal.json").read_text())                 # the 1.2x decimal check is sealed by its own task
    dec12 = DIRS[1.2] / "root_decimal_verification.json"
    assert sha(dec12) == seal2[str(dec12)]
    pend.append((1.2, dec12))
    seal = {**seal, str(dec12): seal2[str(dec12)]}
    J = lambda f, r: json.loads((DIRS[f] / r).read_text())
    R = {"seal": seal, "pending": pend, "cmp": json.loads((TASK / "comparison.json").read_text()), "val": json.loads((TASK / "root_parameter_validation.json").read_text())}
    for f in FACT:
        R[f] = {"res": J(f, "actual_numba/result.json"), "cfg": J(f, "case_numba.json"), "tm": J(f, "actual_numba/timing.json"), "wall": J(f, "actual_numba_runs/result.json"), "audit": J(f, "root_result_audit.json")}
        if f != 1:
            R[f]["dec"] = J(f, "root_decimal_verification.json")
        if f in (2, 3, 4):
            R[f]["np"] = J(f, "actual_numpy/result.json")
            R[f]["d900"] = J(f, "actual_numba_dt900/result.json")
    return R


def prep(R):
    R["obs"] = np.array([[np.nan if v is None else v for v in r["obs_theta"]] for r in R[1]["res"]["rows"]])
    for f in FACT:
        rows = R[f]["res"]["rows"]
        R[f]["theta"] = np.array([r["layer_theta"] for r in rows])
        other = np.array([[np.nan if v is None else v for v in r["obs_theta"]] for r in rows])
        assert np.array_equal(np.nan_to_num(other, nan=-1), np.nan_to_num(R["obs"], nan=-1))
        assert R[f]["res"]["plan"]["initial"]["layer_theta"] == R[1]["res"]["plan"]["initial"]["layer_theta"]
    R["theta0"] = np.array(R[1]["res"]["plan"]["initial"]["layer_theta"])
    R["obs0"] = np.array(R[1]["res"]["plan"]["initial"]["point_theta"])
    R["close"] = [dt.date.fromisoformat(r["date"]) + dt.timedelta(days=1) for r in R[1]["res"]["rows"]]
    R["start"] = dt.date.fromisoformat(R[1]["res"]["rows"][0]["date"])
    return R


def stats(e):
    return {"n": int(e.size), "rmse": 100 * math.sqrt(float(np.mean(e**2))), "bias": 100 * float(e.mean())}


def scores(R, f):
    th, ob = R[f]["theta"], R["obs"]
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


def flux(R, f):
    rows, th = R[f]["res"]["rows"], R[f]["theta"]
    d = np.array([r["diffusion_m3"] for r in rows]) * 1000
    w = np.array([r["water_m"] for r in rows]) * 1000
    pk = int(np.argmax(th[:, 2]))
    jan = next(i for i, r in enumerate(rows) if r["date"] == "2024-01-23")
    return {"peak_date": rows[pk]["date"], "peak_gain_mm": float(w[pk, 2] - R["theta0"][2] * 100), "final": (th[-1] * 100).tolist(), "top_mid_up": float(-d[:, 0].sum()), "md_down": float(d[d[:, 1] > 0, 1].sum()),
            "md_up": float(-d[d[:, 1] < 0, 1].sum()), "jan_top": float((th[jan, 0] - th[jan - 1, 0]) * 100), "jan_mid72": float((th[jan + 2, 1] - th[jan - 1, 1]) * 100), "obs_deep_last": float(R["obs"][~np.isnan(R["obs"][:, 2]), 2][-1] * 100)}


def check(R):
    c = {rec["factor"]: rec for rec in R["cmp"]["records"]}
    S, B, F = {}, {}, {}
    for f in FACT:
        S[f], F[f] = scores(R, f), flux(R, f)
        rows, res = R[f]["res"]["rows"], R[f]["res"]
        rec = c[f]
        for v in LV:
            assert math.isclose(S[f][v]["aligned"]["rmse"], rec["scores"][v]["offset_aligned"]["rmse_percentage_points"], rel_tol=1e-9) and math.isclose(S[f][v]["raw"]["rmse"], rec["scores"][v]["raw"]["rmse_percentage_points"], rel_tol=1e-9)
            assert S[f][v]["aligned"]["n"] == rec["scores"][v]["offset_aligned"]["n"]
        assert math.isclose(S[f]["pooled"]["aligned"]["rmse"], rec["pooled"]["offset_aligned"]["rmse_percentage_points"], rel_tol=1e-9) and math.isclose(S[f]["pooled"]["raw"]["rmse"], rec["pooled"]["raw"]["rmse_percentage_points"], rel_tol=1e-9)
        assert S[f]["pooled"]["aligned"]["n"] == 282
        B[f] = {"rain": 1000 * sum(r["rain_m3"] for r in rows), "intake": 1000 * sum(r["intake_m3"] for r in rows), "evap": 1000 * sum(r["evaporation_m3"] for r in rows), "final": 1000 * rows[-1]["storage_m3"]}
        assert abs(22.9424 + B[f]["intake"] - B[f]["evap"] - B[f]["final"]) < 1e-9 and math.isclose(B[f]["intake"], B[f]["rain"], rel_tol=1e-12)
        assert all(r["runoff_pool_m3"] == 0 and r["surface_return_m3"] == 0 and r["drainage_m3"] == 0 and r["n_ponded_steps"] == 0 and r["n_onsets"] == 0 and all(x == 0 for x in r["percolation_m3"]) for r in rows)
        assert math.isclose(B[f]["evap"], rec["budgets_mm"]["evaporation"], rel_tol=1e-12) and math.isclose(B[f]["final"], rec["budgets_mm"]["final"], rel_tol=1e-12)
        assert res["complete"] and res["physical_end_t_s"] == 13923000.0 and res["ledger"]["n_matrix_steps"] == 55 and res["ledger"]["n_soil_steps"] == 215 and rows[-1]["nominal_close_utc"] == "2024-06-26T07:00:00Z"
        assert max(r["theta_max"] if False else max(r["layer_theta"]) for r in rows) < 0.3951
        if f in (2, 3, 4):
            assert R[f]["np"]["rows"] == rows
            d9 = np.array([r["layer_theta"] for r in R[f]["d900"]["rows"]])
            assert math.isclose(float(np.abs(d9 - R[f]["theta"]).max()), rec["dt900_max_theta_difference"], rel_tol=1e-9)
    # configuration differences vs the original: ten numeric entries x factor, nothing else
    base = R[1]["cfg"]
    R["diffs"] = {}
    for f in FACT[1:]:
        diffs = []

        def walk(x, y, p="", diffs=diffs):
            if isinstance(x, dict):
                for k in x.keys() | y.keys():
                    if k in x and k in y:
                        walk(x[k], y[k], f"{p}.{k}", diffs)
                    elif not p.startswith(".provenance"):
                        raise AssertionError(f"key mismatch {p}.{k}")
            elif isinstance(x, list):
                assert len(x) == len(y)
                for i, (u, v) in enumerate(zip(x, y)):
                    walk(u, v, f"{p}[{i}]", diffs)
            elif x != y:
                diffs.append((p, x, y))
        walk(base, R[f]["cfg"])
        num = [d for d in diffs if d[0] != ".name"]
        assert len(num) == 10 and all(math.isclose(d[2] / d[1], f, rel_tol=1e-14) for d in num), (f, num)
        R["diffs"][f] = num
    return S, B, F, c


# ------------------------------------------------------------------------------------------------ figures
def style(ax):
    ax.grid(alpha=0.25)
    ax.tick_params(labelsize=8)


def rainspans(ax, R):
    for r, c in zip(R[1]["res"]["rows"], R["close"]):
        if r["storm_rows"]:
            ax.axvspan(c - dt.timedelta(days=1), c, color="#BFD9F0", alpha=0.5, lw=0, zorder=0)


def monthaxis(ax):
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))


def fig_raw(R):
    fig, axs = plt.subplots(3, 1, figsize=(7.2, 6.8), sharex=True)
    x = [R["start"]] + R["close"]
    for j, ax in enumerate(axs):
        for f in FACT:
            ax.plot(x, np.r_[R["theta0"][j], R[f]["theta"][:, j]] * 100, color=COL[f], ls=LS[f], lw=1.3 if f != 1.2 else 1.0, label=NAME[f])
        ax.scatter(R["close"], R["obs"][:, j] * 100, facecolors="none", edgecolors="black", s=12, lw=0.7, label="sensor point value")
        ax.scatter([R["start"]], [R["theta0"][j] * 100], marker="^", color=COL[1], s=40, zorder=5, label="initial model layer mean")
        ax.scatter([R["start"]], [R["obs0"][j] * 100], marker="^", facecolors="none", edgecolors="black", s=40, zorder=5, label="initial sensor value")
        rainspans(ax, R)
        ax.set_ylabel("water content (vol. %)", fontsize=8.5)
        ax.set_title(DEPTH[LV[j]], fontsize=9.5, loc="left")
        style(ax)
    monthaxis(axs[-1])
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=7.5, frameon=False)
    fig.suptitle("RAW values at each nominal daily close (shaded = nominal rain days)", fontsize=10)
    fig.tight_layout(rect=(0, 0.1, 1, 0.97))
    fig.savefig(FIG / "fig1_raw_overlay.png", dpi=150)
    plt.close(fig)


def fig_diff(R):
    fig, axs = plt.subplots(3, 1, figsize=(7.2, 6.0), sharex=True)
    for j, ax in enumerate(axs):
        ax.axhline(0, color="black", lw=0.7)
        for f in FACT[1:]:
            ax.plot(R["close"], (R[f]["theta"][:, j] - R[1]["theta"][:, j]) * 100, color=COL[f], ls=LS[f], lw=1.3, label=NAME[f])
        rainspans(ax, R)
        ax.set_ylabel("change vs 1x\n(percentage points)", fontsize=8)
        ax.set_title(DEPTH[LV[j]], fontsize=9.5, loc="left")
        style(ax)
    monthaxis(axs[-1])
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.suptitle("Modelled water content minus the original 1x run (shaded = rain days)", fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(FIG / "fig2_difference.png", dpi=150)
    plt.close(fig)


def fig_zoom(R):
    idx = range(2, 18)
    x = [R["close"][i] for i in idx]
    fig, axs = plt.subplots(4, 1, figsize=(7.2, 7.0), sharex=True, gridspec_kw={"height_ratios": [0.8, 1.4, 1.4, 1.4]})
    axs[0].bar([R["close"][i] - dt.timedelta(days=0.5) for i in idx], [R[1]["res"]["rows"][i]["observed_rain_mm"] for i in idx], width=0.8, color="#1F5FA8")
    axs[0].set_ylabel("rain (mm/day)", fontsize=8.5)
    axs[0].set_title("Observed rain per nominal day", fontsize=9.5, loc="left")
    style(axs[0])
    for j in range(3):
        ax = axs[j + 1]
        ax.axhline(0, color="#777777", lw=0.8, ls="--")
        for f in FACT:
            ax.plot(x, [(R[f]["theta"][i, j] - R["theta0"][j]) * 100 for i in idx], color=COL[f], ls=LS[f], lw=1.2, label=NAME[f])
        ax.scatter(x, [(R["obs"][i, j] - R["obs0"][j]) * 100 for i in idx], facecolors="none", edgecolors="black", s=20, label="sensor change since day 0")
        ax.set_ylabel("change (pct. points)", fontsize=8.5)
        ax.set_title(DEPTH[LV[j]], fontsize=9.5, loc="left")
        style(ax)
    axs[-1].xaxis.set_major_locator(mdates.DayLocator(interval=2))
    axs[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    h, l = axs[1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=7.5, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(FIG / "fig3_rain_pulse_zoom.png", dpi=150)
    plt.close(fig)


def fig_scores(S):
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.3), sharey=False)
    xs = np.arange(len(FACT))
    for ax, kind, title in ((axs[0], "aligned", "ALIGNED (one fixed day-0 offset)"), (axs[1], "raw", "RAW")):
        for v, c, lab in (("501", "#B5651D", "5.8 cm"), ("502", "#1F5FA8", "15.8 cm"), ("503", "#7B3FA0", "25.8 cm")):
            ax.plot(xs, [S[f][v][kind]["rmse"] for f in FACT], marker="o", ms=3.5, color=c, lw=1.1, label=lab)
        ax.plot(xs, [S[f]["pooled"][kind]["rmse"] for f in FACT], marker="s", ms=4.5, color="black", lw=1.8, label="all 282 pooled")
        ax.axhline(S[1]["pooled"]["persistence"]["rmse"], color="#888888", ls=":", lw=1)
        ax.set_xticks(xs)
        ax.set_xticklabels([NAME[f].split(" ")[0] for f in FACT], fontsize=8)
        ax.set_xlabel("conductivity multiplier (all K and Ks)", fontsize=8)
        ax.set_ylabel("RMSE (percentage points)", fontsize=8)
        ax.set_title(title, fontsize=9)
        style(ax)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h + [plt.Line2D([], [], color="#888888", ls=":")], l + ["pooled no-change reference (black curve only)"], loc="lower center", ncol=5, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(FIG / "fig4_rmse_vs_factor.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ tables
def g(x, n=3):
    return f"{x:.{n}f}"


def tables(R, S, B, F, C):
    T = {}
    rows = []
    for lab, fn in (("Ks, layer 0-10 cm (source estimate)", lambda f: R[f]["cfg"]["provenance"]["layer_properties"][0]["Ks_mm_h"]), ("Ks, layer 10-20 cm (source estimate)", lambda f: R[f]["cfg"]["provenance"]["layer_properties"][1]["Ks_mm_h"]),
                    ("Ks, layer 20-30 cm (source estimate)", lambda f: R[f]["cfg"]["provenance"]["layer_properties"][2]["Ks_mm_h"]), ("Matrix K, top layer (direct input)", lambda f: R[f]["cfg"]["matrix"]["conductivity_m_s"] * 3.6e6),
                    ("Interface K, 0-10 / 10-20 cm (direct input)", lambda f: R[f]["cfg"]["layers"]["interface_conductivity_m_s"][0] * 3.6e6), ("Interface K, 10-20 / 20-30 cm (direct input)", lambda f: R[f]["cfg"]["layers"]["interface_conductivity_m_s"][1] * 3.6e6)):
        rows.append([lab] + [g(fn(f), 1) for f in FACT])
    for i, rec in enumerate(R["cmp"]["records"]):            # cross-check against Root's parameter table
        assert math.isclose(rec["parameters"]["matrix_K_mm_h"], R[rec["factor"]]["cfg"]["matrix"]["conductivity_m_s"] * 3.6e6, rel_tol=1e-12)
    T["params"] = (["Quantity (mm/h)"] + [NAME[f] for f in FACT], rows)
    rows = []
    for v in LV + ["pooled"]:
        label = DEPTH[v].split(" (")[0] if v != "pooled" else "All depths pooled"
        n = S[1][v]["raw"]["n"]
        rows.append([f"{label} - aligned", n, g(S[1][v]["persistence"]["rmse"])] + [g(S[f][v]["aligned"]["rmse"]) for f in FACT])
        rows.append([f"{label} - RAW", n, "-"] + [g(S[f][v]["raw"]["rmse"]) for f in FACT])
    T["scores"] = (["Depth - scoring", "n", "No-change RMSE"] + [NAME[f] for f in FACT], rows)
    rows = [["Initial storage (mm)"] + ["22.9424"] * 5, ["Rain = intake (mm)"] + [g(B[f]["intake"]) for f in FACT], ["Evaporation (mm)"] + [g(B[f]["evap"], 4) for f in FACT], ["Runoff, percolation, drainage (mm)"] + ["0"] * 5,
            ["Final storage (mm)"] + [g(B[f]["final"], 4) for f in FACT], ["Deep-layer peak gain (mm)"] + [g(F[f]["peak_gain_mm"], 3) for f in FACT], ["Deep-layer peak, nominal day"] + [F[f]["peak_date"][5:] for f in FACT],
            ["Deep layer, final (vol. %)"] + [g(F[f]["final"][2], 2) for f in FACT], ["Middle layer, final (vol. %)"] + [g(F[f]["final"][1], 2) for f in FACT], ["Net upward top-middle exchange (mm)"] + [g(F[f]["top_mid_up"], 2) for f in FACT],
            ["Jan 23 top-layer rise (points)"] + [g(F[f]["jan_top"], 2) for f in FACT], ["Jan 23 middle layer, 72 h (points)"] + [g(F[f]["jan_mid72"], 3) for f in FACT]]
    T["budget"] = (["Quantity"] + [NAME[f] for f in FACT], rows)
    rows = []
    for f in FACT:
        tm = R[f]["tm"]
        cl = R[f]["res"]["cumulative_closure"]
        rows.append([NAME[f], g(tm["integration_s"], 4), g(tm["operator_prepare_s"], 2), g(R[f]["wall"]["seconds"], 2), f"{abs(cl['soil_residual_m3']):.1e} / {cl['soil_bound_m3']:.2e}",
                     "-" if f == 1 else f"{100 * C[f]['dt900_max_theta_difference']:.5f}", "-" if f == 1 else f"{R[f]['dec']['cases'][0]['max_water_error_m']:.2e}"])
    T["checks"] = (["Run", "Integration (s)", "Operator prep + JIT (s)", "Full fresh command (s)", "Absolute closure residual / native bound (m³)", "900 s step: max change (points)", "90-digit check: max water error (m)"], rows)
    return T


# ------------------------------------------------------------------------------------------------ text
REFS = [(1, "Tietjen, B., Zehe, E., & Jeltsch, F. (2009). Simulating plant water availability in dry lands under climate change: a generic model of two soil layers. Water Resources Research 45, W01418.", "https://doi.org/10.1029/2007WR006589"),
        (2, "Saxton, K. E., & Rawls, W. J. (2006). Soil water characteristic estimates by texture and organic matter for hydrologic solutions. Soil Science Society of America Journal 70, 1569-1578.", "https://doi.org/10.2136/sssaj2005.0117"),
        (3, "Hargreaves, G. H., & Samani, Z. A. (1985). Reference crop evapotranspiration from temperature. Applied Engineering in Agriculture 1, 96-99.", "https://doi.org/10.13031/2013.26773"),
        (4, "NEON (RELEASE-2026). Soil water content and water salinity, DP1.00094.001.", "https://data.neonscience.org/data-products/DP1.00094.001/RELEASE-2026"),
        (5, "NEON (RELEASE-2026). Precipitation, DP1.00045.001.", "https://data.neonscience.org/data-products/DP1.00045.001/RELEASE-2026"),
        (6, "NEON (RELEASE-2026). Single aspirated air temperature, DP1.00002.001.", "https://data.neonscience.org/data-products/DP1.00002.001/RELEASE-2026"),
        (7, "Earlier reports of this study, cited by file name only: SYRUP_JORN_160day_rainfall_comparison_20261119.docx (original 1x run) and SYRUP_JORN_160day_K_Ks_plus20_comparison_20261120.docx (1.2x run).", "")]


def blocks(R, S, B, F, C):
    P = {f: S[f]["pooled"] for f in FACT}
    imp = {f: 100 * (1 - P[f]["aligned"]["rmse"] / P[1]["aligned"]["rmse"]) for f in FACT}
    for f in FACT:
        assert math.isclose(imp[f], C[f]["aligned_pooled_improvement_percent_vs_original"], rel_tol=1e-9, abs_tol=1e-12)
    best_al = min(FACT, key=lambda f: P[f]["aligned"]["rmse"])
    best_raw = min(FACT, key=lambda f: P[f]["raw"]["rmse"])
    assert best_al == 2 and best_raw == 1.2
    top_best = min(FACT, key=lambda f: S[f]["501"]["aligned"]["rmse"])
    ps = S[1]["501"]["persistence"]["rmse"], S[1]["502"]["persistence"]["rmse"], S[1]["503"]["persistence"]["rmse"]
    assert all(S[f]["501"]["aligned"]["rmse"] > ps[0] for f in FACT) and all(S[f]["503"]["aligned"]["rmse"] < ps[2] for f in (2, 3, 4)) and S[1.2]["503"]["aligned"]["rmse"] > ps[2]
    off = R["theta0"] - R["obs0"]
    a = lambda f, v: S[f][v]["aligned"]["rmse"]
    return [
        ("title", "What happens if every conductivity is doubled, tripled or quadrupled? JORN, January-June 2024"),
        ("sub", "A prescribed sensitivity of the native SYRUP column: every K and Ks is multiplied once by 2, 3 or 4 relative to the original run, with everything else identical. The original 1x and 1.2x results are shown for comparison. This is not a recalibration."),
        ("h1", "1. Summary"),
        ("bul", [f"**Lowest pooled error at 2x, but not a best parameter.** Pooled aligned RMSE against 282 sensor values is {P[1]['aligned']['rmse']:.3f} (1x), {P[1.2]['aligned']['rmse']:.3f} (1.2x), {P[2]['aligned']['rmse']:.3f} (2x), {P[3]['aligned']['rmse']:.3f} (3x) and {P[4]['aligned']['rmse']:.3f} (4x) percentage points; relative to 1x that is {imp[1.2]:.1f}%, {imp[2]:.1f}%, {imp[3]:.1f}% and {imp[4]:.1f}% lower. The no-change reference is {P[1]['persistence']['rmse']:.3f}. 2x is only the lowest of the five tested values under this one scoring; it is not a calibration or a recommended setting.",
                 f"**The RAW score disagrees.** Pooled RAW RMSE is {P[1]['raw']['rmse']:.3f}, {P[1.2]['raw']['rmse']:.3f}, {P[2]['raw']['rmse']:.3f}, {P[3]['raw']['rmse']:.3f}, {P[4]['raw']['rmse']:.3f}: the best tested value is 1.2x, and 3x and 4x are worse than the original.",
                 "**All rain still enters the soil in every case** (25.088 mm; peak rain 10.034 mm/h is below the matrix K of 38-153 mm/h and pore room is always available), with no runoff, ponding, percolation or drainage. These runs therefore do not constrain or exercise the infiltration-capacity limit: every rain rate stays below K with room available, so changing the matrix K has no effect on intake. This is a test of the unponded mechanism only, not a general statement about infiltration. All differences come from faster layer-to-layer exchange and the evaporation it indirectly feeds.",
                 f"**Depth by depth the answer differs.** Top layer: every multiplier is worse than no change (best {a(top_best, '501'):.2f} vs {ps[0]:.2f}). Middle: best at 2x ({a(2, '502'):.3f}), degrading at 3x and 4x ({a(3, '502'):.3f}, {a(4, '502'):.3f}; original {a(1, '502'):.3f}). Deep: 2x, 3x and 4x become better than no change ({a(2, '503'):.2f}, {a(3, '503'):.2f}, {a(4, '503'):.2f} vs {ps[2]:.2f}), best at 3x.",
                 f"**The large surface pulse remains.** On Jan 23 the 0-10 cm layer rises {F[1]['jan_top']:.2f} points at 1x and {F[4]['jan_top']:.2f} at 4x against 3.06 observed. Higher K never means better in every respect."]),
        ("h1", "2. What was changed"),
        ("p", "The baseline is the frozen JORN run of the original report [7]: one 1 m² column of three 10 cm layers; NEON rain [5]; Hargreaves evaporation demand from NEON air temperature [6] (the native EcoHyD-style family of Hargreaves and Samani [3], not an exact FAO-56 implementation); initial water from the NEON soil sensors [4]; diffusion coefficient d = 0.0010239461376024856; evaporation factor 1; stress value 0.2017; transpiration zero; open bottom; gravity on; matrix-only infiltration; no plants, wind, sediment, bed or routing. Soil properties are the approved Saxton-Rawls [2] surrogates (the density and air-entry floors are this project's choices, not published recommendations). The Tietjen et al. [1] gradient exchange scales with d x K, so a larger K acts like a larger d for the layers it joins; the two cannot be separated here."),
        ("p", "Each new case multiplies ten conductivity entries once by the factor, relative to the ORIGINAL (not compounded on 1.2x): the three layer Ks records (mm/h and m/s), the top matrix Ks, the matrix K and both interface K. I compared each sealed configuration with the original: exactly ten numeric entries differ, each by a ratio equal to the factor. Retention, field capacity (0.395-0.399), saturation, suction (0.166 m), the local-K factor (0.31635738822500753), d, E, stress, T, initial water, forcing, clocks and source pins are identical. The physics reads the matrix K and the two interface K directly; the configuration check also reads the top matrix Ks and requires matrix K = local factor x Ks, and the interface K are harmonic means of the derived local K of adjacent layers. The layer Ks records are user-perturbed source estimates (not new Saxton-Rawls predictions) that act only through those derived K values, so there is no second or double multiplication."),
        ("tab", "params", "Table 1. Conductivities (mm/h) by multiplier; the first three rows are source estimates, the last three are direct inputs of the run."),
        ("p", "**Clock.** As in the earlier runs this is a daily-weather approximation: one 86,400 s inter-event step per nominal day plus the observed storm sub-steps (27.5 h in total, so the physical clock ends at 161.146 days against 160 nominal days). Evaporation and transpiration are off during rain. It is not an exact hourly field clock. Sensor values after day 0 never drive the model."),
        ("h1", "3. Water: what the multipliers change"),
        ("tab", "budget", "Table 2. Budget and layer-exchange results by multiplier."),
        ("p", f"Because every case takes in all 25.088 mm, the multipliers cannot add water. Faster exchange lets the middle layer re-supply the dry top layer, so net upward top-middle exchange rises from {F[1]['top_mid_up']:.2f} to {F[4]['top_mid_up']:.2f} mm and evaporation from {B[1]['evap']:.2f} to {B[4]['evap']:.2f} mm; final storage falls from {B[1]['final']:.2f} to {B[4]['final']:.2f} mm. In the deep layer, faster downward exchange speeds and enlarges the early gain (peak {F[1]['peak_gain_mm']:.2f} to {F[4]['peak_gain_mm']:.2f} mm, reached on {F[1]['peak_date'][5:]} at 1x and {F[4]['peak_date'][5:]} at 4x; nominal day, closing the next day at 07:00 UTC) and the later upward return is faster too, so the deep layer ends drier ({F[1]['final'][2]:.2f} to {F[4]['final'][2]:.2f} vol. %). The last deep sensor reading is {F[1]['obs_deep_last']:.2f}% (a point value; the model's layer mean starts {off[2] * 100:+.2f} points from its point reading). A stronger exchange can therefore compensate for processes the model omits (deep transpiration, bottom loss), which is why a better deep score is not evidence of physically correct K."),
        ("fig", "fig1_raw_overlay.png", "Figure 1. RAW comparison at the nominal daily closes: five model runs (colours and dash styles in the legend), black open circles for sensor point values, triangles for day-0 values. Missing sensor values are left blank (no interpolation). The last shallow-sensor value used is at the 2024-03-15 07:00 UTC close."),
        ("fig", "fig2_difference.png", "Figure 2. Modelled water content minus the original 1x run, in percentage points (a change in volumetric water content, not a level). Shaded bands are nominal days with rain."),
        ("h1", "4. Comparison with the sensors"),
        ("p", f"Two scorings are used. **RAW** compares model layer means directly with sensor point values. **Aligned** removes one constant start-up offset per depth, fixed at day 0, because a 10 cm layer mean differs from a point reading (offsets {off[0] * 100:+.2f}, {off[1] * 100:+.2f} and {off[2] * 100:+.2f} points at 5.8, 15.8 and 25.8 cm). The offsets are unchanged by this study; model water is never altered by the alignment and nothing is re-aligned after storms. Pooled scores weight every observation equally (n = 56, 147 and 79 at the three depths), so the middle sensor dominates."),
        ("tab", "scores", "Table 3. RMSE in percentage points (100 x m³/m³). Aligned and RAW rows are shown for every depth; no-change = holding the day-0 sensor value."),
        ("fig", "fig4_rmse_vs_factor.png", "Figure 3. RMSE against the conductivity multiplier. The multipliers (1, 1.2, 2, 3, 4) are plotted as equally spaced categories and the lines only guide the eye: they are not a continuous fit or minimization. The dotted line is the POOLED no-change reference (1.799) and applies to the black pooled curve only; the per-depth no-change values (0.881, 2.343, 0.886 points at 5.8, 15.8, 25.8 cm) are in Table 3. The pooled aligned and RAW curves have different minima (2x and 1.2x); no shallow value beats its own no-change value."),
        ("p", f"The scorings answer different questions, and no multiplier is better on everything. Aligned scoring gives its lowest pooled RMSE at 2x because the combined per-observation errors are smallest there, mainly through the 147 middle values (the middle sensor is best at 2x, the deep sensor best at 3x and the shallow sensor best at 4x, although every shallow value is worse than no change); RAW favours 1.2x because the middle layer is already drier than its point reading (RAW bias {S[1]['502']['raw']['bias']:+.2f} at 1x) and stronger exchange dries it further (RAW middle RMSE rises steadily from {S[1]['502']['raw']['rmse']:.2f} to {S[4]['502']['raw']['rmse']:.2f} while RAW deep falls from {S[1]['503']['raw']['rmse']:.2f} to {S[4]['503']['raw']['rmse']:.2f}). Beyond 2x the middle sensor degrades (aligned {a(2, '502'):.3f} to {a(4, '502'):.3f}) while the deep sensor is best at 3x ({a(3, '503'):.3f}) and the shallow sensor keeps improving slowly ({a(1, '501'):.2f} to {a(4, '501'):.2f}) without approaching no change. The old d and stress were fitted jointly to 53 dry-period sensor values from 2025 (JORN and MOAB); the 2024 data were not used in that fit, but selecting a multiplier from these 2024 scores would be exploratory, not independent qualification. There is no significance test, no R² and no claim of physical validation of any K."),
        ("fig", "fig3_rain_pulse_zoom.png", "Figure 4. Jan 20 - Feb 4 around the 10 mm Jan 23 rain. Changes since day 0 with the fixed day-0 offset (no re-offset after the storm). Rain bars cover the nominal rainfall day; model states and sensor values are plotted at the daily close that follows that day."),
        ("p", f"Around the Jan 23 storm (Figure 4) the surface pulse is almost unchanged: the 0-10 cm layer rises {F[1]['jan_top']:.2f}, {F[1.2]['jan_top']:.2f}, {F[2]['jan_top']:.2f}, {F[3]['jan_top']:.2f} and {F[4]['jan_top']:.2f} points against 3.06 at the 5.8 cm sensor, so even 4x trims it by only {F[1]['jan_top'] - F[4]['jan_top']:.2f} points. The 72 h middle-layer response (observed +1.06) is {F[1]['jan_mid72']:+.3f}, {F[1.2]['jan_mid72']:+.3f}, {F[2]['jan_mid72']:+.3f}, {F[3]['jan_mid72']:+.3f} and {F[4]['jan_mid72']:+.3f}: it moves toward the observation as K grows (4x reaches about half of it) but stays below it at every multiplier, so timing and magnitude discrepancies in the middle-layer response remain. Two untested explanations for the surface pulse remain: infiltration that is too deep or too large, and immediate mixing of a whole 10 cm layer compared with a point at 5.8 cm (a constant day-0 offset cannot remove that difference after rain). Other persistent limits are the high surrogate field capacity that blocks gravity flow, no transpiration, a thin 30 cm column with no capillary rise or lateral flow, and surrogate properties."),
        ("h1", "5. Verification and run time"),
        ("bul", ["For 2x, 3x and 4x the NumPy and Numba results are identical (I confirmed row-by-row equality here), and the source pair is guarded before and after each run. A 900 s storm step changes layer water by at most 9.1e-6, 1.3e-5 and 1.7e-5 m³/m³ (0.0009, 0.0013 and 0.0017 points). Root's independent 90-digit calculation agrees to better than 1e-17 m (below 0.06 of the native one-step bound) and the native closure bounds pass in all five cases (Table 4).",
                 "Input-only experiment on unchanged physics code: no new checkpoint or restart test, no rerun of the 77-test suite, no GPU, no large-domain or coupled-model acceptance."]),
        ("tab", "checks", "Table 4. Run checks. Times are seconds from separate fresh runs on a shared host with other jobs (empty caches): not a controlled speed comparison."),
        ("p", "Integration takes about 0.04 s in every case. The fresh-command time is dominated by one-time compilation and the source-verified bootstrap and says nothing about solver speed or the effect of K. One 1 m² CPU column: no scaling claim."),
        ("h1", "6. Limits"),
        ("bul", ["Prescribed sensitivities: no calibration, pedotransfer prediction, default change or adoption; K and d remain confounded empirical settings and the 2x minimum is only the lowest of five tested values under one scoring.", "Rain is too light to test ponding, suction, gravity drainage or K as an intake limit.",
                 "No transpiration or plants (a general limitation for spring), point sensors versus layer means, a daily clock, and the 030 vs 020 air-temperature height difference inherited from the original run."]),
        ("h1", "References"), ("refs", REFS),
        ("h1", "Appendix. Identity (compact)"),
        ("p", "Sources (CPU, Root's source-verified bootstrap): SYRUP package 180 files, SHA-256 37d89f23dbf60626c85f85ea5dd6d1734d3bc7a4aa61d8f2fdfc29f237bcfbd0; MAPLE package 412 files, SHA-256 fbc504bbcd1ec0c02e6a96089134a06187532455ef7ba5e6f146801c8313c02e; ten frozen helper and test files unchanged. Baseline main 483dc3d; production unchanged. Case-file hashes: " + "; ".join(f"{NAME[f].split(' ')[0]} {sha(DIRS[f] / 'case_numba.json')[:16]}..." for f in FACT) + ". Full hashes, the parameter differences and daily trajectories of all five runs are in tables/provenance_seal.csv, tables/case_hashes.csv, tables/parameter_diff.csv and tables/daily_trajectories.csv. Raw NEON files are not copied.")]


# ------------------------------------------------------------------------------------------------ docx
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
    tt.font.size = Pt(8)
    tt.paragraph_format.space_after = Pt(0)
    tt.paragraph_format.line_spacing = 1.0


def add_table(doc, hd, rows, width_in, caption):
    cp = doc.add_paragraph(style="RepCaption")
    runs(cp, caption)
    cp.paragraph_format.keep_with_next = True
    cp.paragraph_format.space_after = Pt(3)
    t = doc.add_table(rows=1, cols=len(hd))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    wts = [min(max(len(str(h)) * 0.4, max(len(str(r[i])) for r in rows), 5), 34) for i, h in enumerate(hd)]
    ws = [Inches(width_in * w / sum(wts)) for w in wts]
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


def to_docx(B, T, path):
    doc = Document()
    setup_styles(doc)
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Inches(8.5), Inches(11)
    sec.left_margin = sec.right_margin = Inches(0.9)
    sec.top_margin = sec.bottom_margin = Inches(0.8)
    fp = sec.footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    fr = fp.add_run("SYRUP JORN 160-day comparison: all K and Ks x1, 1.2, 2, 3, 4 · prescribed sensitivity, not a recalibration · page ")
    fr.font.size = Pt(8)
    add_field(fp, "PAGE")
    for b in B:
        k = b[0]
        if k == "title":
            doc.add_paragraph(b[1], style="RepTitle")
        elif k == "sub":
            doc.add_paragraph(b[1], style="RepSubtitle")
        elif k == "h1":
            doc.add_heading(b[1], level=1)
        elif k == "p":
            runs(doc.add_paragraph(), b[1])
        elif k == "bul":
            for x in b[1]:
                runs(doc.add_paragraph(style="List Bullet"), x)
        elif k == "tab":
            hd, rows = T[b[1]]
            add_table(doc, hd, rows, 6.7, b[2])
        elif k == "fig":
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.keep_with_next = True
            p.add_run().add_picture(str(FIG / b[1]), width=Inches(6.0))
            runs(doc.add_paragraph(style="RepCaption"), b[2])
        elif k == "refs":
            for n, c, u in b[1]:
                p = doc.add_paragraph()
                p.paragraph_format.left_indent = Inches(0.25)
                p.paragraph_format.first_line_indent = Inches(-0.25)
                p.paragraph_format.space_after = Pt(2)
                p.add_run(f"[{n}] ").bold = True
                p.add_run(f"{c} {u}".strip()).font.size = Pt(9)
    doc.core_properties.title = "SYRUP JORN 160-day comparison: all K and Ks x1, 1.2, 2, 3, 4"
    doc.core_properties.author = "SYRUP project"
    doc.save(path)


def verify_docx(path, B):
    z = zipfile.ZipFile(path)
    names = z.namelist()
    media = [n for n in names if n.startswith("word/media/")]
    xml = z.read("word/document.xml").decode()
    ntab, nhdr = xml.count("<w:tbl>"), xml.count("<w:tblHeader")
    nfig, ntabs = sum(1 for b in B if b[0] == "fig"), sum(1 for b in B if b[0] == "tab")
    pagef = "PAGE" in "".join(z.read(n).decode(errors="ignore") for n in names if "footer" in n)
    text = re.sub(r"<[^>]+>", " ", xml)
    assert z.testzip() is None and len(media) == nfig == 4 and ntab == ntabs == nhdr and pagef
    assert not re.search(r"TODO|TBD|XXX|placeholder|DRAFT|\bnan\b", text)
    return {"zip_ok": True, "media": len(media), "tables": ntab, "header_rows_repeat": nhdr, "paragraphs": xml.count("<w:p>") + xml.count("<w:p "), "words": len(text.split()), "bytes": Path(path).stat().st_size, "sha256": sha(path)}


def to_markdown(B, T):
    out = []
    for b in B:
        k = b[0]
        if k == "title":
            out.append(f"# {b[1]}\n")
        elif k == "sub":
            out.append(f"*{b[1]}*\n")
        elif k == "h1":
            out.append("## " + b[1] + "\n")
        elif k == "p":
            out.append(b[1] + "\n")
        elif k == "bul":
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
    S, B, F, C = check(R)
    for p in list(TAB.glob("*.csv")) + list(DATA.glob("*")):
        p.unlink()
    fig_raw(R)
    fig_diff(R)
    fig_zoom(R)
    fig_scores(S)
    T = tables(R, S, B, F, C)
    for name, (hd, rows) in T.items():
        csvw(TAB / f"{name}.csv", hd, rows)
    csvw(TAB / "parameter_diff.csv", ["factor", "field", "original", "scaled", "ratio"], [[f, d[0], repr(d[1]), repr(d[2]), repr(d[2] / d[1])] for f, ds in R["diffs"].items() for d in ds])
    csvw(TAB / "provenance_seal.csv", ["file", "sha256"], [[k, v] for k, v in R["seal"].items()])
    csvw(TAB / "case_hashes.csv", ["factor", "config_sha256_in_result", "case_numba_json_sha256"], [[f, R[f]["res"]["config_sha256"], sha(DIRS[f] / "case_numba.json")] for f in FACT])
    hdr = ["day_index", "date_utc_minus7", "nominal_close_utc", "physical_t_s", "rain_mm", "pet_mm"] + [f"evap_mm_x{f}" for f in FACT] + [f"theta_l{k}_x{f}" for f in FACT for k in (1, 2, 3)] + ["obs_501", "obs_502", "obs_503"]
    rows = []
    for i, r0 in enumerate(R[1]["res"]["rows"]):
        rows.append([r0["day_index"], r0["date"], r0["nominal_close_utc"], r0["physical_t_s"], 1000 * r0["rain_m3"], r0["pet_mm_day"]] + [1000 * R[f]["res"]["rows"][i]["evaporation_m3"] for f in FACT]
                    + [R[f]["res"]["rows"][i]["layer_theta"][k] for f in FACT for k in range(3)] + ["" if v is None else v for v in r0["obs_theta"]])
    csvw(TAB / "daily_trajectories.csv", hdr, rows)
    for f in FACT:
        tag = str(f).replace(".", "p")
        shutil.copyfile(DIRS[f] / "case_numba.json", DATA / f"case_numba_x{tag}.json")
        shutil.copyfile(DIRS[f] / "root_result_audit.json", DATA / f"root_result_audit_x{tag}.json")
        shutil.copyfile(DIRS[f] / "actual_numba/timing.json", DATA / f"timing_numba_x{tag}.json")
        if f in (2, 3, 4):
            shutil.copyfile(DIRS[f] / "parameter_changes.json", DATA / f"parameter_changes_x{tag}.json")
            shutil.copyfile(DIRS[f] / "root_decimal_verification.json", DATA / f"root_decimal_verification_x{tag}.json")
    shutil.copyfile(TASK / "comparison.json", DATA / "comparison.json")
    shutil.copyfile(TASK / "root_parameter_validation.json", DATA / "root_parameter_validation.json")
    blk = blocks(R, S, B, F, C)
    to_docx(blk, T, DOCX)
    info = verify_docx(DOCX, blk)
    (HERE / "report.md").write_text(to_markdown(blk, T), newline="\n")
    (HERE / "verification.md").write_text("# Verification record\n\nBefore building, every sealed file used was re-hashed against `root_run_seal.json`. From the actual rows of all five runs the builder recomputed: per-depth and pooled RAW, aligned and no-change RMSE/bias/counts (matches Root's comparison.json), budgets, "
                                          "all-zero runoff/percolation/drainage/ponding/onset counts, the 900 s maximum differences, NumPy = Numba row equality for 2x/3x/4x, maximum modelled water content below field capacity, the pooled improvement percentages, and the exact configuration "
                                          "difference of each scaled case from the original (ten numeric entries x the factor, nothing else). Static DOCX checks: zip integrity, one media file per figure (4), table count equals header-row count, footer page field, no placeholder or DRAFT text. "
                                          "No Office renderer was available, so rendered pagination and legibility were not checked by the builder.\n\n```\n" + json.dumps(info, indent=1) + "\n```\n", newline="\n")
    man = {"docx": {"file": DOCX.name, **info}, "figures": {p.name: sha(p) for p in sorted(FIG.glob("*.png"))}, "tables": {p.name: sha(p) for p in sorted(TAB.iterdir())}, "data": {p.name: sha(p) for p in sorted(DATA.iterdir())},
           "builder": sha(Path(__file__)), "report_md": sha(HERE / "report.md"), "sealed_inputs_checked": {str(p): R["seal"][str(p)] for _, p in R["pending"]}}
    (HERE / "artifact_manifest.json").write_text(json.dumps(man, indent=1) + "\n", newline="\n")
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
