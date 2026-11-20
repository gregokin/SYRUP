"""Builds the JORN 160-day all-K/Ks +20% comparison report: figures, CSV tables, editable Markdown, verification note, manifest and the Word document.

    PYTHONPATH=<tasks>/site_hydraulics_interface_comparison_20261117/report_tools MPLCONFIGDIR=/tmp/syrup-k20-report PYTHONDONTWRITEBYTECODE=1 /home/okin/MAPLE/.venv/bin/python -B build_report.py

DOCUMENTS ONLY. Reads Root's sealed results (every file used is re-hashed against `root_run_seal.json` first); declare other locations with JORN_K20_TASK (this task folder) and JORN_BASE_TASK (the original 160-day task).
It is not portable without those inputs. It runs no model, fits nothing, uses no network or GPU and writes only inside its own folder.
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
NEW = Path(os.environ.get("JORN_K20_TASK", str(HERE.parent)))
OLD = Path(os.environ.get("JORN_BASE_TASK", str(NEW.parent / "jorn_160_day_run_20261119")))
FIG, TAB, DATA = HERE / "figures", HERE / "tables", HERE / "data"
DOCX = HERE / "SYRUP_JORN_160day_K_Ks_plus20_comparison_20261120.docx"
LV = ["501", "502", "503"]
DEPTH = {"501": "5.8 cm sensor (layer 0-10 cm)", "502": "15.8 cm sensor (layer 10-20 cm)", "503": "25.8 cm sensor (layer 20-30 cm)"}
GREEN, BLUE, GREY = "#2E8B57", "#1F5FA8", "#777777"
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
NEW_USED = ["case_numba.json", "parameter_changes.json", "root_parameter_validation.json", "comparison.json", "root_result_audit.json", "root_decimal_verification.json", "frozen_helper_manifest.json", "actual_numba/result.json",
            "actual_numba/timing.json", "actual_numpy/result.json", "actual_numpy/timing.json", "actual_numba_dt900/result.json", "actual_numba_dt900/timing.json", "actual_numba_runs/result.json", "actual_numpy_runs/result.json",
            "actual_numba_dt900_runs/result.json"]
OLD_USED = ["case_numba.json", "actual_numba/result.json", "actual_numba/timing.json", "actual_numba_runs/result.json", "root_result_audit.json"]


def load():
    seal = json.loads((NEW / "root_run_seal.json").read_text())
    pending = []
    for base, names in ((NEW, NEW_USED), (OLD, OLD_USED)):
        for rel in names:
            key = str(base / rel)
            assert key in seal, f"{key} is not in root_run_seal.json"
            assert sha(base / rel) == seal[key], f"{key} differs from root_run_seal.json"
            pending.append((base / rel, key))
    J = lambda b, r: json.loads((b / r).read_text())
    R = {"seal": seal, "new": J(NEW, "actual_numba/result.json"), "old": J(OLD, "actual_numba/result.json"), "np": J(NEW, "actual_numpy/result.json"), "d900": J(NEW, "actual_numba_dt900/result.json"),
         "cfg_new": J(NEW, "case_numba.json"), "cfg_old": J(OLD, "case_numba.json"), "chg": J(NEW, "parameter_changes.json"), "val": J(NEW, "root_parameter_validation.json"), "cmp": J(NEW, "comparison.json"),
         "audit": J(NEW, "root_result_audit.json"), "dec": J(NEW, "root_decimal_verification.json"), "tm": {k: J(NEW, f"actual_{k}/timing.json") for k in ("numba", "numpy", "numba_dt900")},
         "wall": {k: J(NEW, f"actual_{k}_runs/result.json") for k in ("numba", "numpy", "numba_dt900")}, "wall_old": J(OLD, "actual_numba_runs/result.json"), "tm_old": J(OLD, "actual_numba/timing.json"),
         "audit_old": J(OLD, "root_result_audit.json"), "pending": pending}
    return R


def prep(R):
    for tag in ("old", "new"):
        rows = R[tag]["rows"]
        R[tag + "_theta"] = np.array([r["layer_theta"] for r in rows])
    R["obs"] = np.array([[np.nan if v is None else v for v in r["obs_theta"]] for r in R["new"]["rows"]])
    assert np.array_equal(np.nan_to_num(R["obs"], nan=-1), np.nan_to_num(np.array([[np.nan if v is None else v for v in r["obs_theta"]] for r in R["old"]["rows"]]), nan=-1))
    R["theta0"] = np.array(R["new"]["plan"]["initial"]["layer_theta"])
    assert R["theta0"].tolist() == R["old"]["plan"]["initial"]["layer_theta"]
    R["obs0"] = np.array(R["new"]["plan"]["initial"]["point_theta"])
    R["close"] = [dt.date.fromisoformat(r["date"]) + dt.timedelta(days=1) for r in R["new"]["rows"]]
    R["start"] = dt.date.fromisoformat(R["new"]["rows"][0]["date"])
    return R


def stats(e):
    return {"n": int(e.size), "rmse": 100 * math.sqrt(float(np.mean(e**2))), "bias": 100 * float(e.mean())}


def scores(R, tag):
    th, ob = R[tag + "_theta"], R["obs"]
    out = {}
    allr, alla, allp = [], [], []
    for j, v in enumerate(LV):
        ok = np.isfinite(ob[:, j])
        raw = th[ok, j] - ob[ok, j]
        al = (th[ok, j] - R["theta0"][j]) - (ob[ok, j] - R["obs0"][j])
        ps = -(ob[ok, j] - R["obs0"][j])
        out[v] = {"raw": stats(raw), "aligned": stats(al), "persistence": stats(ps)}
        allr.append(raw), alla.append(al), allp.append(ps)
    out["pooled"] = {"raw": stats(np.concatenate(allr)), "aligned": stats(np.concatenate(alla)), "persistence": stats(np.concatenate(allp))}
    return out


def check(R):
    """Recompute headline numbers from the actual rows and compare with Root's comparison/audit."""
    c = R["cmp"]
    S = {t: scores(R, t) for t in ("old", "new")}
    for row in c["scores"]:
        kind = "raw" if row["kind"] == "raw" else "aligned"
        assert row["n"] == S["new"][row["sensor"]][kind]["n"]
        assert math.isclose(row["baseline_rmse_pp"], S["old"][row["sensor"]][kind]["rmse"], rel_tol=1e-9) and math.isclose(row["new_rmse_pp"], S["new"][row["sensor"]][kind]["rmse"], rel_tol=1e-9), row
    p = c["pooled"]
    assert math.isclose(p["baseline_aligned_rmse_pp"], S["old"]["pooled"]["aligned"]["rmse"], rel_tol=1e-9) and math.isclose(p["new_aligned_rmse_pp"], S["new"]["pooled"]["aligned"]["rmse"], rel_tol=1e-9)
    assert math.isclose(p["raw_new_rmse_pp"], S["new"]["pooled"]["raw"]["rmse"], rel_tol=1e-9) and S["new"]["pooled"]["aligned"]["n"] == 282
    assert math.isclose(p["relative_rmse_improvement_percent"], 100 * (1 - S["new"]["pooled"]["aligned"]["rmse"] / S["old"]["pooled"]["aligned"]["rmse"]), rel_tol=1e-9)
    B = {}
    for t in ("old", "new"):
        rows = R[t]["rows"]
        B[t] = {"initial": 22.9424, "rain": 1000 * sum(r["rain_m3"] for r in rows), "intake": 1000 * sum(r["intake_m3"] for r in rows), "evap": 1000 * sum(r["evaporation_m3"] for r in rows),
                "final": 1000 * rows[-1]["storage_m3"], "runoff": 1000 * sum(r["runoff_pool_m3"] + r["surface_return_m3"] for r in rows), "drain": 1000 * sum(r["drainage_m3"] for r in rows),
                "perc": sum(abs(x) for r in rows for x in r["percolation_m3"]), "pond": sum(r["n_ponded_steps"] for r in rows), "onsets": sum(r["n_onsets"] for r in rows)}
        assert abs(B[t]["initial"] + B[t]["intake"] - B[t]["evap"] - B[t]["final"]) < 1e-9 and B[t]["runoff"] == 0 and B[t]["drain"] == 0 and B[t]["perc"] == 0 and B[t]["pond"] == 0 and B[t]["onsets"] == 0
        assert math.isclose(B[t]["intake"], B[t]["rain"], rel_tol=1e-12)
    assert math.isclose(B["new"]["evap"], c["budgets_mm"]["plus20"]["evaporation"], rel_tol=1e-12) and math.isclose(B["old"]["evap"], c["budgets_mm"]["baseline"]["evaporation"], rel_tol=1e-12)
    # exact parameter diff of the two sealed configs: 10 conductivity fields x 1.2, nothing else
    diffs = []

    def walk(x, y, p=""):
        if isinstance(x, dict):
            for k in x.keys() | y.keys():
                if k in x and k in y:
                    walk(x[k], y[k], f"{p}.{k}")
                elif p != ".provenance" or k != "conductivity_sensitivity":
                    raise AssertionError(f"key mismatch {p}.{k}")
        elif isinstance(x, list):
            assert len(x) == len(y)
            for i, (u, v) in enumerate(zip(x, y)):
                walk(u, v, f"{p}[{i}]")
        elif x != y:
            diffs.append((p, x, y))
    walk(R["cfg_old"], R["cfg_new"])
    num = [d for d in diffs if d[0] != ".name"]
    assert len(num) == 10 and all(math.isclose(d[2] / d[1], 1.2, rel_tol=1e-15) for d in num) and [d[0] for d in diffs if d[0] == ".name"] == [".name"]
    assert {d[0] for d in num} == {f".provenance.layer_properties[{i}].{k}" for i in range(3) for k in ("Ks_mm_h", "Ks_m_s")} | {".layers.interface_conductivity_m_s[0]", ".layers.interface_conductivity_m_s[1]", ".matrix.ks_top_m_s", ".matrix.conductivity_m_s"}
    R["diffs"] = num
    for t in ("old", "new"):
        assert R[t]["complete"] and R[t]["physical_end_t_s"] == 13923000.0 and R[t]["ledger"]["n_matrix_steps"] == 55 and R[t]["ledger"]["n_soil_steps"] == 215
        assert R[t]["rows"][-1]["nominal_close_utc"] == "2024-06-26T07:00:00Z"
    # exact backend agreement and the timestep control
    assert R["np"]["rows"] == R["new"]["rows"]
    d900 = np.array([r["layer_theta"] for r in R["d900"]["rows"]])
    assert math.isclose(float(np.abs(d900 - R["new_theta"]).max()), c["dt900_max_theta_diff"], rel_tol=1e-9)
    R["S"], R["B"] = S, B
    return S, B


def flux(R, t):
    rows = R[t]["rows"]
    d = np.array([r["diffusion_m3"] for r in rows]) * 1000                        # mm, positive downward: (days, 2)
    th = R[t + "_theta"]
    pk = int(np.argmax(th[:, 2]))
    jan = next(i for i, r in enumerate(rows) if r["date"] == "2024-01-23")
    w = np.array([r["water_m"] for r in rows]) * 1000
    return {"peak_idx": pk, "peak_date": rows[pk]["date"], "peak_close": R["close"][pk], "peak_theta": th[pk, 2] * 100, "peak_gain_mm": w[pk, 2] - R["theta0"][2] * 100,
            "mid_deep_down": float(d[d[:, 1] > 0, 1].sum()), "mid_deep_up": float(-d[d[:, 1] < 0, 1].sum()), "top_mid_up": float(-d[:, 0].sum()), "final_theta": (th[-1] * 100).tolist(),
            "jan_top": (th[jan, 0] - th[jan - 1, 0]) * 100, "jan_mid72": (th[jan + 2, 1] - th[jan - 1, 1]) * 100, "jan_mid24": (th[jan, 1] - th[jan - 1, 1]) * 100, "jan_idx": jan,
            "rev_idx": int(next(i for i, v in enumerate(d[:, 1]) if v < 0)), "rev_date": rows[int(next(i for i, v in enumerate(d[:, 1]) if v < 0))]["date"]}


# ------------------------------------------------------------------------------------------------ figures
def style(ax):
    ax.grid(alpha=0.25)
    ax.tick_params(labelsize=8)


def rainspans(ax, R):
    for r, c in zip(R["new"]["rows"], R["close"]):
        if r["storm_rows"]:
            ax.axvspan(c - dt.timedelta(days=1), c, color="#BFD9F0", alpha=0.55, lw=0, zorder=0)


def fig_raw(R):
    fig, axs = plt.subplots(3, 1, figsize=(7.2, 6.4), sharex=True)
    x = [R["start"]] + R["close"]
    for j, ax in enumerate(axs):
        ax.plot(x, np.r_[R["theta0"][j], R["old_theta"][:, j]] * 100, color=GREEN, lw=1.4, label="baseline model")
        ax.plot(x, np.r_[R["theta0"][j], R["new_theta"][:, j]] * 100, color=BLUE, lw=1.3, ls="--", label="all K and Ks +20%")
        ax.scatter(R["close"], R["obs"][:, j] * 100, facecolors="none", edgecolors="black", s=13, lw=0.8, label="sensor point value")
        ax.scatter([R["start"]], [R["theta0"][j] * 100], marker="^", color=GREEN, s=44, zorder=5, label="initial model layer mean")
        ax.scatter([R["start"]], [R["obs0"][j] * 100], marker="^", facecolors="none", edgecolors="black", s=44, zorder=5, label="initial sensor value")
        rainspans(ax, R)
        ax.set_ylabel("water content (vol. %)", fontsize=8.5)
        ax.set_title(DEPTH[LV[j]], fontsize=9.5, loc="left")
        style(ax)
    axs[-1].xaxis.set_major_locator(mdates.MonthLocator())
    axs[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=7.5, frameon=False)
    fig.suptitle("RAW values at each nominal daily close (shaded = nominal rain days)", fontsize=10)
    fig.tight_layout(rect=(0, 0.09, 1, 0.97))
    fig.savefig(FIG / "fig1_raw_overlay.png", dpi=150)
    plt.close(fig)


def fig_diff(R):
    fig, axs = plt.subplots(3, 1, figsize=(7.2, 5.6), sharex=True)
    diff = (R["new_theta"] - R["old_theta"]) * 100
    for j, ax in enumerate(axs):
        ax.axhline(0, color=GREY, lw=0.8)
        ax.plot(R["close"], diff[:, j], color=BLUE, lw=1.3)
        rainspans(ax, R)
        ax.set_ylabel("new - baseline\n(percentage points)", fontsize=8)
        ax.set_title(DEPTH[LV[j]], fontsize=9.5, loc="left")
        style(ax)
    axs[-1].xaxis.set_major_locator(mdates.MonthLocator())
    axs[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.suptitle("Effect of the +20% change on the modelled water content (shaded = rain days)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIG / "fig2_difference.png", dpi=150)
    plt.close(fig)


def fig_zoom(R):
    idx = range(2, 18)
    x = [R["close"][i] for i in idx]
    fig, axs = plt.subplots(4, 1, figsize=(7.2, 6.6), sharex=True, gridspec_kw={"height_ratios": [0.9, 1.4, 1.4, 1.4]})
    axs[0].bar([R["close"][i] - dt.timedelta(days=0.5) for i in idx], [R["new"]["rows"][i]["observed_rain_mm"] for i in idx], width=0.8, color=BLUE)
    axs[0].set_ylabel("rain (mm/day)", fontsize=8.5)
    axs[0].set_title("Observed rain per nominal day", fontsize=9.5, loc="left")
    style(axs[0])
    for j in range(3):
        ax = axs[j + 1]
        ax.axhline(0, color=GREY, lw=0.8, ls="--")
        ax.plot(x, [(R["old_theta"][i, j] - R["theta0"][j]) * 100 for i in idx], color=GREEN, marker="o", ms=3, lw=1.2, label="baseline model change since day 0")
        ax.plot(x, [(R["new_theta"][i, j] - R["theta0"][j]) * 100 for i in idx], color=BLUE, marker="s", ms=3, lw=1.2, ls="--", label="+20% model change since day 0")
        ax.scatter(x, [(R["obs"][i, j] - R["obs0"][j]) * 100 for i in idx], facecolors="none", edgecolors="black", s=22, label="sensor change since day 0")
        ax.set_ylabel("change (pct. points)", fontsize=8.5)
        ax.set_title(DEPTH[LV[j]], fontsize=9.5, loc="left")
        style(ax)
    axs[-1].xaxis.set_major_locator(mdates.DayLocator(interval=2))
    axs[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    h, l = axs[1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=7.5, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(FIG / "fig3_rain_pulse_zoom.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ tables
def f(x, n=3):
    return f"{x:.{n}f}"


def tables(R, S, B, F):
    T = {}
    P = {x["parameter"]: x for x in R["cmp"]["parameter_rows"]}
    rows = []
    for name in ("Ks layer1", "Ks layer2", "Ks layer3", "effective K layer1", "effective K layer2", "effective K layer3", "matrix infiltration K", "interface K1", "interface K2"):
        p = P[name]
        label = {"Ks layer1": "Ks, layer 0-10 cm", "Ks layer2": "Ks, layer 10-20 cm", "Ks layer3": "Ks, layer 20-30 cm", "effective K layer1": "Effective K, layer 0-10 cm", "effective K layer2": "Effective K, layer 10-20 cm",
                 "effective K layer3": "Effective K, layer 20-30 cm", "matrix infiltration K": "Matrix infiltration K (top layer)", "interface K1": "Interface K, 0-10 / 10-20 cm", "interface K2": "Interface K, 10-20 / 20-30 cm"}[name]
        used = {"Ks": "source estimate; represented by the derived K", "effective K": "derived from Ks; used to form interface K", "matrix": "direct physical input (intake law)", "interface": "direct physical input (exchange law)"}
        key = "matrix" if name.startswith("matrix") else "interface" if name.startswith("interface") else "effective K" if name.startswith("effective") else "Ks"
        rows.append([label, f(p["baseline_mm_h"], 2), f(p["new_mm_h"], 2), "derived from Ks; equals the matrix K" if name == "effective K layer1" else used[key]])
    T["params"] = (["Quantity", "Baseline (mm/h)", "+20% (mm/h)", "Used by the native run as"], rows)
    sc = []
    for v in LV + ["pooled"]:
        o, n = S["old"][v], S["new"][v]
        sc.append([DEPTH[v].split(" (")[0] if v != "pooled" else "All depths pooled", n["raw"]["n"], f"{f(o['raw']['rmse'])} / {f(n['raw']['rmse'])}", f"{f(o['aligned']['rmse'])} / {f(n['aligned']['rmse'])}", f(o["persistence"]["rmse"]),
                   f"{f(o['raw']['bias'], 2)} / {f(n['raw']['bias'], 2)}", f"{f(o['aligned']['bias'], 2)} / {f(n['aligned']['bias'], 2)}"])
    T["scores"] = (["Depth", "n", "RAW RMSE base / +20%", "Aligned RMSE base / +20%", "No-change RMSE", "RAW bias base / +20%", "Aligned bias base / +20%"], sc)
    o, n = B["old"], B["new"]
    T["budget"] = (["Term (mm over 1 m²)", "Baseline", "+20%", "Change"], [["Initial storage", f(o["initial"], 4), f(n["initial"], 4), "0"], ["+ Rain = intake", f(o["intake"], 3), f(n["intake"], 3), "0"], ["- Evaporation", f(o["evap"], 4), f(n["evap"], 4), f"{n['evap'] - o['evap']:+.4f}"],
                                                                         ["- Runoff / percolation / drainage", "0", "0", "0"], ["= Final storage", f(o["final"], 4), f(n["final"], 4), f"{n['final'] - o['final']:+.4f}"],
                                                                         ["Deep-layer peak gain (mm)", f(F["old"]["peak_gain_mm"], 4), f(F["new"]["peak_gain_mm"], 4), f"{F['new']['peak_gain_mm'] - F['old']['peak_gain_mm']:+.4f}"],
                                                                         ["Deep-layer peak (nominal day)", F["old"]["peak_date"], F["new"]["peak_date"], "earlier"], ["Deep layer, final (vol. %)", f(F["old"]["final_theta"][2]), f(F["new"]["final_theta"][2]), f"{F['new']['final_theta'][2] - F['old']['final_theta'][2]:+.3f}"],
                                                                         ["Jan 23 top-layer rise (points)", f(F["old"]["jan_top"], 3), f(F["new"]["jan_top"], 3), f"{F['new']['jan_top'] - F['old']['jan_top']:+.3f}"],
                                                                         ["Jan 23 middle layer, 72 h (points)", f(F["old"]["jan_mid72"], 3), f(F["new"]["jan_mid72"], 3), f"{F['new']['jan_mid72'] - F['old']['jan_mid72']:+.3f}"]])
    tm, wl = R["tm"], R["wall"]
    old = R["tm_old"]
    T["timing"] = (["Run", "Integration (s)", "Operator prep + JIT (s)", "Driver call wall (s)", "Full fresh command (s)"], [
        ["Baseline, Numba", f(old["integration_s"], 4), f(old["operator_prepare_s"], 3), f(R["cmp"]["timing"]["baseline"]["driver_call_wall_s"], 3), f(R["wall_old"]["seconds"], 3)],
        ["+20%, Numba", f(tm["numba"]["integration_s"], 4), f(tm["numba"]["operator_prepare_s"], 3), f(R["cmp"]["timing"]["plus20"]["driver_call_wall_s"], 3), f(wl["numba"]["seconds"], 3)],
        ["+20%, NumPy", f(tm["numpy"]["integration_s"], 4), f(tm["numpy"]["operator_prepare_s"], 3), "-", f(wl["numpy"]["seconds"], 3)],
        ["+20%, Numba, 900 s storm step", f(tm["numba_dt900"]["integration_s"], 4), f(tm["numba_dt900"]["operator_prepare_s"], 3), "-", f(wl["numba_dt900"]["seconds"], 3)]])
    return T


# ------------------------------------------------------------------------------------------------ text
REFS = [(1, "Tietjen, B., Zehe, E., & Jeltsch, F. (2009). Simulating plant water availability in dry lands under climate change: a generic model of two soil layers. Water Resources Research 45, W01418.", "https://doi.org/10.1029/2007WR006589"),
        (2, "Saxton, K. E., & Rawls, W. J. (2006). Soil water characteristic estimates by texture and organic matter for hydrologic solutions. Soil Science Society of America Journal 70, 1569-1578.", "https://doi.org/10.2136/sssaj2005.0117"),
        (3, "Hargreaves, G. H., & Samani, Z. A. (1985). Reference crop evapotranspiration from temperature. Applied Engineering in Agriculture 1, 96-99.", "https://doi.org/10.13031/2013.26773"),
        (4, "NEON (RELEASE-2026). Soil water content and water salinity, DP1.00094.001.", "https://data.neonscience.org/data-products/DP1.00094.001/RELEASE-2026"),
        (5, "NEON (RELEASE-2026). Precipitation, DP1.00045.001.", "https://data.neonscience.org/data-products/DP1.00045.001/RELEASE-2026"),
        (6, "NEON (RELEASE-2026). Single aspirated air temperature, DP1.00002.001.", "https://data.neonscience.org/data-products/DP1.00002.001/RELEASE-2026"),
        (7, "Original report of the baseline run: SYRUP on a real 160-day rain record, JORN, January-June 2024 (SYRUP_JORN_160day_rainfall_comparison_20261119.docx).", "")]


def blocks(R, S, B, F):
    c = R["cmp"]
    hi = float(R["new_theta"].max())
    p = S["new"]["pooled"]
    po = S["old"]["pooled"]
    cut = c["pooled"]["relative_rmse_improvement_percent"]
    o, n = S["old"], S["new"]
    Fo, Fn = F["old"], F["new"]
    K = {x["parameter"]: x for x in c["parameter_rows"]}
    return [
        ("title", "What happens if every conductivity is 20% larger? JORN, January-June 2024"),
        ("sub", "A prescribed sensitivity of the native SYRUP column: every K and Ks raised by exactly 1.2, everything else identical to the baseline 160-day run. This is not a recalibration."),
        ("h1", "1. Summary"),
        ("bul", [f"**Small, mixed change.** The pooled error against 282 sensor values after one fixed start-up offset falls from {po['aligned']['rmse']:.3f} to {p['aligned']['rmse']:.3f} percentage points ({cut:.1f}% lower RMSE). The no-change reference is {po['persistence']['rmse']:.3f}.",
                 f"**All rain still enters the soil.** The matrix infiltration K rises from {K['matrix infiltration K']['baseline_mm_h']:.1f} to {K['matrix infiltration K']['new_mm_h']:.1f} mm/h, but peak rain is only 10.034 mm/h and the soil never lacks pore room, so intake (25.088 mm) is identical. There is no runoff, ponding, percolation or drainage in either run. The change acts only through faster exchange between layers and, indirectly, evaporation.",
                 f"**The surface pulse is not fixed.** On Jan 23 the 0-10 cm layer rises {Fo['jan_top']:.2f} points in the baseline and {Fn['jan_top']:.2f} with +20%; the 5.8 cm sensor rose 3.06.",
                 f"**Middle sensor:** aligned RMSE improves ({o['502']['aligned']['rmse']:.3f} to {n['502']['aligned']['rmse']:.3f}) but RAW RMSE gets worse ({o['502']['raw']['rmse']:.3f} to {n['502']['raw']['rmse']:.3f}). Both views are shown and explained below.",
                 f"**Shallow and deep sensors** remain worse than assuming no change ({n['501']['aligned']['rmse']:.2f} vs {n['501']['persistence']['rmse']:.2f}; {n['503']['aligned']['rmse']:.2f} vs {n['503']['persistence']['rmse']:.2f}). The result does not show that larger K values are physically better."]),
        ("h1", "2. What was changed"),
        ("p", "The baseline is the frozen JORN run described in the original report [7]: one 1 m² column of three 10 cm layers, NEON rain [5] and Hargreaves evaporation demand from NEON air temperature [6] (the native EcoHyD-style family of Hargreaves and Samani [3], not an exact FAO-56 implementation), initial water from the NEON soil sensors [4], diffusion coefficient d = 0.0010239461376024856, evaporation factor 1, stress value 0.2017, transpiration zero, open bottom, gravity on, matrix-only infiltration, no plants, wind, sediment or routing. The soil properties are the approved Saxton-Rawls [2] surrogates (the density and air-entry floors are this project's choices, not published recommendations). The Tietjen et al. [1] gradient exchange scales with d x K, so a 20% larger K acts like a 20% larger d for the layers it joins."),
        ("p", f"Root scaled ten conductivity entries once by 1.2 and nothing else. I re-compared the two sealed configurations: exactly ten numeric entries differ, each by a ratio of 1.2 (three layer Ks records in mm/h and m/s, the top matrix Ks, the matrix K, and both interface K); the retention values, suction (0.166 m), the local-K factor (0.31635738822500753), d, E, stress, initial water, forcing, clocks and source pins are identical. The physics reads only the matrix K and the two interface K. The configuration check, however, also reads the top matrix Ks and requires matrix K = local factor x top Ks, so the Ks and K entries had to be scaled together; the run would be refused if only one were changed. The layer Ks records are source estimates (user-perturbed here, not new Saxton-Rawls predictions) that act through those derived K values, so there is no second or double 1.2 multiplier; the effective K of layers 2 and 3 are derived for display only. No retention curve was recomputed. Field capacity (0.395-0.399) and saturation are unchanged, and modelled water stays far below field capacity (highest {hi:.3f} in the +20% run), so gravity flow and open-bottom outflow are zero in both runs."),
        ("tab", "params", "Table 1. Conductivities (mm/h): source estimates (Ks), values derived from them, and the direct physical inputs of the run. Ks and derived K are scaled together by the same 1.2, once."),
        ("p", "**Clock.** As in the baseline, this is a daily-weather approximation: one 86,400 s inter-event step per nominal day plus the observed storm sub-steps (27.5 h in total, so the physical clock ends at 161.146 days against 160 nominal days). Evaporation and transpiration are off during rain. It is not an exact hourly field clock."),
        ("h1", "3. Water: where the 20% matters"),
        ("tab", "budget", "Table 2. Budget and layer-exchange results (baseline vs +20%)."),
        ("p", f"All 25.088 mm of rain infiltrates in both runs, so the larger matrix K cannot add water. Evaporation rises by {B['new']['evap'] - B['old']['evap']:.3f} mm because faster upward exchange from the middle layer re-supplies the dry top layer (whole-run net upward top-middle exchange {Fo['top_mid_up']:.2f} to {Fn['top_mid_up']:.2f} mm), so final storage is {B['old']['final'] - B['new']['final']:.3f} mm lower. In the deep layer, faster exchange both speeds and strengthens the early downward gain (peak {Fo['peak_gain_mm']:.3f} to {Fn['peak_gain_mm']:.3f} mm, reached on {Fo['peak_date'][5:]} in the baseline and {Fn['peak_date'][5:]} with +20%; nominal day, closing the next day at 07:00 UTC) and the later upward return, so the deep layer ends drier ({Fo['final_theta'][2]:.2f} to {Fn['final_theta'][2]:.2f} vol. %). The same increase works in both directions: whole-run downward and upward middle-deep exchange are {Fo['mid_deep_down']:.3f}/{Fo['mid_deep_up']:.3f} mm (baseline) and {Fn['mid_deep_down']:.3f}/{Fn['mid_deep_up']:.3f} mm (+20%)."),
        ("fig", "fig1_raw_overlay.png", "Figure 1. RAW comparison at the nominal daily closes. Green: baseline; blue dashed: +20%; black open circles: sensor point values; triangles: day-0 values. The two model curves are almost indistinguishable at this scale; missing sensor values are left blank (no interpolation). The last shallow-sensor value used is at the 2024-03-15 07:00 UTC close."),
        ("fig", "fig2_difference.png", "Figure 2. New minus baseline water content, in percentage points (a change in volumetric water content, not a level). The largest absolute changes are about a quarter of a point in the middle and deep layers (maximum 0.232 and 0.236, deep the larger) and 0.071 in the top layer; shaded bands are the nominal days with rain."),
        ("h1", "4. Comparison with the sensors"),
        ("p", "Two scorings are given. **RAW** compares model layer means directly with sensor point values. **Aligned** removes one constant start-up offset per depth, fixed at day 0, because a 10 cm layer mean differs from a point reading (offsets of +0.19, -0.59 and +0.44 points at 5.8, 15.8 and 25.8 cm). The offsets are unchanged by this study; model water is never altered by the alignment and nothing is re-aligned after storms."),
        ("tab", "scores", "Table 3. Scores in percentage points (100 x m³/m³), baseline / +20%. No-change = holding the day-0 sensor value."),
        ("p", f"The middle sensor shows why both views matter. The model's middle layer starts {abs(S['old']['502']['raw']['bias'] - S['old']['502']['aligned']['bias']):.2f} points drier than the point reading; the +20% run lowers the middle layer by a further small amount through faster draw to the top, which removes some of the aligned error (bias {o['502']['aligned']['bias']:+.2f} to {n['502']['aligned']['bias']:+.2f}) while moving the RAW bias from {o['502']['raw']['bias']:+.2f} to {n['502']['raw']['bias']:+.2f}, so RAW RMSE rises. Neither is the more correct scoring; they answer different questions. At the shallow and deep sensors the aligned RMSE drops by only {o['501']['aligned']['rmse'] - n['501']['aligned']['rmse']:.3f} and {o['503']['aligned']['rmse'] - n['503']['aligned']['rmse']:.3f} points, and both stay worse than no change; the pooled gain comes mostly from the 147 middle values. There is no significance test, no R² and no claim that either set of K values is physically validated. The old d and stress values were fitted jointly to 53 dry-period sensor values from 2025 (JORN and MOAB); the 2024 data remain a test only, and nothing was refitted here."),
        ("fig", "fig3_rain_pulse_zoom.png", "Figure 3. Jan 20 - Feb 4 around the 10 mm Jan 23 rain. Changes since day 0 with the fixed day-0 offset (no re-offset after the storm). Rain bars cover the nominal rainfall day; model states and sensor values are plotted at the daily close that follows that day. The 27.5 h of extra physical storm time remains the disclosed daily-weather approximation."),
        ("p", f"Around the Jan 23 storm (Figure 3) the larger K barely changes the top-layer pulse ({Fo['jan_top']:.2f} vs {Fn['jan_top']:.2f} points against 3.06 observed) and brings the middle-layer response only slightly closer to the observed 72 h rise of +1.06 points ({Fo['jan_mid72']:+.3f} baseline, {Fn['jan_mid72']:+.3f} with +20%, closer by {Fn['jan_mid72'] - Fo['jan_mid72']:.3f}); it remains far below the observation, so the larger K does not resolve the lag or the magnitude of the middle-layer response. Increasing every K by 20% therefore does not resolve the large surface pulse. Plausible reasons remain those of the baseline report: a 5.8 cm point versus an instantly mixed 10 cm layer mean, high surrogate field capacity that blocks gravity flow, no transpiration, a 30 cm column with no capillary rise or lateral flow, surrogate properties, and the d x K identifiability (a 20% larger K is similar to a 20% larger d)."),
        ("h1", "5. Verification and run time"),
        ("bul", [f"NumPy and Numba results are exactly identical (all trajectories and fluxes); I confirmed the row-by-row equality here. A 900 s storm step changes the layer water content by at most {c['dt900_max_theta_diff']:.2e} m³/m³ ({100 * c['dt900_max_theta_diff']:.6f} points).",
                 f"Root's independent 90-digit calculation agrees to {R['dec']['cases'][0]['max_water_error_m']:.2e} m ({R['dec']['cases'][0]['max_error_over_native_single_step_bound']:.3f} of the native one-step bound; {R['dec']['cases'][2]['max_water_error_m']:.2e} m, {R['dec']['cases'][2]['max_error_over_native_single_step_bound']:.3f} for the 900 s control). Native budgets and the source guards pass; closure residual {R['new']['cumulative_closure']['soil_residual_m3']:.1e} m³ against a bound of {R['new']['cumulative_closure']['soil_bound_m3']:.2e} m³.",
                 "The input-only change did not rerun the earlier 77-test suite, and no new midpoint-restart check was made for this parameter set; the earlier restart qualification belongs to the original case. No GPU was used."]),
        ("tab", "timing", "Table 4. Measured seconds from separate fresh runs on a shared host (cold caches, other jobs running): not a controlled speed comparison."),
        ("p", "Integration takes about 0.04 s in both runs; the fresh-command time is dominated by one-time compilation and the source-verified bootstrap, so it says nothing about model speed or the effect of the parameter change. One 1 m² column only: no scaling or coupled-model claim."),
        ("h1", "6. Limits"),
        ("bul", ["A prescribed +20% sensitivity: no new calibration, pedotransfer prediction or default change; K, d and the stress value remain conditional, partly degenerate empirical settings.", "Rain was too light to test ponding, suction, gravity drainage or the matrix K as an intake limit.",
                 "No transpiration or plants (a general limitation for spring), point sensors versus layer means, a daily clock, and the 030 vs 020 air-temperature height difference inherited from the baseline."]),
        ("h1", "References"), ("refs", REFS),
        ("h1", "Appendix. Identity (compact)"),
        ("p", f"Sources (CPU, Root's source-verified bootstrap): SYRUP package 180 files, SHA-256 37d89f23dbf60626c85f85ea5dd6d1734d3bc7a4aa61d8f2fdfc29f237bcfbd0; MAPLE package 412 files, SHA-256 fbc504bbcd1ec0c02e6a96089134a06187532455ef7ba5e6f146801c8313c02e; ten frozen helper/test files unchanged. Original main 2591847. Baseline case SHA-256 {sha(OLD / 'case_numba.json')}; +20% case SHA-256 {sha(NEW / 'case_numba.json')}."),
        ("p", "Every sealed file used is listed with its SHA-256 in tables/provenance_seal.csv and artifact_manifest.json; the parameter differences are in tables/parameter_diff.csv; daily trajectories for both runs are in tables/daily_trajectories.csv. Raw NEON files are not copied.")]


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
    for name, sz, col in (("Heading 1", 14, "1F3A5F"),):
        h = st[name]
        h.font.name, h.font.size, h.font.bold = "Calibri", Pt(sz), True
        h.font.color.rgb = RGBColor.from_string(col)
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
    wts = [min(max(len(str(h)) * 0.5, max(len(str(r[i])) for r in rows), 5), 34) for i, h in enumerate(hd)]
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
    fr = fp.add_run("SYRUP JORN 160-day comparison: all K and Ks +20% · prescribed sensitivity, not a recalibration · page ")
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
            p.add_run().add_picture(str(FIG / b[1]), width=Inches(6.2))
            runs(doc.add_paragraph(style="RepCaption"), b[2])
        elif k == "refs":
            for n, c, u in b[1]:
                p = doc.add_paragraph()
                p.paragraph_format.left_indent = Inches(0.25)
                p.paragraph_format.first_line_indent = Inches(-0.25)
                p.paragraph_format.space_after = Pt(2)
                p.add_run(f"[{n}] ").bold = True
                p.add_run(f"{c} {u}".strip()).font.size = Pt(9)
    doc.core_properties.title = "SYRUP JORN 160-day comparison: all K and Ks +20%"
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
    assert z.testzip() is None and len(media) == nfig == 3 and ntab == ntabs == nhdr and pagef
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
    S, B = check(R)
    F = {"old": flux(R, "old"), "new": flux(R, "new")}
    for p in list(TAB.glob("*.csv")) + list(DATA.glob("*")):
        p.unlink()
    fig_raw(R)
    fig_diff(R)
    fig_zoom(R)
    T = tables(R, S, B, F)
    for name, (hd, rows) in T.items():
        csvw(TAB / f"{name}.csv", hd, rows)
    csvw(TAB / "parameter_diff.csv", ["field", "baseline", "plus20", "ratio"], [[d[0], repr(d[1]), repr(d[2]), repr(d[2] / d[1])] for d in R["diffs"]])
    csvw(TAB / "provenance_seal.csv", ["file", "sha256"], [[k, v] for k, v in R["seal"].items()])
    hdr = ["day_index", "date_utc_minus7", "nominal_close_utc", "physical_t_s", "rain_mm", "pet_mm", "base_evap_mm", "plus20_evap_mm", "base_theta_l1", "base_theta_l2", "base_theta_l3", "plus20_theta_l1", "plus20_theta_l2", "plus20_theta_l3", "obs_501", "obs_502", "obs_503"]
    rows = []
    for a, b in zip(R["old"]["rows"], R["new"]["rows"]):
        rows.append([a["day_index"], a["date"], a["nominal_close_utc"], a["physical_t_s"], 1000 * a["rain_m3"], a["pet_mm_day"], 1000 * a["evaporation_m3"], 1000 * b["evaporation_m3"], *a["layer_theta"], *b["layer_theta"], *["" if v is None else v for v in a["obs_theta"]]])
    csvw(TAB / "daily_trajectories.csv", hdr, rows)
    for src, name in ((NEW / "case_numba.json", "case_numba_plus20.json"), (OLD / "case_numba.json", "case_numba_baseline.json"), (NEW / "parameter_changes.json", "parameter_changes.json"), (NEW / "comparison.json", "comparison.json"),
                      (NEW / "root_parameter_validation.json", "root_parameter_validation.json"), (NEW / "root_result_audit.json", "root_result_audit.json"), (NEW / "root_decimal_verification.json", "root_decimal_verification.json"),
                      (NEW / "actual_numba/timing.json", "timing_numba_plus20.json"), (NEW / "actual_numpy/timing.json", "timing_numpy_plus20.json"), (NEW / "actual_numba_dt900/timing.json", "timing_numba_dt900_plus20.json"),
                      (NEW / "frozen_helper_manifest.json", "frozen_helper_manifest.json")):
        shutil.copyfile(src, DATA / name)
    blk = blocks(R, S, B, F)
    to_docx(blk, T, DOCX)
    info = verify_docx(DOCX, blk)
    (HERE / "report.md").write_text(to_markdown(blk, T), newline="\n")
    (HERE / "verification.md").write_text("# Verification record\n\nBefore building, every sealed file used was re-hashed against `root_run_seal.json`. From the actual rows the builder recomputed: the per-depth and pooled RAW, aligned and no-change RMSE/bias/counts (matches Root's comparison.json), the budgets, "
                                          "the all-zero runoff/percolation/drainage/ponding/onset counts, exact NumPy = Numba row equality, the 900 s maximum difference, and the exact configuration difference (ten numeric entries x 1.2, nothing else). "
                                          "Static DOCX checks: zip integrity, one media file per figure (3), table count equals header-row count, footer page field, no placeholder or DRAFT text. No Office renderer was available, so rendered pagination and legibility were "
                                          "not checked by the builder.\n\n```\n" + json.dumps(info, indent=1) + "\n```\n", newline="\n")
    man = {"docx": {"file": DOCX.name, **info}, "figures": {p.name: sha(p) for p in sorted(FIG.glob("*.png"))}, "tables": {p.name: sha(p) for p in sorted(TAB.iterdir())}, "data": {p.name: sha(p) for p in sorted(DATA.iterdir())},
           "builder": sha(Path(__file__)), "report_md": sha(HERE / "report.md"), "sealed_inputs_checked": {k: v for _, k in R["pending"] for v in [R["seal"][k]]}}
    (HERE / "artifact_manifest.json").write_text(json.dumps(man, indent=1) + "\n", newline="\n")
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
