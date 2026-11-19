"""Builds the JORN 160-day rainfall comparison report (figures, tables, editable Markdown, verification note, manifest, Word document).

    PYTHONPATH=<task>/../site_hydraulics_interface_comparison_20261117/report_tools MPLCONFIGDIR=/tmp/syrup-jorn160-report PYTHONDONTWRITEBYTECODE=1 /home/okin/MAPLE/.venv/bin/python -B build_report.py

DOCUMENTS ONLY. Reusable with DECLARED inputs: set JORN160_TASK_DIR to the task folder holding Root's sealed results (not portable without them). The model was executed by Root; this script only READS the sealed actual results (every file is re-hashed against `root_run_seal.json` before use), computes summaries from them and draws them.
No model, fit, network, GPU or raw NEON file is touched. Compact copies of the small result files are written under `data/`.
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
TASK = Path(os.environ.get("JORN160_TASK_DIR", str(HERE.parents[4])))
FIG, TAB, DATA = HERE / "figures", HERE / "tables", HERE / "data"
DOCX = HERE / "SYRUP_JORN_160day_rainfall_comparison_20261119.docx"
LV = ["501", "502", "503"]
DEPTH_LABEL = {"501": "5.8 cm sensor (layer 0-10 cm)", "502": "15.8 cm sensor (layer 10-20 cm)", "503": "25.8 cm sensor (layer 20-30 cm)"}
GREEN, BLUE, GREY = "#2E8B57", "#1F5FA8", "#777777"
for d in (FIG, TAB, DATA):
    d.mkdir(exist_ok=True)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def J(rel):
    return json.loads((TASK / rel).read_text())


# ------------------------------------------------------------------------------------------------------ sealed inputs
USED = ["actual_numba/result.json", "actual_numba/timing.json", "actual_numpy/timing.json", "actual_numba_dt900/timing.json", "root_result_audit.json", "root_decimal_verification.json", "root_restart_verification.json",
        "root_independent_inputs.json", "root_physical_bounds.json", "frozen_helper_manifest.json"]
SMALL = ["actual_numba/timing.json", "actual_numpy/timing.json", "actual_numba_dt900/timing.json", "root_result_audit.json", "root_decimal_verification.json", "root_restart_verification.json", "root_physical_bounds.json", "frozen_helper_manifest.json"]


def load():
    seal = J("root_run_seal.json")
    for rel in USED:                                                      # every sealed file used is re-hashed first
        assert sha(TASK / rel) == seal[rel], f"{rel} differs from root_run_seal.json"
    R = {"seal": seal, "res": J("actual_numba/result.json"), "audit": J("root_result_audit.json"), "dec": J("root_decimal_verification.json"), "rst": J("root_restart_verification.json"),
         "bounds": J("root_physical_bounds.json"), "inputs": J("root_independent_inputs.json"), "case": J("case_numba.json"), "manifest": J("root_inputs_manifest.json"), "helper": J("frozen_helper_manifest.json"),
         "tm": {k: J(f"actual_{k}/timing.json") for k in ("numba", "numpy", "numba_dt900")},
         "wall": {k: J(f"actual_{k}_runs/result.json") for k in ("numba", "numpy", "numba_dt900")}, "rt": {k: J(f"runtime_{k}.json") for k in ("numba", "numpy", "numba_dt900")}}
    for rel in SMALL:
        shutil.copyfile(TASK / rel, DATA / Path(rel).parent.name / Path(rel).name if "/" in rel else DATA / rel) if False else None
    for rel in SMALL:
        dest = DATA / rel.replace("/", "__")
        shutil.copyfile(TASK / rel, dest)
        assert sha(dest) == seal[rel]
    for rel, name in (("case_numba.json", "case_numba.json"), ("actual_numba/result.json", "actual_numba__result.json"), ("runtime_numba.json", "runtime_numba.json"), ("runtime_numpy.json", "runtime_numpy.json"),
                      ("runtime_numba_dt900.json", "runtime_numba_dt900.json"), ("actual_numba_runs/result.json", "actual_numba_runs__result.json"), ("actual_numpy_runs/result.json", "actual_numpy_runs__result.json"),
                      ("actual_numba_dt900_runs/result.json", "actual_numba_dt900_runs__result.json"), ("root_independent_inputs.json", "root_independent_inputs.json"), ("root_inputs_manifest.json", "root_inputs_manifest.json")):
        shutil.copyfile(TASK / rel, DATA / name)                         # exact copies of Root's files (hashes in the manifest)
    rows = R["res"]["rows"]
    R["rows"] = rows
    R["n"] = len(rows)
    R["theta"] = np.array([r["layer_theta"] for r in rows])
    R["obs"] = np.array([[np.nan if v is None else v for v in r["obs_theta"]] for r in rows])
    R["theta0"] = np.array(R["res"]["plan"]["initial"]["layer_theta"])
    R["obs0"] = np.array(R["res"]["plan"]["initial"]["point_theta"])
    R["end"] = [dt.date.fromisoformat(r["date"]) + dt.timedelta(days=1) for r in rows]             # nominal close = end of the UTC-7 day
    R["start"] = dt.date.fromisoformat(rows[0]["date"])
    return R


def check(R):
    """Recompute the headline numbers independently of the figures and compare with Root's audit."""
    a, rows, res = R["audit"], R["rows"], R["res"]
    assert R["n"] == 160 and res["complete"] and R["rows"][-1]["nominal_close_utc"] == "2024-06-26T07:00:00Z"
    off = R["theta0"] - R["obs0"]
    e_raw, e_al, e_ps = {}, {}, {}
    for j, v in enumerate(LV):
        ok = np.isfinite(R["obs"][:, j])
        e_raw[v] = R["theta"][ok, j] - R["obs"][ok, j]
        e_al[v] = (R["theta"][ok, j] - R["theta0"][j]) - (R["obs"][ok, j] - R["obs0"][j])
        e_ps[v] = -(R["obs"][ok, j] - R["obs0"][j])
        for kind, e in (("raw", e_raw[v]), ("offset_aligned", e_al[v]), ("persistence_offset_aligned", e_ps[v])):
            assert e.size == a["scores"][v][kind]["n"] and math.isclose(100 * math.sqrt(np.mean(e**2)), a["scores"][v][kind]["rmse_percentage_points"], rel_tol=1e-9), (v, kind)
        assert math.isclose(100 * e_al[v].mean(), a["scores"][v]["offset_aligned"]["bias_percentage_points"], rel_tol=1e-9)
    for kind, d in (("raw", e_raw), ("offset_aligned", e_al), ("persistence_offset_aligned", e_ps)):
        e = np.concatenate([d[v] for v in LV])
        assert e.size == 282 and math.isclose(100 * math.sqrt(np.mean(e**2)), a["pooled"][kind]["rmse_percentage_points"], rel_tol=1e-9), kind
    b = a["budgets_mm"]
    ev = 1000 * sum(r["evaporation_m3"] for r in rows)
    assert math.isclose(ev, b["evaporation"], rel_tol=1e-12) and math.isclose(1000 * sum(r["rain_m3"] for r in rows), b["rain"], rel_tol=1e-12)
    assert math.isclose(b["initial"] + b["rain"] - b["evaporation"] - b["runoff"] - b["bottom_drainage"], b["final"], abs_tol=1e-9)
    assert res["ledger"]["n_matrix_steps"] == 55 and res["ledger"]["n_soil_steps"] == 215 and res["physical_end_t_s"] == 13923000.0 and res["added_rain_hours"] == 27.5
    assert all(a["matrix_regime"][k] == 0 for k in ("n_ponded_steps", "n_onsets", "room_rejected_m3", "runoff_matrix_pool_m3", "surface_return_m3"))
    assert all(r["drainage_m3"] == 0.0 and all(x == 0.0 for x in r["percolation_m3"]) for r in rows)
    return {"initial_offsets_pts": (100 * off).tolist(), "n_wet_days": sum(1 for r in rows if r["storm_rows"])}


# ------------------------------------------------------------------------------------------------------ figures
def style(ax):
    ax.grid(alpha=0.25)
    ax.tick_params(labelsize=8)


def dateaxis(ax):
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))


def wet_marks(ax, R):
    for r, d in zip(R["rows"], R["end"]):
        if r["storm_rows"]:
            ax.axvline(d - dt.timedelta(days=0.5), color="#9EC5E8", lw=0.8, zorder=0)


def fig_raw(R):
    fig, axs = plt.subplots(3, 1, figsize=(7.2, 6.2), sharex=True)
    x = [R["start"]] + R["end"]
    for j, ax in enumerate(axs):
        ax.plot(x, np.r_[R["theta0"][j], R["theta"][:, j]] * 100, color=GREEN, lw=1.4, label="model layer mean (daily close)")
        ax.scatter(R["end"], R["obs"][:, j] * 100, facecolors="none", edgecolors="black", s=14, lw=0.8, label="sensor point value (daily close)")
        ax.scatter([R["start"]], [R["theta0"][j] * 100], marker="^", color=GREEN, s=46, zorder=5, label="initial model layer mean")
        ax.scatter([R["start"]], [R["obs0"][j] * 100], marker="^", facecolors="none", edgecolors="black", s=46, zorder=5, label="initial sensor value")
        wet_marks(ax, R)
        ax.set_ylabel("water content (vol. %)", fontsize=9)
        ax.set_title(DEPTH_LABEL[LV[j]], fontsize=9.5, loc="left")
        style(ax)
    dateaxis(axs[-1])
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=8, frameon=False)
    fig.suptitle("RAW values: model layer means vs sensor point values (light blue lines = rain days)", fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    fig.savefig(FIG / "fig1_raw_160day.png", dpi=150)
    plt.close(fig)


def fig_aligned(R):
    fig, axs = plt.subplots(3, 1, figsize=(7.2, 6.2), sharex=True)
    x = [R["start"]] + R["end"]
    for j, ax in enumerate(axs):
        mod = (np.r_[R["theta0"][j], R["theta"][:, j]] - R["theta0"][j]) * 100
        obs = (R["obs"][:, j] - R["obs0"][j]) * 100
        ax.axhline(0, color=GREY, lw=0.8, ls="--", label="no change from day 0 (reference)")
        ax.plot(x, mod, color=GREEN, lw=1.4, label="model change since day 0")
        ax.scatter(R["end"], obs, facecolors="none", edgecolors="black", s=14, lw=0.8, label="sensor change since day 0")
        ax.scatter([R["start"]], [0], marker="^", color=GREEN, s=50, zorder=5)
        ax.scatter([R["start"]], [0], marker="^", facecolors="none", edgecolors="black", s=80, zorder=6, label="day 0 (both = 0 by construction)")
        wet_marks(ax, R)
        ax.set_ylabel("change (pct. points)", fontsize=8.5)
        ax.set_title(DEPTH_LABEL[LV[j]], fontsize=9.5, loc="left")
        style(ax)
    dateaxis(axs[-1])
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=8, frameon=False)
    fig.suptitle("Changes since day 0 with ONE fixed offset per depth (scoring view only; model water is not altered)", fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    fig.savefig(FIG / "fig2_aligned_160day.png", dpi=150)
    plt.close(fig)


def fig_zoom(R):
    lo, hi = 2, 18                                                      # nominal days closing Jan 20 .. Feb 4
    idx = range(lo, hi)
    x = [R["end"][i] for i in idx]
    fig, axs = plt.subplots(4, 1, figsize=(7.2, 6.8), sharex=True, gridspec_kw={"height_ratios": [1, 1.5, 1.5, 1.5]})
    axs[0].bar([R["end"][i] - dt.timedelta(days=0.5) for i in idx], [R["rows"][i]["observed_rain_mm"] for i in idx], width=0.8, color=BLUE)
    axs[0].set_ylabel("rain (mm per day)", fontsize=9)
    axs[0].set_title("Observed rain on each nominal day (the 10.0 mm Jan 23 event is the largest of the 160 days)", fontsize=9.5, loc="left")
    style(axs[0])
    for j in range(3):
        ax = axs[j + 1]
        ax.axhline(0, color=GREY, lw=0.8, ls="--")
        ax.plot(x, [(R["theta"][i, j] - R["theta0"][j]) * 100 for i in idx], color=GREEN, marker="o", ms=3, lw=1.3, label="model change since day 0")
        ax.scatter(x, [(R["obs"][i, j] - R["obs0"][j]) * 100 for i in idx], facecolors="none", edgecolors="black", s=22, label="sensor change since day 0")
        ax.set_ylabel("change (pct. points)", fontsize=9)
        ax.set_title(DEPTH_LABEL[LV[j]], fontsize=9.5, loc="left")
        style(ax)
    axs[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    axs[-1].xaxis.set_major_locator(mdates.DayLocator(interval=2))
    h, l = axs[1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIG / "fig3_rain_pulse_zoom.png", dpi=150)
    plt.close(fig)


def fig_budget(R):
    rows = R["rows"]
    x = R["end"]
    rain = np.cumsum([1000 * r["rain_m3"] for r in rows])
    pet = np.cumsum([r["pet_mm_day"] for r in rows])
    ev = np.cumsum([1000 * r["evaporation_m3"] for r in rows])
    sto = np.array([1000 * r["storage_m3"] for r in rows])
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.6), gridspec_kw={"width_ratios": [1.4, 1]})
    ax = axs[0]
    ax.plot(x, pet, color="#B5651D", lw=1.5, label="potential evaporation (PET), cumulative")
    ax.plot(x, ev, color=GREEN, lw=1.5, label="actual evaporation from the soil, cumulative")
    ax.plot(x, rain, color=BLUE, lw=1.5, label="rain, cumulative")
    ax.plot([R["start"]] + x, np.r_[22.9424, sto], color="black", lw=1.2, ls="--", label="water stored in the 30 cm column")
    ax.set_ylabel("millimetres of water over the 1 m² column", fontsize=8.5)
    dateaxis(ax)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    style(ax)
    b = R["audit"]["budgets_mm"]
    ax = axs[1]
    labels = ["initial", "+ rain", "- evap.", "- runoff", "- drain", "final"]
    vals = [b["initial"], b["rain"], -b["evaporation"], -b["runoff"], -b["bottom_drainage"], b["final"]]
    cols = [GREY, BLUE, GREEN, "#C0392B", "#C0392B", "black"]
    ax.bar(range(6), vals, color=cols)
    for i, v in enumerate(vals):
        ax.text(i, v + (1.2 if v >= 0 else -3.2) if v else 1.2, f"{v:.2f}" if v else "0", ha="center", fontsize=7)
    ax.set_xticks(range(6))
    ax.set_xticklabels(labels, fontsize=7, rotation=40, ha="right")
    ax.set_ylabel("mm", fontsize=8.5)
    ax.set_title("Water budget, whole run", fontsize=9)
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_water_budget.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------------ tables
def f(x, n=2):
    return "n/a" if x is None else f"{x:.{n}f}"


def build_tables(R):
    a, case, res, tm, wall, rt = R["audit"], R["case"], R["res"], R["tm"], R["wall"], R["rt"]
    T = {}
    props = []
    for i, p in enumerate(case["provenance"]["layer_properties"]):
        props.append([["0-10", "10-20", "20-30"][i], f(p["theta_field_capacity"], 3), f(p["theta_saturation"], 3), "0.010", f(p["Ks_mm_h"], 1), f(p["effective_density_g_cm3"], 3)])
    T["props"] = (["Layer (cm)", "Field capacity", "Saturation", "Residual", "Ks (mm/h)", "Density used (g/cm³)"], props)
    T["setup"] = (["Item", "Value"], [
        ["Period", "160 nominal days, 2024-01-18 to 2024-06-25 (UTC-7 days)"], ["Column", "one 1 m² column, three 10 cm layers (0-30 cm), open bottom, gravity on"],
        ["Observed rain", "25.088 mm; 26 positive half-hours on 11 days; peak 10.034 mm/h"], ["Matrix intake limit K", f"{res['plan']['matrix']['K_mm_h']:.4f} mm/h (peak rain is {10.034 / res['plan']['matrix']['K_mm_h'] * 100:.0f}% of it)"],
        ["Diffusion d / evaporation stress", "0.0010239461376024856 / 0.20170581117963332 (m³/m³); E factor 1; transpiration 0"],
        ["Initial water", "22.9424 mm (layers 7.295, 8.625, 7.022 mm)"], ["Steps", "55 matrix-intake steps, 215 soil steps"],
        ["Clock", "nominal 160.000 days; physical 161.146 days (+27.5 h of storm time)"]])
    b = a["budgets_mm"]
    T["budget"] = (["Term", "mm of water", "Note"], [["Initial storage", f(b["initial"], 4), "integral of the three sensor values"], ["+ Rain", f(b["rain"], 3), "all entered the soil"], ["- Evaporation", f(b["evaporation"], 4), "actual soil evaporation"],
                                                    ["- Runoff", f(b["runoff"], 1), "none"], ["- Bottom drainage", f(b["bottom_drainage"], 1), "none"], ["= Final storage", f(b["final"], 4), "initial + rain - evap. = 13.985 mm"],
                                                    ["Closure residual", f"{res['cumulative_closure']['soil_residual_m3']:.2e} m³", f"native bound {res['cumulative_closure']['soil_bound_m3']:.3e} m³"]])
    sc = a["scores"]
    rows = []
    for v in LV:
        s = sc[v]
        rows.append([DEPTH_LABEL[v].split(" (")[0], s["raw"]["n"], f(s["raw"]["rmse_percentage_points"], 3), f(s["offset_aligned"]["rmse_percentage_points"], 3), f(s["persistence_offset_aligned"]["rmse_percentage_points"], 3),
                     f(s["offset_aligned"]["bias_percentage_points"], 3), "better" if s["offset_aligned"]["rmse_percentage_points"] < s["persistence_offset_aligned"]["rmse_percentage_points"] else "worse"])
    p = a["pooled"]
    rows.append(["All depths pooled", p["raw"]["n"], f(p["raw"]["rmse_percentage_points"], 3), f(p["offset_aligned"]["rmse_percentage_points"], 3), f(p["persistence_offset_aligned"]["rmse_percentage_points"], 3), "-", "better"])
    T["scores"] = (["Depth", "n", "RAW RMSE", "Aligned RMSE", "No-change RMSE", "Aligned bias", "vs no change"], rows)
    ev = a["wet_day_responses"]
    rr = []
    for e in ev:
        r1, r2, r3 = (e["responses_percentage_points"][v] for v in LV)
        rr.append([e["date"][5:], f(e["rain_mm"], 1), f"{f(r1['1d']['observed'])} / {f(r1['1d']['model'])}", f"{f(r2['1d']['observed'])} / {f(r2['1d']['model'])}", f"{f(r2['3d']['observed'])} / {f(r2['3d']['model'])}",
                   f"{f(r3['1d']['observed'])} / {f(r3['1d']['model'])}"])
    T["events"] = (["Wet day", "Rain (mm)", "5.8 cm 24 h obs / model", "15.8 cm 24 h obs / model", "15.8 cm 72 h obs / model", "25.8 cm 24 h obs / model"], rr)

    def tr(k, name):
        return [name, f(tm[k]["integration_s"], 4), f(tm[k]["operator_prepare_s"], 3), f(tm[k]["jit_warmup_control_s"], 4), f(tm[k]["source_checks_and_input_preparation_s"], 3), f(tm[k]["output_write_s"], 3),
                f(rt[k]["driver_call_wall_s"], 3), f(wall[k]["seconds"], 3)]
    T["runtime"] = (["Run", "Integration (s)", "Operator prep + JIT (s)", "Warm-up control (s)", "Source checks + input prep (s)", "Output write (s)", "Driver call wall (s)", "Full fresh command (s)"],
                    [tr("numba", "Numba, 1800 s storm step"), tr("numpy", "NumPy, 1800 s storm step"), tr("numba_dt900", "Numba, 900 s storm step")])
    d = R["dec"]["cases"]
    T["checks"] = (["Check", "Result"], [
        ["NumPy vs Numba", "every trajectory/flux array identical (max difference 0)"], ["Storm step 1800 s vs 900 s", f"max water-content difference {a['timestep_refinement_max_percentage_point_difference']:.6f} percentage points"],
        ["Independent 90-digit calculation", f"max water error {d[0]['max_water_error_m']:.2e} m (Numba/NumPy), {d[2]['max_water_error_m']:.2e} m (900 s); under 0.05 of the native one-step bound"],
        ["Restart at day 80 (separate process)", "rows, ledgers, clocks, scores identical (driver-level, not a production checkpoint qualification)"], ["Closure", f"{res['cumulative_closure']['soil_residual_m3']:.2e} m³ vs bound {res['cumulative_closure']['soil_bound_m3']:.3e} m³"],
        ["Tests", "77 focused tests passed, Ruff clean (Root); no GPU run"], ["Ponding / onset / runoff / drainage counts", "all zero (Root's a-priori bound: max water content 0.3371 < lowest field capacity 0.3951)"]])
    return T


def derived(R):
    """Numbers used in the prose, all computed from the sealed results."""
    a = R["audit"]
    ev = a["wet_day_responses"]
    g = lambda e, v, k, w: e["responses_percentage_points"][v][k][w]
    top = [(g(e, "501", "1d", "observed"), g(e, "501", "1d", "model")) for e in ev if g(e, "501", "1d", "observed") is not None]
    mid3 = [(g(e, "502", "3d", "observed"), g(e, "502", "3d", "model")) for e in ev if g(e, "502", "3d", "observed") is not None]
    mid1 = [(g(e, "502", "1d", "observed"), g(e, "502", "1d", "model")) for e in ev if g(e, "502", "1d", "observed") is not None]
    d = {"n_top": len(top), "top_model_larger": sum(1 for o, m in top if m > o), "n_mid3": len(mid3), "mid3_obs_exceeds": sum(1 for o, m in mid3 if o - m >= 0.3), "n_mid1": len(mid1),
         "mid1_positive": sum(1 for o, m in mid1 if o > 0), "mid1_pos_list": [o for o, m in mid1 if o > 0], "jan": ev[0]}
    assert [e["date"] for e in ev if g(e, "502", "3d", "observed") is not None and g(e, "502", "3d", "observed") - g(e, "502", "3d", "model") >= 0.3] == ["2024-01-23", "2024-02-11", "2024-02-13"]
    p = a["pooled"]
    d["mse_cut"] = 100 * (1 - p["offset_aligned"]["mse"] / p["persistence_offset_aligned"]["mse"])
    d["rmse_cut"] = 100 * (1 - p["offset_aligned"]["rmse_percentage_points"] / p["persistence_offset_aligned"]["rmse_percentage_points"])
    return d


# ------------------------------------------------------------------------------------------------------ text
REFS = [(1, "Tietjen, B., Zehe, E., & Jeltsch, F. (2009). Simulating plant water availability in dry lands under climate change: a generic model of two soil layers. Water Resources Research 45, W01418.", "https://doi.org/10.1029/2007WR006589"),
        (2, "Saxton, K. E., & Rawls, W. J. (2006). Soil water characteristic estimates by texture and organic matter for hydrologic solutions. Soil Science Society of America Journal 70, 1569-1578.", "https://doi.org/10.2136/sssaj2005.0117"),
        (3, "Hargreaves, G. H., & Samani, Z. A. (1985). Reference crop evapotranspiration from temperature. Applied Engineering in Agriculture 1, 96-99.", "https://doi.org/10.13031/2013.26773"),
        (4, "NEON (RELEASE-2026). Soil water content and water salinity, DP1.00094.001 (soil moisture sensors).", "https://data.neonscience.org/data-products/DP1.00094.001/RELEASE-2026"),
        (5, "NEON (RELEASE-2026). Precipitation, DP1.00045.001 (rain forcing).", "https://data.neonscience.org/data-products/DP1.00045.001/RELEASE-2026"),
        (6, "NEON (RELEASE-2026). Single aspirated air temperature, DP1.00002.001 (evaporation-demand forcing).", "https://data.neonscience.org/data-products/DP1.00002.001/RELEASE-2026"),
        (7, "EcoHyD reference code used for the Green-Ampt matrix law and the unmeasured reference values psi = 0.166 m and the vegetation factor g: GitHub repository selinaZitrone/Baldauf_et_al_Biocrust_Ecohydrology, commit 80b10e6b6b82ad16e9fd41a31901cc3157d880e9 (files model/src/WaterCell.cpp and model/Parameters/soilparameters_tabernas.txt). A later EcoHyD descendant, not a verified original 2009 release; local archive described in reference_models/ecohyd/README.md.",
         "https://github.com/selinaZitrone/Baldauf_et_al_Biocrust_Ecohydrology/tree/80b10e6b6b82ad16e9fd41a31901cc3157d880e9")]


def blocks(R, T, D, F):
    a, res = R["audit"], R["res"]
    p, sc = a["pooled"], a["scores"]
    jan = D["jan"]["responses_percentage_points"]
    B = [("title", "SYRUP on a real 160-day rain record: JORN, January-June 2024"),
         ("sub", "A frozen-parameter comparison of the native SYRUP column with NEON soil-moisture sensors. The model runs were executed and sealed by Root; this document reports them."),
         ("h1", "1. What this report shows"),
         ("bul", ["One 1 m², 30 cm, three-layer soil column was run for 160 nominal days (Jan 18 - Jun 25, 2024) with NEON observed rain [5] (25.088 mm over this study period, on 11 days), Hargreaves evaporation demand from NEON air temperature [6] and the **frozen** dry-down parameters. Nothing was refitted. Soil-moisture sensors [4] only start the run (day 0) and score it.",
                  "**All rain entered the soil.** Two separate reasons: (i) rain (peak 10.0 mm/h) stayed below the unponded intake limit K (38.2 mm/h) and the top layer always had pore room, so no pond or runoff formed; (ii) modelled water content stayed far below field capacity (highest 0.153 vs lowest field capacity 0.395), so there was no gravity flow or bottom drainage. Ponding being absent does not by itself imply no drainage; both were checked. The reference suction psi and vegetation factor g did not control intake because the unponded branch was always taken [7].",
                  f"**Water budget:** 22.94 mm initially + 25.09 mm rain - 34.05 mm evaporation = 13.98 mm at the end. Rain minus evaporation is -8.96 mm, exactly the loss of stored water (closure error {abs(res['cumulative_closure']['soil_residual_m3']):.1e} m³, far inside the native bound).",
                  f"**Against 282 sensor values** the model's pooled error after one fixed start-up offset is {p['offset_aligned']['rmse_percentage_points']:.2f} percentage points, versus {p['persistence_offset_aligned']['rmse_percentage_points']:.2f} for assuming no change ({D['rmse_cut']:.0f}% lower RMSE). The gain is not uniform: it comes mainly from the 15.8 cm sensor (147 points); the 5.8 cm sensor is about twice as bad as no change (RMSE 1.82 vs 0.88) and the 25.8 cm sensor is also worse (1.05 vs 0.89).",
                  "The model's top layer reacts much more strongly to rain than the 5.8 cm sensor, and the deepest layer is on average too wet. Preferential flow is a possible but **unproven** explanation for part of the middle-depth behavior; it was switched off and nothing was fitted."]),
         ("h1", "2. What was run"),
         ("p", "SYRUP owns the soil water. For each nominal day the native rainfall schedule is integrated in storm sub-steps (at most 30 minutes), in which rain enters through the Green-Ampt matrix intake of the Tietjen-style two-layer scheme [1] and the column redistributes water; any rain the soil cannot take would leave immediately as runoff (the user's open-surface boundary). After the storm, one 86,400 s inter-event step applies evaporation from the day's Hargreaves demand [3] (the native EcoHyD-style Hargreaves family, not an exact FAO-56 implementation). There is no evaporation, transpiration or growth during rain; transpiration is zero all run (no plants are modelled), so the late-spring evaporation here is **not** a field evapotranspiration budget."),
         ("p", "**Clock.** This is a daily-weather approximation, not an hourly field clock. The 27.5 h that rain occupies (99,000 s) is added on top of 160 nominal days, so the physical clock ends at 161.146 days. Sensor targets are matched to the nominal day closes (07:00 UTC = local midnight), not to the physical clock. A storm that crosses local midnight counts as two daily occurrences; wetting history carries across."),
         ("tab", "setup", "Table 1. Case summary."),
         ("p", "**Soil properties** (Table 2) come from the approved Saxton-Rawls [2] surrogate recipe (density floor, air-entry floor of zero), not from measured hydraulic properties. The Saxton-Rawls floors are this project's choices, not published recommendations. The field capacity of 0.395-0.399 was kept as approved. It is high compared with every observed and modelled water content in this run (observed at most 0.10, modelled at most 0.153), which is why gravity drainage never switches on here. Diffusion d and the evaporation stress value are conditional empirical values fitted jointly to 53 dry-period sensor values from 2025 (JORN and MOAB). The 2024 data were not used in that 2025 calibration, so this is a conditional out-of-sample temporal check at one site with surrogate properties; it does not validate the hydraulic properties or other sites. The stress value is an empirical scale, not a measured wilting point."),
         ("tab", "props", "Table 2. Soil layers used (surrogates; volumetric fractions)."),
         ("p", "**Other limits to keep in mind.** The air temperature is the 030 height of the NEON sensor [6], while the dry-down calibration used 020; 36 of 7,680 half-hours (up to 4 h) are interpolated, with flags kept. Preferential flow is off by the user's decision. The sensor values [4] are point readings at 5.8, 15.8 and 25.8 cm, while the model reports 10 cm layer means."),
         ("h1", "3. Where the water went"),
         ("fig", "fig4_water_budget.png", "Figure 1. Left: cumulative rain, PET demand and actual evaporation, and stored water. Right: the whole-run budget. Evaporation (34.05 mm) is lower than PET (777 mm) because the native bare-soil law reduces evaporation as the top layer dries below the stress value, even with the evaporation factor set to 1."),
         ("tab", "budget", "Table 3. Water budget (mm over the 1 m² column)."),
         ("p", "Stored water fell by about 9 mm (22.9 to 14.0 mm): evaporation removed 34 mm, more than the 25 mm of rain that fell. Even so, evaporation stayed far below the demand because the dry top layer limits it."),
         ("h1", "4. Comparison with the sensors"),
         ("p", f"Two views are shown. **Figure 2 (RAW)** compares model layer means directly with the sensor point values. The model's initial layer means differ from the point readings (offsets {F['initial_offsets_pts'][0]:+.2f}, {F['initial_offsets_pts'][1]:+.2f}, {F['initial_offsets_pts'][2]:+.2f} percentage points at 5.8, 15.8 and 25.8 cm) because a layer mean integrates the whole profile. **Figure 3 (aligned)** removes only that constant offset, measured once at day 0, to compare *changes*. It is a scoring view: the model's actual water is never changed, and nothing is re-aligned after storms. Missing sensor values are simply absent (no interpolation). The last shallow-sensor value used is at the daily close of 2024-03-15 07:00 UTC (nominal day Mar 14). A later raw record ending 2024-03-15 14:30 UTC passes quality control, but no valid record ends at any subsequent daily close, so the shallow comparison stops there while the model continues to June; the lower sensors continue. No post-initial moisture value drives the model."),
         ("fig", "fig1_raw_160day.png", "Figure 2. RAW comparison. Black open circles: sensor point values; green line: model layer mean; triangles: day-0 values. Vertical light-blue lines mark rain days. The shallow sensor has no valid value at any daily close after 2024-03-15."),
         ("fig", "fig2_aligned_160day.png", "Figure 3. Changes since day 0 using one fixed offset per depth. Day 0 is a triangle drawn over a circle (both zero by construction)."),
         ("tab", "scores", "Table 4. Scores in percentage points of water content (100 x m³/m³). Aligned = one fixed offset from day 0; no-change = predicting the day-0 value forever."),
         ("p", f"Pooled over all 282 values the aligned RMSE is {p['offset_aligned']['rmse_percentage_points']:.3f} points against {p['persistence_offset_aligned']['rmse_percentage_points']:.3f} for no change (pooled squared error {D['mse_cut']:.0f}% lower; the pooled RAW RMSE is {p['raw']['rmse_percentage_points']:.3f}). The pooled gain is dominated by the 147 middle-sensor points, where the model removes much of the drift that persistence cannot. At 5.8 cm ({sc['501']['offset_aligned']['rmse_percentage_points']:.2f} vs {sc['501']['persistence_offset_aligned']['rmse_percentage_points']:.2f}) and 25.8 cm ({sc['503']['offset_aligned']['rmse_percentage_points']:.2f} vs {sc['503']['persistence_offset_aligned']['rmse_percentage_points']:.2f}) the model is worse than no change. This report does not claim uniform success, a physically verified d, an R², or statistical significance."),
         ("h1", "5. Rain events and the preferential-flow question"),
         ("p", f"The largest event (Jan 23, {jan['501']['1d']['rain_in_window_mm']:.1f} mm) is the best test. Within 24 h the 5.8 cm sensor rose {jan['501']['1d']['observed']:.2f} points while the model's top layer rose {jan['501']['1d']['model']:.2f}; after 72 h the values are {jan['501']['3d']['observed']:.2f} and {jan['501']['3d']['model']:.2f}. The 10 cm layer mean absorbs 10 mm of rain over a thin 10 cm slab, so a much larger modelled pulse than at a point 5.8 cm deep is partly a support difference. At 15.8 cm the sensor changed {jan['502']['1d']['observed']:+.2f} points in 24 h and {jan['502']['3d']['observed']:+.2f} in 72 h, against {jan['502']['1d']['model']:+.2f} and {jan['502']['3d']['model']:+.2f} modelled. The 25.8 cm sensor has no valid pre/post values for this event."),
         ("fig", "fig3_rain_pulse_zoom.png", "Figure 4. Jan 20 - Feb 4 rain pulse. Top: rain per nominal day. Below: model and sensor change since day 0 at each depth (the same single day-0 offset as Figure 3; no pre-storm re-offset)."),
         ("tab", "events", "Table 5. Responses (percentage points) from the daily close before each wet day to 24 h and 72 h after it, (fixed windows), as computed in Root's independent audit. Each cell is observed / model; n/a = no valid sensor value."),
         ("p", f"What the table supports: (i) the top layer is modelled much wetter after rain than the 5.8 cm sensor in {D['top_model_larger']} of {D['n_top']} events with data; (ii) at 15.8 cm, observed 24 h changes are positive in only {D['mid1_positive']} of {D['n_mid1']} events (Feb 11, +0.03, and Feb 13, +0.16 points), and a 72 h rise at least 0.3 points larger than modelled occurs in {D['mid3_obs_exceeds']} of {D['n_mid3']} events with data (Jan 23, Feb 11, Feb 13). The 72 h windows are FIXED windows that can contain further rain (the Feb 11 window includes Feb 13), so they are not independent, isolated storm responses; (iii) only 2 of the 11 wet days have valid 25.8 cm values both before and after, and a missing value is not evidence of no response. The deepest layer is on average too wet (aligned bias +{sc['503']['offset_aligned']['bias_percentage_points']:.2f} points, RAW +{sc['503']['raw']['bias_percentage_points']:.2f}), so there is no support for a blanket bypass of water to the deepest layer, which would make it worse."),
         ("p", "**Assessment.** A selective, middle-layer preferential path is *plausible* for the Jan 23 and a few February events, but the evidence is thin and mixed; it is not established, and no rate or weight was fitted. Several other explanations remain open and can produce the same pattern: (a) instantaneous mixing over a 10 cm layer versus a point value 5.8 cm deep (a constant day-0 offset cannot correct this after rain); (b) wetting-front timing that a daily clock cannot resolve; (c) the surrogate field capacity of about 0.40, which prevents gravity flow in the model; (d) a small shared diffusion d; (e) transpiration by field roots, which is not modelled; (f) the thin 30 cm column, with no capillary rise from below or lateral flow; and (g) temperature or calibration effects on the sensors. The model is not too dry everywhere: more deep water than predicted is not seen at every depth."),
         ("p", "**Recommended next step (for the user to decide).** Do not adopt a new physical model on this evidence. If a diagnostic is wanted, run a small, pre-declared sensitivity test with the existing preferential option on the Jan 23 - Feb 13 window only, targeting the middle layer, with values chosen by the user rather than fitted, and judge it against the quantitative response pattern in Table 5. First check alternatives (a) and (c) by examining sub-layer sensor support and a lower-field-capacity sensitivity."),
         ("h1", "6. Verification and run time"),
         ("tab", "checks", "Table 6. Independent checks by Root."),
         ("tab", "runtime", "Table 7. Measured run time (raw seconds). The Numba column includes one-time compilation."),
         ("p", "The Numba run integrates the 160 days in 0.040 s, about 3.1 times faster than NumPy (0.125 s) for this one small fixture; this is not a general performance claim. In a fresh process, compilation dominates (7.4 s total for Numba against 2.8 s for NumPy). The full fresh command also includes Root's source-verified bootstrap and imports, so the difference between the driver-call wall time and the full command is not model time. Development, data download, qualification and reporting are not included. Software: Python 3.12.3, Numba 0.67.0, one CPU column. No GPU was used and no scaling is claimed. No three-model coupled simulation is accepted by this study."),
         ("h1", "7. Main limitations"),
         ("bul", ["Surrogate soil properties and conditional empirical d and stress; no measured Ks or hydraulic validation.", "The rain was too light to test ponding, runoff, suction, gravity drainage or the preferential-flow path.", "Evaporation only (T = 0); no plants, wind, sediment, bed or routing.",
                  "Point sensors versus 10 cm layer means, a daily weather clock and a 030 vs 020 air-height difference.", "One site, one half-year, one soil column; no significance testing."]),
         ("h1", "References"), ("refs", REFS),
         ("h1", "Appendix A. Provenance (compact)"),
         ("p", "Sources (Root's source-verified bootstrap, CPU): SYRUP production package 180 files, SHA-256 37d89f23dbf60626c85f85ea5dd6d1734d3bc7a4aa61d8f2fdfc29f237bcfbd0, unchanged; MAPLE package 412 files, SHA-256 fbc504bbcd1ec0c02e6a96089134a06187532455ef7ba5e6f146801c8313c02e. Baseline main 7c14c74 plus a frozen manifest of ten files: four pre-existing subsurface_validation helpers and six new files (four wet-validation driver modules and two test files). Primary Numba case config SHA-256 " + res["config_sha256"] + " (exact copy in data/case_numba.json)."),
         ("p", "Full evidence is in the attached files rather than this document: tables/provenance_seal.csv (SHA-256 of every sealed result and audit file), tables/provenance_inputs.csv (all pinned forcing and 18 soil-moisture input files with SHA-256 and sizes), data/ (primary config, result, timings, audit and verification JSON) and artifact_manifest.json. Raw NEON files are referenced by hash only and are not copied.")]
    return B


def prov_tables(R):
    seal = R["seal"]
    T1 = (["File", "SHA-256"], [[k, v] for k, v in seal.items()])
    files = R["manifest"]["files"]
    rows = [[Path(x["path"]).name, x["sha256"], str(x["bytes"])] for x in files]
    with open(TAB / "provenance_inputs.csv", "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["file", "sha256", "bytes"])
        w.writerows(rows)
    short = [[n if len(n) < 46 else n[:20] + "..." + n[-22:], h, b] for n, h, b in rows[:4]]
    short.append([f"{len(rows) - 4} SWC files (Jan-Jun 2024, three depths)", "see tables/provenance_inputs.csv", "-"])
    return T1, (["Input file", "SHA-256", "bytes"], short)


# ------------------------------------------------------------------------------------------------------ DOCX
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


def runs(par, text, size=None):
    for i, part in enumerate(re.split(r"\*\*(.+?)\*\*", text)):
        if part:
            r = par.add_run(part)
            r.bold = bool(i % 2)
            if size:
                r.font.size = Pt(size)


def setup_styles(doc):
    st = doc.styles
    n = st["Normal"]
    n.font.name, n.font.size = "Calibri", Pt(10.5)
    n.paragraph_format.space_after = Pt(6)
    n.paragraph_format.line_spacing = 1.08
    for name, sz, col in (("Heading 1", 15, "1F3A5F"), ("Heading 2", 12, "2B5C8A")):
        h = st[name]
        h.font.name, h.font.size, h.font.bold = "Calibri", Pt(sz), True
        h.font.color.rgb = RGBColor.from_string(col)
        h.paragraph_format.space_before, h.paragraph_format.space_after = Pt(12), Pt(4)
        h.paragraph_format.keep_with_next = True
    for nm, base in (("RepTitle", "Title"), ("RepSubtitle", "Subtitle"), ("RepCaption", "Normal"), ("RepTable", "Normal")):
        s = st.add_style(nm, WD_STYLE_TYPE.PARAGRAPH)
        s.base_style = st[base]
    t = st["RepTitle"]
    t.font.name, t.font.size, t.font.bold = "Calibri", Pt(20), True
    t.font.color.rgb = RGBColor.from_string("1F3A5F")
    st["RepSubtitle"].font.size, st["RepSubtitle"].font.italic = Pt(10.5), True
    c = st["RepCaption"]
    c.font.size, c.font.italic = Pt(9), True
    c.paragraph_format.space_after = Pt(10)
    c.paragraph_format.keep_together = True
    tt = st["RepTable"]
    tt.font.size = Pt(8)
    tt.paragraph_format.space_after = Pt(0)
    tt.paragraph_format.line_spacing = 1.0


def add_table(doc, hd, rows, width_in, caption, mono=False):
    cp = doc.add_paragraph(style="RepCaption")
    runs(cp, caption)
    cp.paragraph_format.keep_with_next = True
    cp.paragraph_format.space_after = Pt(3)
    t = doc.add_table(rows=1, cols=len(hd))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    wts = [min(max(len(str(h)) * 0.5, max(len(str(r[i])) for r in rows), 5), 40) for i, h in enumerate(hd)]
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
            r = cells[i].paragraphs[0].add_run(str(v))
            if mono and len(str(v)) == 64:
                r.font.name, r.font.size = "Courier New", Pt(6)
            cells[i].width = ws[i]
            if ri % 2:
                shade(cells[i], "F3F6FA")
    doc.add_paragraph().paragraph_format.space_after = Pt(4)


def to_docx(B, T, path):
    doc = Document()
    setup_styles(doc)
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Inches(8.5), Inches(11)
    sec.left_margin = sec.right_margin = Inches(0.9)
    sec.top_margin = sec.bottom_margin = Inches(0.85)
    fp = sec.footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    fr = fp.add_run("SYRUP JORN 160-day rainfall comparison · frozen-parameter, conditional out-of-sample check · page ")
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
            add_table(doc, hd, rows, 6.7, b[2], mono=b[1] in ("prov", "prov2"))
        elif k == "fig":
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.keep_with_next = True
            p.add_run().add_picture(str(FIG / b[1]), width=Inches(6.4))
            runs(doc.add_paragraph(style="RepCaption"), b[2])
        elif k == "refs":
            for n, c, u in b[1]:
                p = doc.add_paragraph()
                p.paragraph_format.left_indent = Inches(0.25)
                p.paragraph_format.first_line_indent = Inches(-0.25)
                p.add_run(f"[{n}] ").bold = True
                p.add_run(f"{c} {u}".strip()).font.size = Pt(9)
    doc.core_properties.title = "SYRUP on a real 160-day rain record: JORN, January-June 2024 "
    doc.core_properties.author = "SYRUP project"
    doc.save(path)


def verify_docx(path, T, B):
    z = zipfile.ZipFile(path)
    names = z.namelist()
    media = [n for n in names if n.startswith("word/media/")]
    xml = z.read("word/document.xml").decode()
    ntab, nhdr = xml.count("<w:tbl>"), xml.count("<w:tblHeader")
    nfig = sum(1 for b in B if b[0] == "fig")
    ntabs = sum(1 for b in B if b[0] == "tab")
    pagef = "PAGE" in "".join(z.read(n).decode(errors="ignore") for n in names if "footer" in n)
    text = re.sub(r"<[^>]+>", " ", xml)
    assert z.testzip() is None and len(media) == nfig == 4 and ntab == ntabs == nhdr and pagef
    assert not re.search(r"TODO|TBD|XXX|placeholder|\bnan\b", text)
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


def write_tables(T):
    for name, (hd, rows) in T.items():
        with open(TAB / f"{name}.csv", "w", newline="") as fh:
            w = csv.writer(fh, lineterminator="\n")
            w.writerow(hd)
            w.writerows(rows)
    with open(TAB / "daily_results.csv", "w", newline="") as fh:                  # compact daily export of the actual run (read from the sealed result)
        pass


def write_daily(R):
    with open(TAB / "daily_results.csv", "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["day_index", "date_utc_minus7", "nominal_close_utc", "physical_t_s", "rain_mm", "pet_mm", "evap_mm", "theta_l1", "theta_l2", "theta_l3", "obs_501", "obs_502", "obs_503"])
        for r in R["rows"]:
            w.writerow([r["day_index"], r["date"], r["nominal_close_utc"], r["physical_t_s"], 1000 * r["rain_m3"], r["pet_mm_day"], 1000 * r["evaporation_m3"], *r["layer_theta"], *["" if v is None else v for v in r["obs_theta"]]])


def main():
    R = load()
    F = check(R)
    D = derived(R)
    fig_raw(R)
    fig_aligned(R)
    fig_zoom(R)
    fig_budget(R)
    T = build_tables(R)
    T["prov"], T["prov2"] = prov_tables(R)
    PROV = {k: T.pop(k) for k in ("prov", "prov2")}
    write_tables(T)
    with open(TAB / "provenance_seal.csv", "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(PROV["prov"][0])
        w.writerows(PROV["prov"][1])
    write_daily(R)
    B = blocks(R, T, D, F)
    to_docx(B, T, DOCX)
    info = verify_docx(DOCX, T, B)
    (HERE / "report.md").write_text(to_markdown(B, T))
    (HERE / "verification.md").write_text("# Verification record\n\nBefore building, every sealed file used was re-hashed against `root_run_seal.json`; the 282 per-depth/pooled RAW, aligned and no-change RMSEs, biases and counts, the water budget, step counts, clocks and all-zero "
                                          "ponding/runoff/drainage were recomputed from the actual rows and matched Root's audit. Static DOCX checks: zip integrity, one media file per figure (4), table count equals header-row count, footer page field, "
                                          "no placeholder text. No Office renderer was available, so rendered pagination and figure legibility were not checked by this builder.\n\n```\n" + json.dumps(info, indent=1) + "\n```\n")
    man = {"docx": {"file": DOCX.name, **info}, "figures": {p.name: sha(p) for p in sorted(FIG.glob("*.png"))}, "tables": {p.name: sha(p) for p in sorted(TAB.iterdir())}, "builder": sha(Path(__file__)), "report_md": sha(HERE / "report.md"),
           "data": {p.name: sha(p) for p in sorted(DATA.iterdir())}, "sealed_inputs_checked": {k: R["seal"][k] for k in USED}, "note": "publication artifact = the DOCX named here only; the earlier DRAFT file is under superseded/ and is not part of this manifest"}
    (HERE / "artifact_manifest.json").write_text(json.dumps(man, indent=1) + "\n")
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
