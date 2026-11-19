"""Builds the joint d + evaporation-stress calibration report: figures, tables, editable Markdown, verification note, manifest and the Word document.

    PYTHONPATH=<task>/site_hydraulics_interface_comparison_20261117/report_tools MPLCONFIGDIR=/tmp/syrup-joint-report PYTHONDONTWRITEBYTECODE=1 /home/okin/MAPLE/.venv/bin/python -B build_report.py

Every number is computed here from the frozen JSON in `data/` (Root's actual CPU joint fits and independent checks) and the READ-ONLY previous study in `../evaporation_stress_calibration/data/`. Nothing is simulated, no network is used and
no raw NEON file is opened. The previous study's builder is read as a style source, never imported or modified; no earlier figure, table or receipt is written to.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from docx import Document
from docx.enum.section import WD_ORIENT, WD_SECTION
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

HERE = Path(__file__).resolve().parent
DATA, FIG, TAB = HERE / "data", HERE / "figures", HERE / "tables"
OLD = HERE.parent / "evaporation_stress_calibration"
FROZEN = HERE.parent.parent.parent.parent / "frozen_report_data_manifest.json"          # task-level manifest of the frozen data hashes (Root)
DOCX = HERE / "SYRUP_JORN_MOAB_joint_d_stress_calibration_20261118.docx"
FIG.mkdir(exist_ok=True)
TAB.mkdir(exist_ok=True)

WINDOWS = ["JORN4_A", "JORN4_B", "MOAB2_C"]
WSHORT = {"JORN4_A": "JORN plot 4, window A", "JORN4_B": "JORN plot 4, window B", "MOAB2_C": "MOAB plot 2, window C"}
SITE = {"JORN4_A": "JORN", "JORN4_B": "JORN", "MOAB2_C": "MOAB"}
LV = ["501", "502", "503"]
GREEN, PURPLE, BLUE = "#2E8B57", "#7B3FA0", "#1F5FA8"
D_SCALE = 0.001


def J(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def pt(x, n=3):
    """volumetric fraction -> volume percentage points"""
    return f"{x * 100:.{n}f}"


def zt(d):
    return np.log1p(np.asarray(d, dtype=float) / D_SCALE)


def arr(x):
    return np.array([np.nan if v is None else v for v in x], dtype=float)


# ----------------------------------------------------------------------------------------------------------------- data
def load():
    R = {"fit": {k: J(DATA / "actual_cpu" / "fits" / f"{k}.json") for k in ("harmonic", "geometric")}, "summary": J(DATA / "actual_cpu" / "summary.json"), "protocol": J(DATA / "actual_cpu" / "protocol.json"),
         "pi": {w: J(DATA / "actual_cpu" / "prepared_inputs" / f"{w}.json") for w in WINDOWS}, "ind": J(DATA / "root_independent_verification.json"), "run": J(DATA / "runtime_cpu.json"),
         "accept": J(DATA / "root_run_acceptance.json"), "helper": J(DATA / "frozen_helper_manifest.json"), "stage1": J(DATA / "stage1_acceptance.json"),
         "old_summary": J(OLD / "data" / "actual_cpu" / "summary.json"), "old_fit": {k: J(OLD / "data" / "actual_cpu" / "fits" / f"{k}.json") for k in ("main", "secondary", "sensitivity_d0")}}
    return R


def sel(R, fit="harmonic"):
    return R["fit"][fit]["selected"]["summary"]


def sensors(R, fit, w, case="selected"):
    s = R["fit"][fit]["selected"]["summary"] if case == "selected" else R["fit"][fit]["controls"][case]["summary"]
    return s["windows"][w]["sensors"]


def bd(R, fit, case="selected"):
    s = R["fit"][fit]["selected"]["summary"] if case == "selected" else R["fit"][fit]["controls"][case]["summary"]
    return s["breakdown"]


def check(R):
    """Hash and number checks: the frozen data are unchanged, the previous study's files match ITS manifest, and every reported number in the assignment is reproduced from the data."""
    fro = J(FROZEN)
    for rel, h in fro.items():
        assert sha(HERE / rel) == h, f"frozen data changed: {rel}"
    om = J(OLD / "artifact_manifest.json")["data"]
    for k in ("main", "secondary", "sensitivity_d0"):
        assert sha(OLD / "data" / "actual_cpu" / "fits" / f"{k}.json") == om[f"actual_cpu/fits/{k}.json"]
    assert sha(OLD / "data" / "actual_cpu" / "summary.json") == om["actual_cpu/summary.json"] and sha(OLD / "artifact_manifest.json") == "4af7b08fc2f1234a2130dabdd82e00b9acc9e28e3ed7fbd4c85388dea3504e23"
    S = R["summary"]["fits"]
    H, G = S["harmonic"], S["geometric"]
    assert H["d"] == 0.0010239461376024856 and H["stress"] == 0.20170581117963332 and math.isclose(H["pooled_rmse"], 0.0037971648344158615)
    assert G["d"] == 0.0010232774828851175 and G["stress"] == 0.20170795278541567 and math.isclose(G["pooled_rmse"], 0.003797236758949968)
    assert H["n"] == G["n"] == 53 and H["n_pairs_evaluated"] == 5103 and G["n_pairs_evaluated"] == 5017 and H["n_local_refinements"] == G["n_local_refinements"] == 8
    A = R["accept"]
    assert A["native_objective_parameter_pairs"] == 10120 == H["n_pairs_evaluated"] + G["n_pairs_evaluated"] and A["native_objective_window_runs"] == 30360 and A["selected_and_control_full_reruns"] == 24 and A["tests"] == 121
    for k in ("harmonic", "geometric"):
        f = R["fit"][k]
        assert not f["search"]["boundary_solution"] and not f["search"]["requires_root_bounded_extension_decision"] and all(m["scipy_success"] for m in f["search"]["local_minima"]) and len(f["search"]["local_minima"]) == 8
        assert f["n_objective_column_runs"] == 3 * f["n_pairs_evaluated"] and f["n_full_rerun_column_runs"] == 12 and f["search"]["n_grid_evaluations"] == 43 * 85
        assert R["ind"][k]["selected_is_lowest_evaluated"] and R["ind"][k]["prepared_inputs_identical_to_previous"] and R["ind"][k]["n"] == 53
        for w in WINDOWS:
            tr = np.array(f["selected"]["native_trajectories"][w]["layer_theta"])
            for i, s in enumerate(sensors(R, k, w)):
                assert np.allclose(arr(s["aligned_model"]), tr[:, i] + s["offset"], atol=1e-14)
    old0, old5 = R["fit"]["harmonic"]["controls"]["previous_sensitivity_d0"], R["fit"]["harmonic"]["controls"]["previous_main"]
    assert math.isclose(old0["summary"]["objective"]["rmse"], 0.0038921035304433273) and math.isclose(old5["summary"]["objective"]["rmse"], 0.009120181829084138)
    assert math.isclose(1 - H["pooled_rmse"] / old0["summary"]["objective"]["rmse"], 0.0244, abs_tol=5e-4) and math.isclose(R["ind"]["harmonic"]["improvement_sse_vs_d0"], 0.04819, abs_tol=1e-5)
    w0, wn = bd(R, "harmonic", "previous_sensitivity_d0")["window"], bd(R, "harmonic")["window"]
    assert math.isclose(wn["JORN4_A"]["rmse"], 0.0041862999, rel_tol=1e-6) and math.isclose(w0["JORN4_A"]["rmse"], 0.0041116556, rel_tol=1e-6)
    assert math.isclose(wn["JORN4_B"]["rmse"], 0.0016553296, rel_tol=1e-6) and math.isclose(w0["JORN4_B"]["rmse"], 0.0020761436, rel_tol=1e-6)
    assert math.isclose(wn["MOAB2_C"]["rmse"], 0.004169054, rel_tol=1e-6) and math.isclose(w0["MOAB2_C"]["rmse"], 0.0044888344, rel_tol=1e-6)
    st0, stn = bd(R, "harmonic", "previous_sensitivity_d0")["site"], bd(R, "harmonic")["site"]
    assert math.isclose(stn["JORN"]["rmse"], 0.0036544468, rel_tol=1e-6) and math.isclose(st0["JORN"]["rmse"], 0.0036542004, rel_tol=1e-6)
    pr = {p["d"]: p for p in R["fit"]["harmonic"]["search"]["profile_over_d"]}
    assert math.isclose(pr[0.0]["objective"] / H["pooled_mse"] - 1, 0.0506, abs_tol=1e-4)
    assert R["fit"]["harmonic"]["search"]["near_optimal"]["fractions"]["0.01"]["d_extent"][1] == 0.001 < H["d"]
    assert math.isclose(0.05 / H["d"], 48.83, abs_tol=0.01)
    assert all(R["summary"]["sources_before"][k]["package_sha256"] == R["summary"]["sources_after"][k]["package_sha256"] for k in ("maple_syrup", "maple"))
    t = {w: (R["pi"][w]["t0_utc"], R["pi"][w]["day_end_utc"][-1], len(R["pi"][w]["day_end_utc"])) for w in WINDOWS}
    assert t == {"JORN4_A": ("2025-04-08T22:00:00Z", "2025-04-18T22:00:00Z", 10), "JORN4_B": ("2025-04-26T06:30:00Z", "2025-04-30T06:30:00Z", 4), "MOAB2_C": ("2025-05-22T03:30:00Z", "2025-05-28T03:30:00Z", 6)}, t
    assert {w: sum(s["n"] for s in sensors(R, "harmonic", w)) for w in WINDOWS} == {"JORN4_A": 28, "JORN4_B": 11, "MOAB2_C": 14}
    for w in WINDOWS:
        cb = R["fit"]["harmonic"]["selected"]["native_trajectories"][w]["cumulative_budget"]
        assert cb["added_m3"] == 0.0 and cb["evaporation_unmet_m3"] == 0.0
    assert "NOT RUN" in A["gpu_status"].upper().replace("NOTRUN", "NOT RUN") or "NOTRUN" in A["gpu_status"]


# --------------------------------------------------------------------------------------------------------------- figures
def zticks(ax, ds=(0, 1e-3, 1e-2, 1e-1, 1.0)):
    lab = [("0" if d == 0 else f"{d:g}") for d in ds]
    ax.set_xticks([float(zt(d)) for d in ds])
    ax.set_xticklabels(lab, fontsize=7)


def fig_surface(R):
    from matplotlib.colors import BoundaryNorm
    f = R["fit"]["harmonic"]
    s = f["search"]
    d, st = np.array(s["d_axis"]), np.array(s["stress_axis"])
    rm = np.sqrt(np.array(s["grid_objective"])) * 100
    z = zt(d)
    sel_ = s["selected"]
    fig, axs = plt.subplots(1, 2, figsize=(10.0, 4.7))
    levels = np.array([0.38, 0.385, 0.39, 0.40, 0.42, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.05, 1.1, 2, 5, 15, 30])
    norm = BoundaryNorm(levels, plt.get_cmap("viridis_r").N, extend="max")
    for ax, zoom in zip(axs, (False, True)):
        cs = ax.contourf(z, st * 100, rm.T, levels=levels, cmap="viridis_r", norm=norm, extend="max")
        ax.contour(z, st * 100, rm.T, levels=levels, colors="white", linewidths=0.35, alpha=0.7)
        gz, gs = np.meshgrid(z, st * 100, indexing="ij")
        ax.plot(gz.ravel(), gs.ravel(), ".", color="black", ms=1.0 if not zoom else 2.4, alpha=0.35, zorder=3)
        ax.plot([0], [20.644317], "D", color=PURPLE, mec="white", ms=7, zorder=6)
        ax.plot([float(zt(0.05))], [21.717030], "s", color=BLUE, mec="white", ms=7, zorder=6)
        ax.plot([float(zt(sel_["d"]))], [sel_["stress"] * 100], "*", color="red", mec="black", ms=13, zorder=7)
        ax.set_xlabel("diffusion multiplier d  (axis log1p(d/0.001); d = 0 at the left edge)", fontsize=7.5)
        ax.tick_params(labelsize=7)
        if zoom:
            zticks(ax, (0, 3e-4, 1e-3, 3e-3, 1e-2, 0.05))
            ax.set_xlim(0, float(zt(0.05)) * 1.02)
            ax.set_ylim(15, 27)
            ax.set_title("(b) zoom: d from 0 to 0.05, θ* 15–27 vol. %", fontsize=9)
        else:
            zticks(ax)
            ax.set_xlim(0, float(zt(1.0)))
            ax.set_ylim(1, 100)
            ax.set_title("(a) whole searched box: d 0–1, θ* 1–100 vol. %", fontsize=9)
            ax.set_ylabel("evaporation-stress threshold θ* (vol. %)", fontsize=8)
    cax = fig.add_axes([0.915, 0.24, 0.014, 0.62])
    cb = fig.colorbar(cs, cax=cax)
    cb.set_label("pooled RMSE of changes (vol. % points)", fontsize=8)
    cb.ax.tick_params(labelsize=6.5)
    h = [plt.Line2D([], [], marker="*", color="w", mfc="red", mec="black", ms=12), plt.Line2D([], [], marker="D", color="w", mfc=PURPLE, mec="white", ms=7), plt.Line2D([], [], marker="s", color="w", mfc=BLUE, mec="white", ms=7),
         plt.Line2D([], [], marker=".", color="gray", ls="none", ms=4)]
    lab = [f"refined joint optimum (d = {sel_['d']:.6f}, θ* = {sel_['stress'] * 100:.3f} %)", "previous fixed-d fit, d = 0", "previous fixed-d fit, d = 0.05", "evaluated grid pairs (43 d × 85 θ*)"]
    fig.legend(h, lab, loc="lower center", ncol=2, fontsize=7.5, frameon=False)
    fig.subplots_adjust(left=0.07, right=0.89, top=0.89, bottom=0.22, wspace=0.2)
    fig.suptitle("Harmonic-interface objective surface from 3655 actual native evaluations (geometric interfaces are nearly identical)", fontsize=9.5)
    fig.savefig(FIG / "fig1_objective_surface.png", dpi=170)
    plt.close(fig)


def fig_profile(R):
    f = R["fit"]["harmonic"]
    pr = f["search"]["profile_over_d"]
    d = np.array([p["d"] for p in pr])
    z = zt(d)
    rm = np.sqrt([p["objective"] for p in pr]) * 100
    sg = np.array([p["stress"] for p in pr]) * 100
    sel_ = f["search"]["selected"]
    best = math.sqrt(sel_["objective"]) * 100
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.9))
    for ax, y, yl, ttl in ((axs[0], rm, "pooled RMSE of changes (vol. % points)", "(a) profile RMSE (θ* re-optimised at each d)"), (axs[1], sg, "best θ* at that d (vol. %)", "(b) profile best θ*")):
        ax.plot(z, y, "-", color="gray", lw=0.8, zorder=1)
        ax.plot(z, y, "o", color="#555555", ms=3.2, zorder=2)
        i0, i5 = 0, int(np.argmin(np.abs(d - 0.05)))
        ax.plot([z[i0]], [y[i0]], "D", color=PURPLE, ms=7, mec="white", zorder=5)
        ax.plot([z[i5]], [y[i5]], "s", color=BLUE, ms=7, mec="white", zorder=5)
        ax.plot([float(zt(sel_["d"]))], [best if ax is axs[0] else sel_["stress"] * 100], "*", color="red", mec="black", ms=13, zorder=6)
        zticks(ax)
        ax.set_xlabel("d  (log1p(d/0.001) axis; 0 at left edge)", fontsize=8)
        ax.set_ylabel(yl, fontsize=8)
        ax.set_title(ttl, fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25)
    axs[0].axhline(math.sqrt(f["comparison"]["zero_change_baseline"]["mse"]) * 100, color="gray", ls="--", lw=1)
    axs[0].text(0.1, math.sqrt(f["comparison"]["zero_change_baseline"]["mse"]) * 100 + 0.015, "zero-change baseline", fontsize=7, color="gray")
    h = [plt.Line2D([], [], marker="o", color="gray", mfc="#555555", ms=4, lw=0.8), plt.Line2D([], [], marker="*", color="w", mfc="red", mec="black", ms=12), plt.Line2D([], [], marker="D", color="w", mfc=PURPLE, ms=7),
         plt.Line2D([], [], marker="s", color="w", mfc=BLUE, ms=7), plt.Line2D([], [], color="gray", ls="--")]
    lab = ["43 grid values of d (line only joins them)", "refined optimum (d = 0.001024)", "d = 0 (previous fixed-d fit)", "d = 0.05 (previous fixed-d fit)", "zero-change baseline"]
    fig.legend(h, lab, loc="lower center", ncol=3, fontsize=7.5, frameon=False)
    fig.subplots_adjust(left=0.09, right=0.985, top=0.88, bottom=0.27, wspace=0.28)
    fig.suptitle("Profile over d (harmonic): lowest RMSE found at each d; no confidence band is implied", fontsize=9)
    fig.savefig(FIG / "fig2_profile_over_d.png", dpi=170)
    plt.close(fig)


def fig_traj(R):
    fig, axs = plt.subplots(3, 3, figsize=(10.0, 7.2))
    for r, w in enumerate(WINDOWS):
        n = len(R["pi"][w]["day_end_utc"])
        t = np.arange(0, n + 1)
        new, o0, o5 = sensors(R, "harmonic", w), sensors(R, "harmonic", w, "previous_sensitivity_d0"), sensors(R, "harmonic", w, "previous_main")
        for c in range(3):
            ax = axs[r, c]
            s = new[c]
            obs = arr(s["observed"])
            ok = np.isfinite(obs)
            series = {}
            for key, ss_, col, ls, lw in (("d = 0.05 fit", o5[c], BLUE, ":", 1.6), ("d = 0 fit", o0[c], PURPLE, "--", 1.4), ("joint fit", s, GREEN, "-", 1.9)):
                y = np.concatenate([[ss_["obs0"]], arr(ss_["aligned_model"])])
                series[key] = y
                ax.plot(t, y, ls=ls, color=col, lw=lw, zorder=3)
            ax.plot(t[1:][ok], obs[ok], "o", mfc="none", mec="black", ms=5.5, ls="none", zorder=4)
            lo = min(np.nanmin(np.concatenate(list(series.values()))), np.nanmin(obs))
            hi = max(np.nanmax(np.concatenate(list(series.values()))), np.nanmax(obs))
            pad = 0.08 * (hi - lo + 1e-9)
            ax.set_ylim(lo - 2.2 * pad, hi + pad)
            if (~ok).any():
                ax.plot(t[1:][~ok], np.full((~ok).sum(), lo - 1.6 * pad), "x", color="gray", ms=5, ls="none", zorder=1)
            ax.plot([0], [s["obs0"]], "o", mfc="none", mec="black", ms=10, mew=1.3, zorder=5)
            ax.plot([0], [s["obs0"]], "^", color=GREEN, mec="black", mew=0.5, ms=6.5, zorder=6)
            ax.set_title(f"{WSHORT[w]} · sensor {LV[c]} at {s['depth_m'] * 100:.1f} cm\noffset {s['offset'] * 100:+.2f} pt, n = {s['n']}", fontsize=7.5)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.25)
            ax.set_xlim(-0.3, n + 0.3)
            if c == 0:
                ax.set_ylabel("θ (m³ m⁻³)", fontsize=8)
            if r == 2:
                ax.set_xlabel("days since t0", fontsize=8)
    h = [plt.Line2D([], [], color=GREEN, lw=1.9), plt.Line2D([], [], color=PURPLE, lw=1.4, ls="--"), plt.Line2D([], [], color=BLUE, lw=1.6, ls=":"), plt.Line2D([], [], marker="o", mfc="none", mec="black", ls="none", ms=6),
         plt.Line2D([], [], marker="o", mfc="none", mec="black", ls="none", ms=10), plt.Line2D([], [], marker="x", color="gray", ls="none", ms=5)]
    lab = ["new joint fit (harmonic)", "previous fixed-d fit, d = 0 (θ* = 20.644 %)", "previous fixed-d fit, d = 0.05 (θ* = 21.717 %)", "observation (valid end-of-day target)",
           "day 0: green triangle on the black circle (coincide by construction)", "no valid observation that day"]
    fig.subplots_adjust(left=0.075, right=0.985, top=0.885, bottom=0.17, hspace=0.62, wspace=0.26)
    fig.suptitle("Offset-aligned model versus observations, all 3 windows × 3 sensors (alignment affects plots and scores only)", fontsize=10, y=0.985)
    fig.legend(h, lab, loc="lower center", ncol=3, fontsize=7, frameon=False, bbox_to_anchor=(0.5, 0.005), columnspacing=1.2, handletextpad=0.4)
    fig.savefig(FIG / "fig3_aligned_windows.png", dpi=170)
    plt.close(fig)


def fig_bars(R):
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    keys = [(w, i) for w in WINDOWS for i in range(3)]
    x = np.arange(len(keys))
    wd = 0.2
    series = [("zero-change baseline", "gray", [sensors(R, "harmonic", w)[i]["zero_change_rmse"] for w, i in keys]), ("d = 0.05 fit", BLUE, [sensors(R, "harmonic", w, "previous_main")[i]["rmse"] for w, i in keys]),
              ("d = 0 fit", PURPLE, [sensors(R, "harmonic", w, "previous_sensitivity_d0")[i]["rmse"] for w, i in keys]), ("joint fit", GREEN, [sensors(R, "harmonic", w)[i]["rmse"] for w, i in keys])]
    for j, (lab, col, v) in enumerate(series):
        ax.bar(x + (j - 1.5) * wd, np.array(v) * 100, wd, color=col, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{w.replace('4_', ' ').replace('2_', ' ')}\n{LV[i]}" for w, i in keys], fontsize=6.5)
    ax.set_ylabel("RMSE of changes (vol. % points)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.25, axis="y")
    ax.legend(fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.17), frameon=False)
    ax.set_title("RMSE by window and sensor (each bar uses only the valid observations of that sensor)", fontsize=9)
    fig.subplots_adjust(left=0.09, right=0.99, top=0.9, bottom=0.27)
    fig.savefig(FIG / "fig4_window_sensor_rmse.png", dpi=170)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------------------- tables
def build_tables(R):
    T = {}
    H, G = R["summary"]["fits"]["harmonic"], R["summary"]["fits"]["geometric"]
    zr = H["zero_change_rmse"]
    old = R["old_summary"]["fits"]
    # setup
    rows = []
    for w in WINDOWS:
        p = R["pi"][w]
        ss_ = sensors(R, "harmonic", w)
        rows.append([WSHORT[w], p["t0_utc"][:16].replace("T", " "), p["day_end_utc"][-1][:16].replace("T", " "), len(p["day_end_utc"]), " / ".join(f"{x * 100:.1f}" for x in p["sensor_depths_m"]),
                     " / ".join(str(s["n"]) for s in ss_), " / ".join(str(s["n_missing"]) for s in ss_), " / ".join(f"{s['offset'] * 100:+.2f}" for s in ss_)])
    T["setup"] = (["Window", "t0 (UTC)", "End (UTC)", "Days", "Sensor depths (cm)", "Valid n (501/502/503)", "Missing", "Fixed offsets (pt)"], rows)
    # fits
    def row(lbl, d, st, rmse, extra=""):
        sse_rel = (rmse / H["pooled_rmse"]) ** 2
        return [lbl, d, f"{st * 100:.4f}", pt(rmse, 4), f"{(1 - (rmse / zr) ** 2) * 100:.1f}", f"{sse_rel:.3f}", extra]
    T["fits"] = (["Fit", "d", "θ* (vol. %)", "Pooled RMSE (pt)", "SSE reduction vs zero-change (%)", "SSE relative to new harmonic fit", "Note"], [
        row("NEW joint, harmonic", f"{H['d']:.7f}", H["stress"], H["pooled_rmse"], "interior optimum"), row("NEW joint, geometric", f"{G['d']:.7f}", G["stress"], G["pooled_rmse"], "interior optimum"),
        row("Previous fixed d = 0, harmonic", "0", old["sensitivity_d0"]["selected_stress"], old["sensitivity_d0"]["pooled_rmse"], "diagnostic, not a default"),
        row("Previous fixed d = 0.05, harmonic", "0.05", old["main"]["selected_stress"], old["main"]["pooled_rmse"], "published main fit"),
        row("Previous fixed d = 0.05, geometric", "0.05", old["secondary"]["selected_stress"], old["secondary"]["pooled_rmse"], "published secondary fit"),
        row("Control: d = 0.05, θ* = 18 %", "0.05", 0.18, R["fit"]["harmonic"]["comparison"]["baseline_d0.05_stress0.18"]["rmse"], "no refit"),
        row("Zero-change baseline", "–", float("nan"), zr, "model change = 0")])
    T["fits"][1][-1][2] = "–"
    # windows
    bw, b0, b5 = bd(R, "harmonic"), bd(R, "harmonic", "previous_sensitivity_d0"), bd(R, "harmonic", "previous_main")
    rows = []
    for lab, key, g in ([(WSHORT[w], w, "window") for w in WINDOWS] + [("Pooled JORN (39 obs.)", "JORN", "site"), ("Pooled MOAB (14 obs.)", "MOAB", "site")]):
        a, b, c = bw[g][key], b0[g][key], b5[g][key]
        rows.append([lab, a["n"], pt(a["zero_change_rmse"]), pt(c["rmse"]), pt(b["rmse"]), pt(a["rmse"]), "better" if a["rmse"] < b["rmse"] - 1e-9 else "slightly worse", f"{pt(a['bias_signed'])} / {pt(b['bias_signed'])}"])
    rows.append(["All windows", 53, pt(zr), pt(old["main"]["pooled_rmse"]), pt(old["sensitivity_d0"]["pooled_rmse"]), pt(H["pooled_rmse"]), "better", "–"])
    T["windows"] = (["Subset", "n", "Zero-change RMSE (pt)", "d = 0.05 fit (pt)", "d = 0 fit (pt)", "Joint fit (pt)", "Joint vs d = 0", "Signed bias joint / d = 0 (pt)"], rows)
    # depth
    rows = []
    for v in LV:
        a, b, c = bw["depth"][v], b0["depth"][v], b5["depth"][v]
        rows.append([f"{v} (JORN ≈ {5.8 + 10 * LV.index(v):.1f} cm, MOAB ≈ {5.5 + 10 * LV.index(v):.1f} cm)", a["n"], pt(a["zero_change_rmse"]), pt(c["rmse"]), pt(b["rmse"]), pt(a["rmse"]), pt(c["bias_signed"]), pt(b["bias_signed"]), pt(a["bias_signed"])])
    T["depth"] = (["Sensor", "n", "Zero-change RMSE", "d = 0.05 RMSE", "d = 0 RMSE", "Joint RMSE", "d = 0.05 bias", "d = 0 bias", "Joint bias"], rows)
    # profile
    pr = {p["d"]: p for p in R["fit"]["harmonic"]["search"]["profile_over_d"]}
    best = H["pooled_mse"]
    want = [0.0, 1e-5, 1e-4, 3.16228e-4, 7.07946e-4, 1e-3, 1.41254e-3, 1.99526e-3, 3.98107e-3, 0.0112202, 0.0316228, 0.05, 0.177828, 1.0]
    keys = sorted({min(pr, key=lambda d, x=x: abs(d - x)) for x in want})
    rows = [[f"{d:.6g}", f"{pr[d]['stress'] * 100:.3f}", pt(math.sqrt(pr[d]["objective"]), 4), f"{(pr[d]['objective'] / best - 1) * 100:+.4f}", pr[d]["n_cached_stresses"]] for d in keys]
    rows.insert(0, [f"{H['d']:.7f} (refined optimum, off-grid)", f"{H['stress'] * 100:.3f}", pt(H["pooled_rmse"], 4), "0 (reference)", "–"])
    T["profile"] = (["d (grid value)", "Best θ* at this d (vol. %)", "Profile RMSE (pt)", "MSE above refined best (%, 4 decimals)", "Distinct θ* evaluated at this d"], rows)
    # grid region
    fr = R["fit"]["harmonic"]["search"]["near_optimal"]["fractions"]
    T["region"] = (["Band (MSE ≤ (1+x) × refined best)", "Grid points inside", "d range of those grid points", "θ* range of those grid points (vol. %)", "Components"], [
        [k, v["n_points"], f"{v['d_extent'][0]:.6g} – {v['d_extent'][1]:.6g}", f"{v['stress_extent'][0] * 100:.2f} – {v['stress_extent'][1] * 100:.2f}", v["n_components"]] for k, v in fr.items()])
    # verification
    vr = R["fit"]["harmonic"]["verification"]
    mx = max(max(abs(x) for x in v["backend_agreement_max_abs"]["numpy_vs_numba"].values()) for v in vr.values()) if vr else 0
    T["verif"] = (["Check", "Result", "Scope"], [
        ["Helper tests and Ruff", f"{R['accept']['tests']} tests PASS; Ruff clean", "new helper + old stress/site suites; synthetic and mocked objectives, no physics qualification"],
        ["Source packages before/after", "maple_syrup 180 files and maple 412 files unchanged", "accepted native sources, hashes in Table B1"],
        ["Pinned raw and air inputs before/after", "unchanged", "same pinned inventory as the previous study"],
        ["Prepared inputs vs previous study", "identical (both fits)", "same initial integrated water, weather and PET"],
        ["Independent exact-fraction scores", "PASS (53 observations, both fits)", "Fraction recomputation by Root; selected is the lowest evaluated pair"],
        ["Native NumPy vs Numba, full trajectory", f"maximum absolute difference over all water, flux and ledger arrays = {mx:g} (3 windows × 2 fits)", "N = 1 column, selected pairs"],
        ["Midpoint public column-state export/reconstruction", "identical continuation (NumPy and Numba)", "driver-level check, NOT a production restart claim"],
        ["Native water ledgers", "per-step and cumulative residuals within bounds (all runs)", "no water added; no unmet evaporation"],
        ["Independent native control reproduction", "PASS (24 controls)", "Root"],
        ["New selected CUDA cases", "NOT RUN", "authorized GPU 0 was occupied (last 2645 MiB, 56 %); GPU 1 not authorized; no GPU child, no fallback"]])
    # sources
    hp = R["helper"]
    rows = [["maple_syrup package", f"{R['summary']['sources_before']['maple_syrup']['files']} files", R["summary"]["sources_before"]["maple_syrup"]["package_sha256"]],
            ["maple package", f"{R['summary']['sources_before']['maple']['files']} files", R["summary"]["sources_before"]["maple"]["package_sha256"]]]
    rows += [[k, "frozen helper", v] for k, v in hp.items()]
    rows += [[f"data/{p.relative_to(DATA)}", "frozen result", sha(p)] for p in sorted(DATA.rglob("*.json")) if "configs" not in p.parts]
    rows += [["previous artifact manifest (immutable e11)", "read-only", sha(OLD / "artifact_manifest.json")]]
    rows += [[f"previous fits/{k}.json", "read-only", sha(OLD / "data" / "actual_cpu" / "fits" / f"{k}.json")] for k in ("main", "secondary", "sensitivity_d0")]
    T["sources"] = (["Item", "Kind", "SHA-256"], rows)
    return T


def write_tables(T):
    for k, (hd, rows) in T.items():
        with open(TAB / f"{k}.csv", "w", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(hd)
            w.writerows(rows)
    (TAB / "report_tables.json").write_text(json.dumps({k: {"header": h, "rows": r} for k, (h, r) in T.items()}, indent=1, default=str) + "\n")


# ---------------------------------------------------------------------------------------------------------------- text
def blocks(R, T):
    H, G = R["summary"]["fits"]["harmonic"], R["summary"]["fits"]["geometric"]
    old = R["old_summary"]["fits"]
    zr = H["zero_change_rmse"]
    r0, r5 = old["sensitivity_d0"]["pooled_rmse"], old["main"]["pooled_rmse"]
    sse_gain = R["ind"]["harmonic"]["improvement_sse_vs_d0"] * 100
    rmse_gain = (1 - H["pooled_rmse"] / r0) * 100
    prof = {p["d"]: p for p in R["fit"]["harmonic"]["search"]["profile_over_d"]}
    d0_excess = (prof[0.0]["objective"] / H["pooled_mse"] - 1) * 100
    bw, b0 = bd(R, "harmonic")["window"], bd(R, "harmonic", "previous_sensitivity_d0")["window"]
    pet = [d["pet_mm_day"] for w in WINDOWS for d in R["pi"][w]["weather_days"]]
    B = []
    a = B.append
    a(("title", "Fitting the diffusion multiplier d and the evaporation-stress threshold together (E = 1)"))
    a(("sub", "SYRUP subsurface diagnostic · joint fit to all three observed JORN/MOAB dry-downs · results of 2026-11-18/19 · calibration, not field validation"))
    a(("h1", "1. Summary"))
    a(("p", ("The previous study fitted one evaporation-stress threshold θ* with the soil-diffusion multiplier d held fixed at 0.05 (and a d = 0 diagnostic). Here we repeat that study with **d fitted as well**, keeping E = 1 exactly, the same layers, hydraulic "
             "properties, weather, initial water and the same 53 valid observations. One common (d, θ*) is fitted to all windows and sites at once, separately for the harmonic and geometric interface-conductivity versions. All three windows are "
             "used for fitting, so **nothing here is a held-out validation**.")))
    a(("bul", [f"**Harmonic fit:** d = **{H['d']:.6f}** (dimensionless), θ* = **{H['stress'] * 100:.3f} vol. %**, pooled RMSE of changes **{pt(H['pooled_rmse'])} percentage points (pt)**. **Geometric fit:** d = {G['d']:.6f}, θ* = {G['stress'] * 100:.3f} vol. %, RMSE {pt(G['pooled_rmse'])} pt. Both optima are interior; no bound was reached or expanded.",
               f"d is about **{0.05 / H['d']:.1f} times smaller** than the earlier fixed 0.05, and the error falls from {pt(r5)} pt (the previous published fit, worse than the zero-change baseline of {pt(zr)} pt) to {pt(H['pooled_rmse'])} pt.",
               f"Against the earlier **d = 0** diagnostic (RMSE {pt(r0)} pt) the improvement is small: **{sse_gain:.2f} % less squared error, but only about {rmse_gain:.2f} % less RMSE**. The difference from d = 0 is modest and does not establish statistically significant positive diffusion; no significance test or confidence interval was computed (the d = 0 profile is {d0_excess:.1f} % above the best in MSE).",
               "The harmonic and geometric fits are nearly identical because the two interface coefficients are close for these profiles, not because exchange is assumed small.",
               "Not every window improves: JORN window A is slightly worse than at d = 0; JORN window B and MOAB window C improve. The shared (d, θ*) is a compromise, not a universal improvement.",
               "d is a dimensionless model multiplier, θ* is an empirical E = 1 parameter (not the measured θ1500), and neither is validated independently. No default changes."]))
    a(("h1", "2. What was fitted and what stayed fixed"))
    a(("p", (f"Everything except d is as in the previous study. The model is SYRUP's actual native column: three 10 cm layers with texture-derived hydraulic surrogates [1], open lower boundary, gravity enabled (inactive here because moisture stays below field capacity), "
             f"no rain, wind, vegetation or bed, T = 0, and potential evaporation from observed-air-temperature Hargreaves PET [3] ({min(pet):.1f}–{max(pet):.1f} mm day⁻¹ in these windows) multiplied by E = 1 and limited by the stress threshold. "
             "The diffusion term is the Tietjen-type moisture-gradient exchange [2]; d scales it, and the actual exchange also depends on the prescribed interface conductivity, geometry and donor-layer thickness.")))
    a(("tab", "setup", "Table 1. The three dry windows (identical to the previous study): dates, sensor depths, valid targets (t0 excluded) and fixed offsets. Pt = volumetric percentage points; offsets are the observed minus model initial values per sensor."))
    a(("p", ("**Scoring.** For each window and sensor we compare model *changes from t0* with observed *changes*, i.e. the model series is shifted by a constant offset so that it starts on the observed initial value. t0 itself is not scored, invalid observations are not filled, and every valid "
             "observation has equal weight (pooled squared error divided by 53). The offsets exist for plots and scores only: the physical initial water in every layer is never changed and no water is added. This is why, in Figure 3, the green model triangle sits exactly on the black observed circle at day 0.")))
    a(("h1", "3. How the search was done"))
    a(("p", (f"Every objective value is an actual native Numba column run on all three windows. Per interface version we evaluated a 43 × 85 grid (d = 0, 41 log-spaced values from 10⁻⁶ to 1, and 0.05; θ* from 1.0001 % to 100 %, including the earlier optima), then "
             f"profiled θ* at every grid d, refined along the box edges, and started {H['n_local_refinements']} bounded Nelder–Mead refinements (6 separated candidates plus the two earlier optima); all 8 reported success for each version. The reported optimum is the lowest value over **every** pair evaluated "
             f"({H['n_pairs_evaluated']} for harmonic, {G['n_pairs_evaluated']} for geometric; {R['accept']['native_objective_parameter_pairs']} in total = {R['accept']['native_objective_window_runs']} native window runs). "
             f"The previous published pairs were re-evaluated, not refitted, and reproduce exactly. A dense grid with multiple starts is evidence, not a proof of a global optimum; the search was confined to d ∈ [0, 1] and θ* ∈ [0.010001, 1].")))
    a(("h1", "4. Results"))
    a(("tab", "fits", "Table 2. Fitted parameters and scores. 'SSE reduction vs zero-change' compares squared error with a model that predicts no change. The last-but-one column is the squared-error ratio, which is the square of the RMSE ratio."))
    a(("p", (f"Because the objective is a *squared* error, percentages of SSE and RMSE differ: the joint fit removes {sse_gain:.2f} % of the d = 0 fit's squared error but only {rmse_gain:.2f} % of its RMSE ({pt(r0, 4)} → {pt(H['pooled_rmse'], 4)} pt). "
             f"Relative to the zero-change baseline ({pt(zr)} pt) the joint fit has {(1 - (H['pooled_rmse'] / zr) ** 2) * 100:.1f} % less SSE than the zero-change reference (a reduction of squared error against predicting no change, not variance explained or an R² against a mean). The earlier fixed d = 0.05 fit does worse than no change at all.")))
    a(("fig", "fig1_objective_surface.png", "Figure 1. Pooled RMSE over the searched (d, θ*) box for harmonic interfaces. The d axis is log1p(d/0.001): approximately linear for d well below 0.001 and logarithmic above it, so d = 0 is drawn explicitly at the left edge. Small dots are the 3655 evaluated grid pairs; the filled shapes are the refined optimum and the two earlier fixed-d optima. The minimum lies near d = 0.001; the best θ* drifts lower as d rises from 0 to about 0.011."))
    a(("fig", "fig2_profile_over_d.png", "Figure 2. Profile over d: at each of the 43 grid values of d, the θ* with the lowest pooled RMSE found, and that θ*. Points are the evaluated d values; the connecting line is only a visual guide. The sampled profile changes little below d ≈ 0.0003, rises above about 0.002 and flattens above d ≈ 0.3 where larger multipliers no longer change the result."))
    a(("p", (f"The profile is the useful way to see how well d is determined. Going from d = 0 to the optimum lowers the RMSE only from {pt(math.sqrt(prof[0.0]['objective']), 3)} to {pt(H['pooled_rmse'], 3)} pt, whereas values above about 0.004 are clearly worse, and the earlier d = 0.05 is far off. "
             "The minimum of the sampled profile is near d = 0.001 with a modest gain over zero; large d is clearly worse. This is a description of the sampled profile (Table 3), not a statement of tightly constrained parameters.")))
    a(("tab", "profile", "Table 3. Profile over d (harmonic) at selected grid values: lowest RMSE and its θ*. 'MSE above refined best' is relative to the off-grid refined optimum and shown to four decimals: the grid value d = 0.001 is about 0.0024 % above it, not exactly zero, so no grid row equals the reference."))
    a(("h1", "5. Where the fit helps and where it does not"))
    a(("tab", "windows", "Table 4. RMSE by window and site for the zero-change baseline, the two earlier fixed-d fits and the joint fit, with signed bias (model − observed change) of the joint and d = 0 fits."))
    a(("p", (f"JORN window A is slightly worse than at d = 0 ({pt(bw['JORN4_A']['rmse'], 4)} versus {pt(b0['JORN4_A']['rmse'], 4)} pt); JORN window B ({pt(bw['JORN4_B']['rmse'], 4)} versus {pt(b0['JORN4_B']['rmse'], 4)}) and MOAB window C ({pt(bw['MOAB2_C']['rmse'], 4)} versus {pt(b0['MOAB2_C']['rmse'], 4)}) improve. "
             "Pooled JORN is essentially unchanged (it holds 39 of 53 observations and is dominated by window A); MOAB improves. With a single shared (d, θ*) the fit trades small losses at one window for gains at others.")))
    a(("figL", "fig3_aligned_windows.png", "Figure 3. Offset-aligned model (green, joint fit) against observations (black open circles) at all nine window/sensor combinations, with the previous d = 0 (purple dashed) and d = 0.05 (blue dotted) fits. Crosses mark days without a valid target (not filled). The green triangle on the black circle at day 0 shows the alignment; the physical model initial water is unchanged."))
    a(("fig", "fig4_window_sensor_rmse.png", "Figure 4. RMSE for each window and sensor for the zero-change baseline and the three fits."))
    a(("tab", "depth", "Table 5. RMSE and signed bias by sensor depth, pooled over windows (pt)."))
    a(("h1", "6. How to read d and θ*"))
    a(("bul", [f"**d is not a diffusivity.** It is a dimensionless multiplier on a moisture-gradient exchange whose size also depends on the prescribed interface hydraulic conductivity (m s⁻¹, from the approved texture-derived property recipe [1]) and on the explicitly specified 10 cm layer geometry and donor-layer thickness (m), which are user-specified rather than derived from texture. A value of {H['d']:.4f} must not be read as m² s⁻¹, and fitting d says nothing about having estimated or validated Ks.",
               "**A small positive d is not established.** The improvement over d = 0 is modest and no significance test or confidence interval was computed; the fit sums every layer of three short, correlated time series (not independent replicates), and the fixed offsets carry the noise of the initial measurements into every score. We do not infer statistically significant diffusion, universal soil properties or an independent validation.",
               "**The fit is conditional.** It depends on the texture-derived hydraulic properties, the moisture-gradient law, Hargreaves PET and the omission of transpiration, roots, litter and site heterogeneity; any of these can be compensated by d or θ*.",
               "**θ* stays an empirical E = 1 parameter.** With moisture well below θ* only E/θ*² is identified by evaporation in these windows; fixing E = 1 selects one conditional scale. It is not the measured θ1500 or a wilting point and does not change the retention curve or Ks. No E ≠ 1 run or fit was made. The value moved from 21.7 % (d = 0.05) to 20.2 % when d was freed.",
               "**Interface versions agree.** Harmonic and geometric fits differ by about 0.00001 pt in RMSE and about 0.0002 vol. % in θ*; this reflects the nearly equal interface coefficients, not a small-exchange assumption."]))
    a(("h1", "7. Verification and what was not run"))
    a(("tab", "verif", "Table 6. Verification of the actual CPU joint fits."))
    a(("p", (f"Root measured {R['run']['elapsed_including_startup_s']:.1f} s for the fit and its CPU verification on one column, with the timer started after the Python, NumPy, Numba and SciPy imports and with existing, previously warmed Numba caches (peak resident memory {R['run']['peak_rss_kib'] / 1024:.0f} MiB). Cache loading and JIT compilation costs were not separated, so this is not a cold-start figure; it is not a storm-timing or scaling benchmark. (The frozen receipt's historical field name is `elapsed_including_startup_s`; its bytes are unchanged.) **The new selected CUDA cases were not run**: the authorized GPU 0 was occupied and no other device is authorized, "
             "so nothing here claims CPU/GPU equivalence for the new fit. The earlier accepted GPU tests at the old parameters remain historical. The midpoint state check exercises the public column-state interface, not a production restart.")))
    a(("h1", "8. Limits and honest reading"))
    a(("bul", ["All 53 observations (only 14 at MOAB) were used for fitting; a lower error than the earlier controls is expected and is not validation.",
               "The search is bounded (d ≤ 1, θ* ≥ 0.010001); the optimum is interior but a dense grid plus multi-start refinement does not prove global optimality.",
               "The windows are dry, rain-free and below field capacity, so gravity drainage, behaviour after rain, near saturation and over longer periods are untested.",
               "Sensors at 5.5–5.8 cm are compared with a 0–10 cm layer mean; representativeness differences remain after the offset.",
               "This is a calibration diagnostic. No model default, hydraulic property or source was changed."]))
    a(("h1", "References"))
    a(("refs", [(1, "Saxton, K. E. and Rawls, W. J. (2006). Soil water characteristic estimates by texture and organic matter for hydrologic solutions. Soil Science Society of America Journal 70, 1569–1578.", "https://doi.org/10.2136/sssaj2005.0117"),
                (2, "Tietjen, B., Zehe, E. and Jeltsch, F. (2009). Simulating plant water availability in dry lands under climate change: a generic model of two soil layers. Water Resources Research 45, W01418.", "https://doi.org/10.1029/2007WR006589"),
                (3, "Hargreaves, G. H. and Samani, Z. A. (1985). Reference crop evapotranspiration from temperature. Applied Engineering in Agriculture 1(2), 96–99.", "https://doi.org/10.13031/2013.26773"),
                (4, "NEON. Soil water content and water salinity, DP1.00094.001 (RELEASE-2026).", "https://data.neonscience.org/data-products/DP1.00094.001"),
                (5, "NEON. Single aspirated air temperature, DP1.00002.001 (RELEASE-2026).", "https://data.neonscience.org/data-products/DP1.00002.001"),
                (6, "NEON. Megapit soil characterization, DP1.00096.001 (RELEASE-2026), source of the texture/density inputs.", "https://data.neonscience.org/data-products/DP1.00096.001"),
                (7, "SYRUP project. Pooled E = 1 evaporation-stress calibration (previous report, 2026-11-17), docs/subsurface_hydrology/evaporation_stress_calibration; fixed-d fits read unchanged.", "")]))
    a(("p", "The physics follows the native SYRUP column code identified in Appendix B; it adopts Saxton–Rawls retention-derived properties as prescribed inputs and the Tietjen-type moisture-gradient exchange, not the full SR unsaturated-conductivity equations. The results are this project's own calibration, not primary field validation."))
    a(("h1", "Appendix A. Near-optimal grid points (descriptive)"))
    a(("p", ("Table A1 lists grid points whose MSE is within 1 % or 5 % of the **refined** best. These are *sampled grid values only*: they are not continuous parameter bounds or confidence intervals and need not contain the refined optimum. In particular the 1 % grid points reach only d = 0.001, "
             f"slightly below the refined best d = {H['d']:.6f}, simply because the grid has no value between 0.001 and 0.00141. The 5 % band almost reaches d = 0 but omits it by construction of this arbitrary MSE criterion (the d = 0 profile is {d0_excess:.2f} % above the best). This is a descriptive threshold, not a confidence test, and it cannot establish statistical exclusion of d = 0.")))
    a(("tab", "region", "Table A1. Grid-sampled near-optimal points for harmonic interfaces (descriptive only)."))
    a(("h1", "Appendix B. Source and receipt identities"))
    a(("p", (f"Production sources were unchanged before and after the runs (maple_syrup 180 files, maple 412 files), as were the pinned raw/air inputs and the {R['accept']['old_project_files_preserved']} existing project files. Python {R['run']['python'].split()[0]}, NumPy {R['run']['numpy']}, Numba {R['run']['numba']}, SciPy {R['run']['scipy']}. "
             "The new helper was frozen before the run and is hashed below, together with the frozen result files and the previous study's read-only inputs.")))
    a(("tab", "sources", "Table B1. Source packages, helper files, frozen data and read-only previous-study files."))
    return B


# ---------------------------------------------------------------------------------------------------------------- docx
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
    for nm, base in (("JointTitle", "Title"), ("JointSubtitle", "Subtitle"), ("JointCaption", "Normal"), ("JointTableText", "Normal")):
        s = st.add_style(nm, WD_STYLE_TYPE.PARAGRAPH)
        s.base_style = st[base]
    t = st["JointTitle"]
    t.font.name, t.font.size, t.font.bold = "Calibri", Pt(20), True
    t.font.color.rgb = RGBColor.from_string("1F3A5F")
    t.paragraph_format.space_after = Pt(8)
    s = st["JointSubtitle"]
    s.font.size, s.font.italic = Pt(11), True
    c = st["JointCaption"]
    c.font.size, c.font.italic = Pt(9), True
    c.paragraph_format.space_after = Pt(10)
    c.paragraph_format.keep_together = True
    tt = st["JointTableText"]
    tt.font.size = Pt(8)
    tt.paragraph_format.space_after = Pt(0)
    tt.paragraph_format.line_spacing = 1.0


def add_table(doc, hd, rows, width_in, caption):
    cp = doc.add_paragraph(style="JointCaption")
    runs(cp, caption)
    cp.paragraph_format.keep_with_next = True
    cp.paragraph_format.space_after = Pt(3)
    t = doc.add_table(rows=1, cols=len(hd))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    wts = [min(max(len(str(h)) * 0.5, max(len(str(r[i])) for r in rows) if rows else 4, 5), 34) for i, h in enumerate(hd)]
    ws = [Inches(width_in * w / sum(wts)) for w in wts]
    for i, h in enumerate(hd):
        c = t.rows[0].cells[i]
        c.text = ""
        c.paragraphs[0].style = "JointTableText"
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
            cells[i].paragraphs[0].style = "JointTableText"
            cells[i].paragraphs[0].add_run(str(v))
            cells[i].width = ws[i]
            if ri % 2:
                shade(cells[i], "F3F6FA")
    doc.add_paragraph().paragraph_format.space_after = Pt(4)


def section(doc, landscape):
    ns = doc.add_section(WD_SECTION.NEW_PAGE)
    if landscape:
        ns.orientation = WD_ORIENT.LANDSCAPE
        ns.page_width, ns.page_height = Inches(11), Inches(8.5)
        ns.left_margin = ns.right_margin = Inches(0.6)
        ns.top_margin = ns.bottom_margin = Inches(0.6)
    else:
        ns.orientation = WD_ORIENT.PORTRAIT
        ns.page_width, ns.page_height = Inches(8.5), Inches(11)
        ns.left_margin = ns.right_margin = Inches(0.9)
        ns.top_margin = ns.bottom_margin = Inches(0.85)


def to_docx(B, T, path):
    doc = Document()
    setup_styles(doc)
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Inches(8.5), Inches(11)
    sec.left_margin = sec.right_margin = Inches(0.9)
    sec.top_margin = sec.bottom_margin = Inches(0.85)
    fp = sec.footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    fr = fp.add_run("SYRUP JORN/MOAB joint d + stress calibration · diagnostic fit to all windows, not field validation · page ")
    fr.font.size = Pt(8)
    add_field(fp, "PAGE")
    in_land = False
    for b in B:
        k = b[0]
        if k != "figL" and in_land:
            section(doc, False)
            in_land = False
        if k == "title":
            doc.add_paragraph(b[1], style="JointTitle")
        elif k == "sub":
            doc.add_paragraph(b[1], style="JointSubtitle")
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
            p.add_run().add_picture(str(FIG / b[1]), width=Inches(6.7))
            runs(doc.add_paragraph(style="JointCaption"), b[2])
        elif k == "figL":
            section(doc, True)
            in_land = True
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.keep_with_next = True
            p.add_run().add_picture(str(FIG / b[1]), width=Inches(8.6))
            runs(doc.add_paragraph(style="JointCaption"), b[2])
        elif k == "refs":
            for n, c, u in b[1]:
                p = doc.add_paragraph()
                p.paragraph_format.left_indent = Inches(0.25)
                p.paragraph_format.first_line_indent = Inches(-0.25)
                p.add_run(f"[{n}] ").bold = True
                p.add_run(f"{c} {u}".strip()).font.size = Pt(9)
    if in_land:
        section(doc, False)
    doc.core_properties.title = "SYRUP JORN/MOAB joint d + evaporation-stress calibration (E = 1)"
    doc.core_properties.author = "SYRUP project"
    doc.save(path)


def verify_docx(path, T, B):
    z = zipfile.ZipFile(path)
    names = z.namelist()
    media = [n for n in names if n.startswith("word/media/")]
    xml = z.read("word/document.xml").decode()
    ntab, nhdr = xml.count("<w:tbl>"), xml.count("<w:tblHeader")
    nfig = sum(1 for b in B if b[0] in ("fig", "figL"))
    ntabs = sum(1 for b in B if b[0] == "tab")
    pagef = "PAGE" in "".join(z.read(n).decode(errors="ignore") for n in names if "footer" in n)
    text = re.sub(r"<[^>]+>", " ", xml)
    assert z.testzip() is None and len(media) == nfig and ntab == ntabs == nhdr and pagef
    assert not re.search(r"TODO|TBD|XXX|placeholder|\bnan\b", text)
    return {"zip_ok": True, "media": len(media), "tables": ntab, "header_rows_repeat": nhdr, "sections": xml.count("<w:sectPr"), "paragraphs": xml.count("<w:p>") + xml.count("<w:p "), "bytes": Path(path).stat().st_size, "sha256": sha(path)}


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
        elif k in ("fig", "figL"):
            out.append(f"![{b[2]}](figures/{b[1]})\n")
        elif k == "refs":
            out.append("\n".join(f"[{n}] {c} {u}".strip() for n, c, u in b[1]) + "\n")
    return "\n".join(out)


def main():
    R = load()
    check(R)
    fig_surface(R)
    fig_profile(R)
    fig_traj(R)
    fig_bars(R)
    T = build_tables(R)
    write_tables(T)
    B = blocks(R, T)
    to_docx(B, T, DOCX)
    info = verify_docx(DOCX, T, B)
    (HERE / "report.md").write_text(to_markdown(B, T))
    (HERE / "verification.md").write_text("# Verification record\n\nStatic checks of the generated DOCX (zip integrity, one media file per figure, table count equals header-row count, footer page field, no placeholder text). "
                                          "No Office renderer was available, so rendered pagination and figure legibility were not checked by this builder.\n\n```\n" + json.dumps(info, indent=1) + "\n```\n\n"
                                          "`check()` re-verified before building: every frozen data hash against the task's frozen manifest, the previous study's fit files against its own manifest (itself pinned to the immutable e11 sha256), and the reported numbers "
                                          "(joint optima, RMSEs, 53 observations and 28/11/14 split, 5103 + 5017 = 10120 pairs, 30360 window runs, 24 full reruns, 8 successful refinements per fit, per-window and per-site RMSEs, profile excess at d = 0, "
                                          "grid near-region extent below the refined d, and the aligned = raw + offset identity for every sensor). New selected CUDA cases were NOT run (GPU 0 occupied, GPU 1 not authorized).\n")
    man = {"docx": {"file": DOCX.name, **info}, "figures": {p.name: sha(p) for p in sorted(FIG.glob("*.png"))}, "tables": {p.name: sha(p) for p in sorted(TAB.iterdir())}, "builder": sha(Path(__file__)), "report_md": sha(HERE / "report.md"),
           "data": {str(p.relative_to(DATA)): sha(p) for p in sorted(DATA.rglob("*.json"))}}
    (HERE / "artifact_manifest.json").write_text(json.dumps(man, indent=1) + "\n")
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
