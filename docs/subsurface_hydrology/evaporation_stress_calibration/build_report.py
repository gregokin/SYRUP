"""Builds the pooled E = 1 evaporation-stress calibration report: figures, tables, editable Markdown, verification note, manifest and the Word document.

    PYTHONPATH=<task>/report_tools MPLCONFIGDIR=/tmp/syrup-stress-report PYTHONDONTWRITEBYTECODE=1 /home/okin/MAPLE/.venv/bin/python -B build_report.py

Every number is computed here from the frozen JSON in `data/` (Root's actual CPU fits, actual CUDA verification and independent scores). Nothing is simulated, no network is used and no raw NEON file is opened.
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
DOCX = HERE / "SYRUP_JORN_MOAB_E1_stress_calibration_20261117.docx"
FIG.mkdir(exist_ok=True)
TAB.mkdir(exist_ok=True)

WINDOWS = ["JORN4_A", "JORN4_B", "MOAB2_C"]
WSHORT = {"JORN4_A": "JORN plot 4, window A", "JORN4_B": "JORN plot 4, window B", "MOAB2_C": "MOAB plot 2, window C"}
SITE = {"JORN4_A": "JORN", "JORN4_B": "JORN", "MOAB2_C": "MOAB"}
FITS = ["main", "secondary", "sensitivity_d0"]
FLABEL = {"main": "Main: harmonic, d 0.05", "secondary": "Secondary: geometric, d 0.05", "sensitivity_d0": "Diagnostic: d 0 (harmonic)"}
FCOL = {"main": "#1F5FA8", "secondary": "#D9822B", "sensitivity_d0": "#7B3FA0"}
LV = ["501", "502", "503"]
GREEN = "#2E8B57"


def J(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def vp(x, n=2):
    """volumetric fraction -> volume percent / percentage points"""
    return f"{x * 100:.{n}f}"


def f4(x):
    return f"{x:.4f}"


# ----------------------------------------------------------------------------------------------------------------- data
def load():
    R = {"fit": {k: J(DATA / "actual_cpu" / "fits" / f"{k}.json") for k in FITS}, "summary": J(DATA / "actual_cpu" / "summary.json"), "pi": {w: J(DATA / "actual_cpu" / "prepared_inputs" / f"{w}.json") for w in WINDOWS},
         "cuda": {k: J(DATA / "actual_cuda" / f"{k}.json") for k in FITS}, "cuda_done": J(DATA / "actual_cuda" / "completion.json"), "root": J(DATA / "root_independent_scores.json"),
         "accept": J(DATA / "root_run_acceptance.json"), "cpu_done": J(DATA / "root_cpu_completion.json"), "stage1": J(DATA / "stage1_root_acceptance.json"), "ident": J(DATA / "root_stress_E_identifiability.json")}
    return R


def sensors(R, fit, w):
    return R["fit"][fit]["selected"]["windows"][w]["sensors"]


def arr(x):
    return np.array([np.nan if v is None else v for v in x], dtype=float)


def check(R):
    """Assert the numbers the assignment reports against the data; the report text is generated from the data itself."""
    S, A = R["summary"]["fits"], R["accept"]
    assert S["main"]["selected_stress"] == 0.21717029762566153 and S["sensitivity_d0"]["selected_stress"] == 0.20644317480382438 and S["secondary"]["selected_stress"] == 0.2171797728919738
    assert math.isclose(S["main"]["pooled_rmse"], 0.009120181829084138) and math.isclose(S["secondary"]["pooled_rmse"], 0.009120727861051134) and math.isclose(S["sensitivity_d0"]["pooled_rmse"], 0.0038921035304433273)
    assert math.isclose(S["main"]["zero_change_rmse"], 0.008860437827250189)
    assert A["observations"] == 53 and A["actual_native_objective_evaluations"] == 624 and A["actual_native_window_runs_for_optimization"] == 1872 and A["tests"] == 83 and A["cuda_configuration_checks"] == 27
    for k in FITS:
        assert S[k]["n_evaluations"] == 208 and S[k]["n_local_minima"] == 1 and not S[k]["boundary_solution"] and R["root"][k]["selected_is_lowest_evaluated"]
        for w in WINDOWS:                                                         # aligned = raw + offset exactly, raw = the actual native layer series
            tr = np.array(R["fit"][k]["selected_native_trajectories"][w]["layer_theta"])
            for i, s in enumerate(sensors(R, k, w)):
                assert np.allclose(arr(s["aligned_model"]), tr[:, i] + s["offset"], atol=1e-14)
    n_by = {w: sum(s["n"] for s in sensors(R, "main", w)) for w in WINDOWS}
    assert n_by == {"JORN4_A": 28, "JORN4_B": 11, "MOAB2_C": 14}
    for i in (1, 2):                                                              # d = 0: middle and deep layers stay constant, so they equal the zero-change baseline
        b = R["fit"]["sensitivity_d0"]["selected"]["breakdown"]["depth"][LV[i]]
        assert math.isclose(b["rmse"], b["zero_change_rmse"], rel_tol=1e-12)
    for w in WINDOWS:
        cb = R["fit"]["main"]["selected_native_trajectories"][w]["cumulative_budget"]
        assert cb["drainage_m3"] == 0.0 and cb["added_m3"] == 0.0 and cb["evaporation_unmet_m3"] == 0.0
    for k in FITS:
        i = R["ident"][k]
        assert i["theta_below_both_018_and_fit"] and i["max_top_theta"] < 0.18 and math.isclose(i["analytical_E_equivalent_at_018"], (0.18 / i["fitted_stress"]) ** 2, rel_tol=1e-9) and i["fitted_stress"] == R["summary"]["fits"][k]["selected_stress"]
    w = {x: (R["pi"][x]["t0_utc"], R["pi"][x]["day_end_utc"][-1], len(R["pi"][x]["day_end_utc"])) for x in WINDOWS}
    assert w == {"JORN4_A": ("2025-04-08T22:00:00Z", "2025-04-18T22:00:00Z", 10), "JORN4_B": ("2025-04-26T06:30:00Z", "2025-04-30T06:30:00Z", 4), "MOAB2_C": ("2025-05-22T03:30:00Z", "2025-05-28T03:30:00Z", 6)}, w
    assert R["cuda_done"]["cuda_configurations"] == 27 and R["cuda_done"]["actual_selected_cuda_compared_to_saved_cpu"]
    assert all(R["summary"]["sources_before"][k]["package_sha256"] == R["summary"]["sources_after"][k]["package_sha256"] for k in ("maple_syrup", "maple"))


def derive(R):
    S = R["summary"]["fits"]
    D = {"stress": {k: S[k]["selected_stress"] for k in FITS}, "rmse": {k: S[k]["pooled_rmse"] for k in FITS}, "base": S["main"]["zero_change_rmse"], "n": S["main"]["n"]}
    D["skill"] = {k: 1 - (D["rmse"][k] / D["base"]) ** 2 for k in FITS}
    D["sse_red"] = {k: 1 - (D["rmse"][k] / D["base"]) ** 2 for k in FITS}
    D["ctl"] = {k: {c: R["fit"][k]["controls"][c]["objective"]["rmse"] for c in ("baseline_0.18", "site_theta1500")} for k in FITS}
    assert math.isclose(D["ctl"]["main"]["baseline_0.18"], 0.009362146708395531) and math.isclose(D["ctl"]["sensitivity_d0"]["site_theta1500"], 0.017830781821668244)
    return D


# --------------------------------------------------------------------------------------------------------------- figures
def fig_profile(R, D):
    fig, axs = plt.subplots(1, 2, figsize=(7.3, 3.7))
    for ax, zoom in zip(axs, (False, True)):
        for k in FITS:
            p = R["fit"][k]["search"]["profile"]
            x, y = np.array(p["stress"]), np.sqrt(np.array(p["objective"])) * 100
            ax.plot(x, y, "--" if k == "secondary" else "-", color=FCOL[k], lw=1.4 if k != "secondary" else 1.8, label=FLABEL[k] + (" [overlaps main]" if k == "secondary" else ""))
            ax.plot([D["stress"][k]], [D["rmse"][k] * 100], "o", color=FCOL[k], ms=7, mec="black", zorder=5)
            ax.plot([0.18], [D["ctl"][k]["baseline_0.18"] * 100], "s", color=FCOL[k], ms=5, mfc="white", zorder=4)
        ax.axhline(D["base"] * 100, color="gray", ls="--", lw=1)
        ax.axvline(0.18, color="gray", ls=":", lw=0.8)
        ax.set_xlabel("evaporation-stress threshold θ* (m³ m⁻³)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25)
        if zoom:
            ax.set_xlim(0.1, 0.4)
            ax.set_ylim(0, 1.6)
            ax.set_title("(b) zoom on the minima", fontsize=9)
        else:
            ax.set_title("(a) whole searched range 0.010001–1", fontsize=9)
            ax.set_ylabel("pooled RMSE of changes (vol. % points)", fontsize=8)
    axs[1].text(0.105, D["base"] * 100 - 0.1, "zero-change baseline", fontsize=7, ha="left", color="gray")
    h, lab = axs[0].get_legend_handles_labels()
    h += [plt.Line2D([], [], marker="o", color="w", mfc="gray", mec="black", ms=7), plt.Line2D([], [], marker="s", color="w", mfc="white", mec="gray", ms=5), plt.Line2D([], [], color="gray", ls="--")]
    lab += ["selected minimum", "no-refit control at θ* = 0.18", "zero-change baseline (constant initial moisture)"]
    fig.legend(h, lab, loc="lower center", ncol=2, fontsize=7, frameon=False)
    fig.suptitle("Pooled objective versus the shared stress threshold (208 actual native evaluations per fit, 53 observations)", fontsize=9)
    fig.tight_layout(rect=(0, 0.17, 1, 0.94))
    fig.savefig(FIG / "fig1_objective_profile.png", dpi=170)
    plt.close(fig)


def fig_traj(R, fit, name, title, raw=False, size=(10.0, 6.6)):
    if raw:
        size = (10.0, 7.4)
    fig, axs = plt.subplots(3, 3, figsize=size)
    for r, w in enumerate(WINDOWS):
        pi = R["pi"][w]
        n = len(pi["day_end_utc"])
        t = np.arange(0, n + 1)
        tr = np.array(R["fit"][fit]["selected_native_trajectories"][w]["layer_theta"])
        for c, s in enumerate(sensors(R, fit, w)):
            ax = axs[r, c]
            obs = arr(s["observed"])
            ok = np.isfinite(obs)
            al = np.concatenate([[s["obs0"]], arr(s["aligned_model"])])
            ax.plot(t, al, "-", color=GREEN, lw=1.8, zorder=3)
            if raw:
                rw = np.concatenate([[s["model0"]], tr[:, c]])
                ax.plot(t, rw, "--", color=GREEN, lw=1.2, alpha=0.8, zorder=2)
            ax.plot(t[1:][ok], obs[ok], "o", mfc="none", mec="black", ms=5.5, ls="none", zorder=4)
            if (~ok).any():
                ax.plot(t[1:][~ok], np.full((~ok).sum(), ax.get_ylim()[0]), "x", color="gray", ms=4.5, ls="none", clip_on=False, zorder=1)
            if raw:
                ax.plot([0], [s["model0"]], "^", color=GREEN, mec=GREEN, mfc="none", ms=7, zorder=5)
            ax.plot([0], [s["obs0"]], "o", mfc="none", mec="black", ms=10, mew=1.3, zorder=5)
            ax.plot([0], [s["obs0"]], "^", color=GREEN, mec="black", mew=0.5, ms=6.5, zorder=6)
            ax.set_title(f"{WSHORT[w]} · sensor {LV[c]} at {s['depth_m'] * 100:.1f} cm" + f"\noffset {s['offset'] * 100:+.2f} pt, n = {s['n']}", fontsize=7.5)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.25)
            ax.set_xlim(-0.3, n + 0.3)
            if c == 0:
                ax.set_ylabel("θ (m³ m⁻³)", fontsize=8)
            if r == 2:
                ax.set_xlabel("days since t0", fontsize=8)
    h = [plt.Line2D([], [], color=GREEN, lw=1.8), plt.Line2D([], [], marker="o", mfc="none", mec="black", ls="none", ms=6), plt.Line2D([], [], marker="^", color="w", mfc=GREEN, mec="black", ms=7),
         plt.Line2D([], [], marker="o", mfc="none", mec="black", ls="none", ms=10), plt.Line2D([], [], marker="x", color="gray", ls="none", ms=5)]
    lab = ["model, aligned (change + observed initial value)", "observation (valid end-of-day target)", "model initial (green triangle) ...", "... drawn on the observed initial value (black circle): they coincide", "no valid observation that day"]
    if raw:
        h.insert(1, plt.Line2D([], [], color=GREEN, lw=1.2, ls="--"))
        lab.insert(1, "model, RAW (unshifted physical column)")
        h.append(plt.Line2D([], [], marker="^", color="w", mfc="none", mec=GREEN, ms=7))
        lab.append("raw model initial (integrated layer mean)")
    if raw:                                                                      # explicit layout: reserved title and two-row legend areas, room for the two-line panel titles
        fig.subplots_adjust(left=0.075, right=0.985, top=0.885, bottom=0.165, hspace=0.62, wspace=0.26)
        fig.suptitle(title, fontsize=10, y=0.985)
        fig.legend(h, lab, loc="lower center", ncol=3, fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, 0.005))
    else:
        fig.legend(h, lab, loc="lower center", ncol=3, fontsize=7.5, frameon=False)
        fig.suptitle(title, fontsize=10)
        fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    fig.savefig(FIG / name, dpi=170)
    plt.close(fig)


def fig_depth(R, D):
    fig, axs = plt.subplots(1, 2, figsize=(7.3, 3.4))
    x = np.arange(3)
    wd = 0.26
    bd = {k: R["fit"][k]["selected"]["breakdown"]["depth"] for k in ("main", "sensitivity_d0")}
    zero = [bd["main"][v]["zero_change_rmse"] * 100 for v in LV]
    for j, (lab, col, vals) in enumerate([("zero-change baseline", "gray", zero), ("main fit (d 0.05)", FCOL["main"], [bd["main"][v]["rmse"] * 100 for v in LV]),
                                         ("d 0 fit", FCOL["sensitivity_d0"], [bd["sensitivity_d0"][v]["rmse"] * 100 for v in LV])]):
        axs[0].bar(x + (j - 1) * wd, vals, wd, color=col, label=lab)
    axs[1].bar(x - wd / 2, [bd["main"][v]["bias_signed"] * 100 for v in LV], wd, color=FCOL["main"], label="main fit (d 0.05)")
    axs[1].bar(x + wd / 2, [bd["sensitivity_d0"][v]["bias_signed"] * 100 for v in LV], wd, color=FCOL["sensitivity_d0"], label="d 0 fit")
    axs[1].axhline(0, color="k", lw=0.6)
    for a, t, yl in ((axs[0], "(a) RMSE of changes by sensor", "RMSE (vol. % points)"), (axs[1], "(b) signed bias of changes", "mean (model − observed change), pt")):
        a.set_xticks(x)
        a.set_xticklabels([f"{v}\n(JORN ≈ {5.8 + 10 * i:.1f} cm,\nMOAB ≈ {5.5 + 10 * i:.1f} cm)" for i, v in enumerate(LV)], fontsize=7)
        a.set_title(t, fontsize=9)
        a.set_ylabel(yl, fontsize=7.5)
        a.tick_params(labelsize=7)
        a.grid(alpha=0.25, axis="y")
        a.legend(fontsize=7)
    fig.suptitle("Where the misfit sits: per-depth RMSE and signed bias (all 53 observations)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIG / "fig5_depth_bias.png", dpi=170)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------------------- tables
def build_tables(R, D):
    T = {}
    S = R["summary"]["fits"]
    T["fits"] = (["Fit", "Interface", "d", "Selected stress θ* (m³ m⁻³)", "(vol. %)", "Pooled RMSE (pt)", "Zero-change RMSE (pt)", "SSE reduction vs zero change", "n", "Evaluations (grid + refine)"],
                 [[FLABEL[k].split(":")[0], S[k]["method"], f"{S[k]['d']:g}", f"{D['stress'][k]:.5f}", vp(D["stress"][k]), f"{D['rmse'][k] * 100:.3f}", f"{D['base'] * 100:.3f}",
                   f"{D['sse_red'][k] * 100:+.1f} %", S[k]["n"], f"{S[k]['n_evaluations']} (199 + {S[k]['n_evaluations'] - 199})"] for k in FITS])
    T["dates"] = (["Window", "Site / plot", "t0 (UTC)", "Last modelled end (UTC)", "Daily updates", "Valid targets", "Missing targets", "Sensor depths (cm)"],
                  [[w, WSHORT[w].split(",")[0], R["pi"][w]["t0_utc"], R["pi"][w]["day_end_utc"][-1], len(R["pi"][w]["day_end_utc"]), sum(x["n"] for x in sensors(R, "main", w)), sum(x["n_missing"] for x in sensors(R, "main", w)),
                    " / ".join(f"{z * 100:.1f}" for z in R["pi"][w]["sensor_depths_m"])] for w in WINDOWS])
    T["ident"] = (["Fit", "Fitted θ*", "Modelled top-layer θ, min–max", "Below 0.18 and θ*?", "Equivalent E at θ* = 0.18 (algebra only)"],
                  [[FLABEL[k].split(":")[0], f"{D['stress'][k]:.4f}", f"{R['ident'][k]['min_top_theta']:.4f}–{R['ident'][k]['max_top_theta']:.4f}", "yes, every day", f"{R['ident'][k]['analytical_E_equivalent_at_018']:.4f}"] for k in FITS])
    T["controls"] = (["Fit", "At selected θ*", "Control θ* = 0.18", "Control site θ1500 (JORN 0.0395, MOAB 0.0590)", "Zero change"],
                     [[FLABEL[k].split(":")[0], f"{D['rmse'][k] * 100:.3f}", f"{D['ctl'][k]['baseline_0.18'] * 100:.3f}", f"{D['ctl'][k]['site_theta1500'] * 100:.3f}", f"{D['base'] * 100:.3f}"] for k in FITS])
    rows = []
    for w in WINDOWS:
        row = [WSHORT[w], sum(s["n"] for s in sensors(R, "main", w))]
        for k in ("main", "sensitivity_d0"):
            b = R["fit"][k]["selected"]["breakdown"]["window"][w]
            row += [f"{b['rmse'] * 100:.3f}", f"{b['bias_signed'] * 100:+.3f}"]
        row.append(f"{R['fit']['main']['selected']['breakdown']['window'][w]['zero_change_rmse'] * 100:.3f}")
        rows.append(row)
    for site in ("JORN", "MOAB"):
        b0, b1 = (R["fit"][k]["selected"]["breakdown"]["site"][site] for k in ("main", "sensitivity_d0"))
        rows.append([f"{site} (all windows)", b0["n"], f"{b0['rmse'] * 100:.3f}", f"{b0['bias_signed'] * 100:+.3f}", f"{b1['rmse'] * 100:.3f}", f"{b1['bias_signed'] * 100:+.3f}", f"{b0['zero_change_rmse'] * 100:.3f}"])
    T["windows"] = (["Window / site", "n", "Main RMSE (pt)", "Main bias (pt)", "d 0 RMSE (pt)", "d 0 bias (pt)", "Zero-change RMSE (pt)"], rows)
    rows = []
    for v in LV:
        b0, b1 = (R["fit"][k]["selected"]["breakdown"]["depth"][v] for k in ("main", "sensitivity_d0"))
        rows.append([f"{v} (all sites)", b0["n"], f"{b0['rmse'] * 100:.3f}", f"{b0['bias_signed'] * 100:+.3f}", f"{b1['rmse'] * 100:.3f}", f"{b1['bias_signed'] * 100:+.3f}", f"{b0['zero_change_rmse'] * 100:.3f}"])
    T["depth"] = (["Vertical sensor", "n", "Main RMSE (pt)", "Main bias (pt)", "d 0 RMSE (pt)", "d 0 bias (pt)", "Zero-change RMSE (pt)"], rows)
    rows = []
    for w in WINDOWS:
        for s in sensors(R, "main", w):
            rows.append([WSHORT[w], s["label"], f"{s['n']} / {s['n'] + s['n_missing']}", f"{s['offset'] * 100:+.3f}", f"{s['rmse'] * 100:.3f}", f"{s['rmse_unshifted'] * 100:.3f}", f"{s['bias_signed'] * 100:+.3f}", f"{s['zero_change_rmse'] * 100:.3f}"])
    T["detail"] = (["Window", "Sensor @ depth", "Valid / targets", "Offset (pt)", "RMSE of changes (pt)", "Unshifted RMSE (pt)", "Bias of changes (pt)", "Zero-change RMSE (pt)"], rows)
    rows = []
    for k in FITS:
        iv = R["fit"][k]["near_optimal"]["intervals"]
        rows.append([FLABEL[k].split(":")[0], f"{D['stress'][k]:.4f}", " ; ".join(f"{a:.4f}–{b:.4f}" for a, b in iv["0.01"]), " ; ".join(f"{a:.4f}–{b:.4f}" for a, b in iv["0.05"])])
    T["bands"] = (["Fit", "Selected θ*", "Grid points with MSE within +1 % of the minimum", "... within +5 %"], rows)
    cu = R["cuda_done"]
    mx = 0.0
    for k in FITS:
        for grp in R["cuda"][k]["verification"].values():
            for win in grp.values():
                for pair in win["backend_agreement_max_abs"].values():
                    mx = max(mx, max(pair.values()))
    T["verif"] = (["Check", "Result"],
                  [["Optimisation", f"{R['accept']['actual_native_objective_evaluations']} stress evaluations (208 per fit) × 3 windows = {R['accept']['actual_native_window_runs_for_optimization']} actual compiled-Numba column runs"],
                   ["Per-step and cumulative water ledger", "checked inside every native run (residual far below the native bound); no water added, drainage 0, unmet evaporation 0 in the selected main trajectories"],
                   ["NumPy vs Numba, midpoint public-state export/reconstruction", "selected solutions and E = 1 controls: PASS (identical trajectories after reconstruction)"],
                   ["CUDA", f"{cu['cuda_configurations']} configurations (3 fits × selected + 2 controls × 3 windows) on {cu['gpu_name']}, CuPy {cu['cupy']}; selected CUDA trajectories match the saved CPU ones; largest NumPy/Numba/CUDA difference over all compared fields {mx:.1e}"],
                   ["Independent exact scoring (Root)", "Fraction-arithmetic recomputation of all 53 observations for all 3 fits: PASS; selected stress is the lowest evaluated in every fit"],
                   ["Unit tests", f"{R['accept']['tests']} tests pass; the 180 production SYRUP modules and 412 MAPLE files are unchanged; the {R['accept']['production_original_files_unchanged']} pre-existing tracked project files were also verified untouched"],
                   ["Scope", "single-column selected-case equivalence; not GPU performance, not scaling, not production restart/lifecycle checkpoint qualification"]])
    sb = R["summary"]["sources_before"]
    rows = [["maple_syrup package", f"{sb['maple_syrup']['files']} files", sb["maple_syrup"]["package_sha256"]], ["maple package", f"{sb['maple']['files']} files", sb["maple"]["package_sha256"]]]
    for rel in ("actual_cpu/summary.json", "actual_cpu/fits/main.json", "actual_cpu/fits/secondary.json", "actual_cpu/fits/sensitivity_d0.json", "actual_cuda/completion.json", "root_independent_scores.json", "root_run_acceptance.json"):
        rows.append([rel, "data file", sha(DATA / rel)])
    for w in WINDOWS:
        rows.append([f"actual_cpu/prepared_inputs/{w}.json", "prepared inputs", sha(DATA / "actual_cpu" / "prepared_inputs" / f"{w}.json")])
    for rel, h in R["cpu_done"]["helper_before"].items():
        rows.append([rel, "frozen helper", h])
    T["sources"] = (["Item", "Kind", "SHA-256"], rows)
    return T


def write_tables(T):
    for k, (hd, rows) in T.items():
        with open(TAB / f"{k}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(hd)
            w.writerows(rows)
    (TAB / "report_tables.json").write_text(json.dumps({k: {"header": h, "rows": r} for k, (h, r) in T.items()}, indent=1, default=str) + "\n")


# ---------------------------------------------------------------------------------------------------------------- text
def blocks(R, D, T):
    S = R["summary"]["fits"]
    st = D["stress"]
    bm, b0 = R["fit"]["main"]["selected"]["breakdown"], R["fit"]["sensitivity_d0"]["selected"]["breakdown"]
    off = {w: [s["offset"] for s in sensors(R, "main", w)] for w in WINDOWS}
    miss = {w: sum(s["n_missing"] for s in sensors(R, "main", w)) for w in WINDOWS}
    pi = {w: R["pi"][w] for w in WINDOWS}
    dj, dm = pi["JORN4_A"]["sensor_depths_m"], pi["MOAB2_C"]["sensor_depths_m"]
    pet = {w: [d["pet_mm_day"] for d in pi[w]["weather_days"]] for w in WINDOWS}
    R["fit"]["main"]["physical_flags"]
    lay = R["fit"]["main"]["selected_config"]["JORN4_A"]["layers"]
    lay_m = R["fit"]["main"]["selected_config"]["MOAB2_C"]["layers"]
    B = []
    a = B.append
    a(("title", "Calibrating one evaporation-stress threshold to JORN and MOAB dry-downs (E = 1)"))
    a(("sub", "SYRUP subsurface diagnostic · pooled fit to all three observed windows · results of 2026-11-17/18 · not field validation"))
    a(("h1", "1. Summary"))
    a(("p", ("We kept the soil-evaporation factor at **E = 1** and asked a single question: which one **evaporation-stress threshold** θ* makes SYRUP's actual native column reproduce the observed *changes* in moisture at the three NEON dry-down windows "
            "(JORN plot 4 windows A and B, MOAB plot 2 window C) when everything else is held at the values used in the earlier hydraulics comparison? One threshold was fitted to all windows and sites together, and the three windows "
            "were all used for the fit, so **there is no held-out claim**.")))
    a(("bul", [f"**Main fit** (harmonic interface mixing, d = 0.05): θ* = **{st['main']:.4f}** m³ m⁻³ ({vp(st['main'])} vol. %), pooled RMSE of changes **{vp(D['rmse']['main'], 3)}** percentage points (pt).",
               f"**Secondary fit** (geometric mixing, d = 0.05): θ* = {st['secondary']:.4f}, RMSE {vp(D['rmse']['secondary'], 3)} pt, practically the same as the main fit.",
               f"**Diagnostic fit** (d = 0): θ* = **{st['sensitivity_d0']:.4f}**, RMSE **{vp(D['rmse']['sensitivity_d0'], 3)}** pt.",
               (f"The benchmark to beat is the **zero-change baseline** (the model simply keeps the initial moisture): RMSE {vp(D['base'], 3)} pt. The main fit is **worse** than that baseline (it has {abs(D['sse_red']['main']) * 100:.1f} % MORE squared error than the no-change reference, skill {D['skill']['main']:+.3f}); only the d = 0 fit beats it "
               f"(SSE reduction {D['sse_red']['sensitivity_d0'] * 100:.1f} %)."),
               "All three optima (one per fit) are interior (not at the 0.010001 or 1 limits) and each fit found exactly one local minimum on the searched grid. Near-optimum bands are descriptive, not confidence intervals."]))
    a(("p", ("**What this does and does not mean.** The fit calibrates an *empirical* stress threshold. It is not a measured wilting point, not the retention at 1500 kPa, and it does not change any hydraulic property. It is not an overall good field validation, and "
            "we do not recommend d = 0 as a new production default: the strong dependence of the score on d says that interlayer redistribution, not the stress threshold, controls most of the misfit in these windows.")))
    a(("h1", "2. What was fitted and what was held fixed"))
    a(("p", ("The setup is the one used for the harmonic/geometric comparison: the same three 10 cm layers, the same texture-derived JORN and MOAB properties with their declared floors, the same observed temperature and Hargreaves daily PET, the same observed rain-free windows and the same "
            "integrated initial water profiles. Each run uses a flat terrain, an open bottom boundary and gravity redistribution, with no rain, wind, vegetation, bed change or transpiration (T = 0). Nothing in this list was altered for the fit.")))
    a(("bul", ["**E = 1** is held constant (never fitted). **d** is never fitted: 0.05 is the main and secondary value, and d = 0 is a diagnostic that switches the interlayer diffusion off, not a tuned value.",
               "**One calibrated stress for all windows and sites in each fit** — not one per site. The two controls are separate runs with no refit: θ* = 0.18 and each site's own predicted θ1500 (JORN 0.0395, MOAB 0.0590).",
               (f"The search covers θ* from 0.010001 (the native column requires the threshold to exceed the 0.01 residual) to 1.0: a 199-point grid including both ends, a bounded local refinement of every local minimum, and a comparison of all {S['main']['n_evaluations']} evaluated "
               "values. Every evaluation is the actual compiled-Numba native column on the three windows (three prepared once, evaluations cached).")]))
    a(("tab", "dates", "Table 1. The three observed windows (all times UTC; initial times differ and are not midnight; every window is a rain-free run with daily end-of-window updates of 86400 s)."))
    a(("h2", "How the stress threshold acts"))
    a(("p", "Only the top layer evaporates. Its evaporation demand is the potential rate times E, scaled by a moisture factor f that depends on the layer's volumetric moisture θ, its residual θr = 0.01 and the threshold θ*:"))
    a(("eq", "f(θ) = 0 for θ ≤ θr;   f(θ) = (θ / θ*)² for θr < θ < θ*;   f(θ) = 1 for θ ≥ θ*"))
    a(("eq", "E_actual = min( f(θ) · E · PET · Δt ,  water above the residual in the layer )"))
    a(("p", (f"So a larger θ* means evaporation starts to be throttled at wetter conditions. Changing θ* alters only this factor: it does **not** change Ks, field capacity, saturation, bulk density, the interface conductivity or the original Saxton–Rawls θ1500 anchor that these properties were derived with. "
            f"A fitted θ* of {st['main']:.3f} therefore does **not** mean the soil holds 21 % water at the limit or that the dry floor is 0.2; the residual stays 0.01. In every window the moisture stays far below field capacity "
            f"(top-layer field capacity {vp(lay['theta_field_capacity'][0], 1)} % at JORN and {vp(lay_m['theta_field_capacity'][0], 1)} % at MOAB), so the calibrated θ* lies below FC and saturation (checked in the data).")))
    a(("h2", "What the data can and cannot identify: E versus the stress threshold"))
    a(("p", "In every window the modelled top-layer moisture stays below both the fitted threshold and 0.18 (Table 2). In that regime the evaporation demand is E × PET × θ² / θ*², so the observations constrain only the **combination E / θ*²**, not E and θ* separately. "
            "Fixing E = 1 therefore gives a valid *conditional* estimate of θ*, but it cannot tell an error in E apart from an error in the threshold, and it cannot validate θ* as a physical moisture limit or θ1500. "
            "Algebraically, θ* = 0.18 with E ≈ " + f"{R['ident']['main']['analytical_E_equivalent_at_018']:.3f}" + " (main), " + f"{R['ident']['secondary']['analytical_E_equivalent_at_018']:.3f}" + " (secondary) or " + f"{R['ident']['sensitivity_d0']['analytical_E_equivalent_at_018']:.3f}" + " (d = 0) would give the same evaporation response while all states stay in this regime. "
            "This is analytical equivalence only: no fit with E ≠ 1 and no extra run was made, and every actual fit used E = 1. Here the stress threshold acts as an amplitude calibration of evaporation rather than as an identified moisture physiology."))
    a(("tab", "ident", "Table 2. Evidence for the E / θ*² degeneracy (analytical equivalence; no E ≠ 1 run was made)."))
    a(("h1", "3. How the score works"))
    a(("p", ("The model's initial layer value is the *integrated* mean of the observed point profile over the 10 cm layer, while the observation is a single point sensor. That constant support difference should not be scored as an evaporation error. "
            "The score therefore compares **changes** from the initial instant t0:")))
    a(("eq", "residual(t) = [ θ_model(t) − θ_model(0) ] − [ θ_obs(t) − θ_obs(0) ]"))
    a(("p", ("Equivalently, the model is plotted as θ_model(t) + offset, with offset = θ_obs(0) − θ_model(0), so the green triangle (model initial) and the black circle (observed initial) coincide. The offset is a fixed constant for each window and sensor, set by the two initial values; it is not optimised "
            "and it is not a mean-bias shift. It is applied only for scoring and plotting: **the physical model state, water inventory and water budget are never shifted**, and no water is added. The raw model series and the aligned series are both kept (Figure 4).")))
    a(("bul", [f"t0 is excluded from the objective (including it would add forced zeros). Only post-initial targets with a valid, quality-passed observation count: **{D['n']} points** — JORN 39 (window A 28, window B 11) and MOAB 14. Targets that failed quality control are simply not counted ({miss['JORN4_A']}, {miss['JORN4_B']} and {miss['MOAB2_C']} missing sensor-days in windows A, B and C), never filled.",
               "The pooled objective is the sum of all squared residuals divided by the number of valid points, so each observation has equal weight (it is not an average of window RMSEs).",
               (f"Sensors are the NEON vertical positions 501, 502 and 503. They sit at {vp(dj[0], 1)}, {vp(dj[1], 1)} and {vp(dj[2], 1)} cm at JORN and {vp(dm[0], 1)}, {vp(dm[1], 1)} and {vp(dm[2], 1)} cm at MOAB (different actual depths, so we aggregate by sensor ID and keep each window's actual depth label). "
               "The model value compared with a sensor is the whole 10 cm layer containing it, so the 'surface' layer is 0–10 cm, not a 5.5–5.8 cm point.")]))
    a(("h1", "4. Results"))
    a(("tab", "fits", "Table 3. Selected stress thresholds and pooled scores (RMSE in volumetric percentage points, pt; 1 pt = 0.01 m³ m⁻³)."))
    a(("fig", "fig1_objective_profile.png", ("Figure 1. Pooled RMSE of changes against the shared stress threshold for the three fits. The geometric curve (dashed) lies on top of the harmonic one. Filled circles are the selected minima, squares the θ* = 0.18 control and the dashed line the zero-change baseline. "
                                          "The site-specific θ1500 control uses two different thresholds (JORN and MOAB), so it cannot be drawn at one abscissa; it is in Table 4.")))
    a(("p", (f"The profile has one clear minimum in each fit, at θ* ≈ 0.217 for d = 0.05 and ≈ 0.206 for d = 0. With d = 0.05 the minimum RMSE ({vp(D['rmse']['main'], 3)} pt) never gets below the zero-change baseline ({vp(D['base'], 3)} pt) at any threshold; with d = 0 the whole minimum is far lower. "
            f"The harmonic and geometric fits are nearly identical (θ* differs by {abs(st['main'] - st['secondary']):.1e}, RMSE by {abs(D['rmse']['main'] - D['rmse']['secondary']) * 100:.1e} pt), because the harmonic and geometric interface coefficients are close for the actual neighbouring-layer conductivities. This is not because the exchange is small: at a given d the exchange dominates the misfit (section 5).")))
    a(("tab", "controls", "Table 4. Pooled RMSE (pt) at the selected stress and for the two no-refit controls (same E = 1 and offset rule)."))
    a(("p", (f"Calibration helps relative to the older choices: for the main fit the RMSE drops from {vp(D['ctl']['main']['baseline_0.18'], 3)} pt at 0.18 and {vp(D['ctl']['main']['site_theta1500'], 3)} pt with the site θ1500 thresholds to {vp(D['rmse']['main'], 3)} pt. "
            "Using the (very low) site θ1500 as the stress threshold is clearly the worst of the three, which is a statement about this empirical parameter, not about the measured wilting point.")))
    a(("tab", "windows", "Table 5. Scores by window and site. Bias is the mean of (model change − observed change); positive means the model dries less (or wets more) than observed."))
    a(("tab", "depth", "Table 6. Scores by vertical sensor ID (all windows pooled)."))
    a(("tab", "bands", "Table 7. Descriptive near-optimum bands from the sampled grid. These are not confidence intervals and their endpoints are grid points, not precise continuous limits."))
    a(("p", (f"The 1 % MSE band is {R['fit']['main']['near_optimal']['intervals']['0.01'][0][0]:.4f}–{R['fit']['main']['near_optimal']['intervals']['0.01'][0][1]:.4f} for the main fit and "
            f"{R['fit']['sensitivity_d0']['near_optimal']['intervals']['0.01'][0][0]:.4f}–{R['fit']['sensitivity_d0']['near_optimal']['intervals']['0.01'][0][1]:.4f} for d = 0. Each fit found a single minimum at this grid resolution; we make no claim of a mathematical global proof outside the searched grid and bounds.")))
    a(("h1", "5. Why the main fit is worse than doing nothing, and what d = 0 reveals"))
    a(("p", (f"The d = 0.05 score is dominated by *redistribution*, not evaporation. In the d = 0 diagnostic the middle and deep layers stay exactly constant (no drainage because moisture is below field capacity, and no exchange), so their RMSE equals the zero-change baseline "
            f"({vp(b0['depth']['502']['zero_change_rmse'], 2)} and {vp(b0['depth']['503']['zero_change_rmse'], 2)} pt). Only the top layer changes, and the fit then reduces the top-sensor RMSE from {vp(b0['depth']['501']['zero_change_rmse'], 2)} to {vp(b0['depth']['501']['rmse'], 2)} pt: "
            f"evaporation with θ* ≈ {st['sensitivity_d0']:.2f} captures much of the observed surface drying. That cuts the pooled SSE by {D['sse_red']['sensitivity_d0'] * 100:.1f} %.")))
    a(("p", (f"With d = 0.05 the model moves water between layers. The signed biases show the result (Figure 5, Table 6): the top layer is too wet in the changes ({vp(bm['depth']['501']['bias_signed'], 2)} pt) while the middle and deep layers are too dry ({vp(bm['depth']['502']['bias_signed'], 2)} and {vp(bm['depth']['503']['bias_signed'], 2)} pt). "
            f"This is consistent with the deeper layers exchanging water with the layers above and below, whereas the observed deeper sensors change only weakly. The middle-layer RMSE rises from {vp(bm['depth']['502']['zero_change_rmse'], 2)} pt (zero change) to {vp(bm['depth']['502']['rmse'], 2)} pt. "
            "This does not show that there is no real diffusion or no plant transpiration in the field; it shows that, with the present exchange coefficient and without plants, the modelled redistribution does not match the weak observed deep changes in these three windows.")))
    a(("fig", "fig5_depth_bias.png", "Figure 5. Per-sensor RMSE (a) and signed bias (b) of changes for the main and d = 0 fits. JORN and MOAB depths differ by about 0.3 cm; the labels give approximate depths."))
    a(("h1", "6. Aligned model and observations"))
    a(("p", ("The next figures show, for every window (rows) and sensor (columns), the aligned model (green line) and the valid observations (open black circles). Observations are drawn only at valid targets and are never joined across a missing day; grey crosses at the bottom of a panel mark days with no valid observation. "
            "The green filled triangle is the model initial value and the black open circle the observed initial value: both are plotted at the same position by construction. The panel titles give the actual sensor depth, the fixed offset and the number of valid points.")))
    a(("figL", "fig2_aligned_main.png", "Figure 2. Main fit (harmonic, d = 0.05, θ* = " + f"{st['main']:.4f}" + "): aligned model versus observations in θ (m³ m⁻³)."))
    a(("figL", "fig3_aligned_d0.png", "Figure 3. Diagnostic d = 0 fit (θ* = " + f"{st['sensitivity_d0']:.4f}" + "): same layout. Middle and deep layers are constant lines; only the top layer evaporates."))
    a(("figL", "fig4_raw_vs_aligned.png", "Figure 4. Main fit, raw physical column (dashed, hollow triangle at the integrated layer mean) beside the aligned series (solid). The only difference is the constant offset; the physical state and budget are unchanged."))
    a(("p", "The offsets are small (pt): " + "; ".join(f"{WSHORT[w]}: " + ", ".join(f"{o * 100:+.2f}" for o in off[w]) for w in WINDOWS) + ". They remove the initial point-versus-layer difference but not later support differences or initial sensor noise."))
    a(("h1", "7. Model verification"))
    a(("tab", "verif", "Table 8. What was verified (actual computation, not analytic substitutes)."))
    a(("h1", "8. Limits and honest reading"))
    a(("bul", ["**E and θ* are not separately identified**: with moisture below the threshold only E / θ*² is constrained (section 2). The fitted θ* is a conditional amplitude calibration at E = 1, not a validated moisture limit or θ1500.",
               "All three windows (53 points, only 14 at MOAB) were used for the fit, so a smaller error than the controls is expected and is not validation. There is no independent test of the fitted threshold.",
               ("The fitted θ* is a lumped empirical parameter. It can absorb errors in the Hargreaves PET (the same observed temperature but a flat, simple formula) "
               f"(PET {min(min(p) for p in pet.values()):.1f}–{max(max(p) for p in pet.values()):.1f} mm/day in these windows), in the E = 1 assumption, in the layer properties, and in missing processes (plant transpiration is omitted, roots and litter are absent)."),
               "A fixed initial offset removes a constant point-versus-layer bias but does not fix representativeness differences later in the window, or noise in the initial sensor value; sensors at 5.5–5.8 cm are compared with a 0–10 cm layer.",
               "The strong sensitivity to d (RMSE 0.91 versus 0.39 pt) means the stress threshold cannot be interpreted alone. We did not tune d and recommend no change of defaults.",
               "Moisture never exceeds field capacity in these windows, so gravity drainage is inactive; behaviour near saturation, after rain, and over longer periods is untested here.",
               "CUDA was checked on one column for selected cases only: no GPU fitting, performance, scaling or production restart claim."]))
    a(("h1", "References"))
    refs = [(1, "Saxton, K. E. and Rawls, W. J. (2006). Soil water characteristic estimates by texture and organic matter for hydrologic solutions. Soil Science Society of America Journal 70, 1569–1578.", "https://doi.org/10.2136/sssaj2005.0117"),
            (2, "Tietjen, B., Zehe, E. and Jeltsch, F. (2009). Simulating plant water availability in dry lands under climate change: a generic model of two soil layers. Water Resources Research 45, W01418.", "https://doi.org/10.1029/2007WR006589"),
            (3, "Hargreaves, G. H. and Samani, Z. A. (1985). Reference crop evapotranspiration from temperature. Applied Engineering in Agriculture 1(2), 96–99.", "https://doi.org/10.13031/2013.26773"),
            (4, "NEON (National Ecological Observatory Network). Soil water content and water salinity, DP1.00094.001 (RELEASE-2026).", "https://data.neonscience.org/data-products/DP1.00094.001"),
            (5, "NEON. Single aspirated air temperature, DP1.00002.001 (RELEASE-2026).", "https://data.neonscience.org/data-products/DP1.00002.001"),
            (6, "NEON. Precipitation (tipping bucket), DP1.00045.001 (RELEASE-2026). Used for the rain-free windows in the project's pinned inventory.", "https://data.neonscience.org/data-products/DP1.00045.001"),
            (7, "NEON. Megapit soil characterization, DP1.00096.001 (RELEASE-2026), source of the texture/density inputs to the derived properties.", "https://data.neonscience.org/data-products/DP1.00096.001")]
    a(("refs", refs))
    a(("h1", "Appendix A. Per-window and per-depth detail (main fit)"))
    a(("tab", "detail", "Table A1. Offsets, counts and scores by window and sensor. 'Unshifted RMSE' compares the raw model with the observation without the offset. Pt = volumetric percentage points."))
    a(("h1", "Appendix B. Source and receipt identities"))
    a(("p", (f"Production sources were unchanged before and after the runs (maple_syrup 180 files, maple 412 files); the {R['accept']['production_original_files_unchanged']} pre-existing tracked project files (not Python modules only) were also verified untouched. The calibration helper and tests are frozen and hashed below. "
            f"The fits and the independent exact-fraction check used {R['accept']['observations']} observations. Hardware for the CUDA check: {R['cuda_done']['gpu_name']}, CuPy {R['cuda_done']['cupy']}.")))
    a(("tab", "sources", "Table B1. Source packages, frozen helper files and data receipts."))
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
    for nm, base in (("StressTitle", "Title"), ("StressSubtitle", "Subtitle"), ("StressCaption", "Normal"), ("StressEquation", "Normal"), ("StressTableText", "Normal")):
        s = st.add_style(nm, WD_STYLE_TYPE.PARAGRAPH)
        s.base_style = st[base]
    t = st["StressTitle"]
    t.font.name, t.font.size, t.font.bold = "Calibri", Pt(21), True
    t.font.color.rgb = RGBColor.from_string("1F3A5F")
    t.paragraph_format.space_after = Pt(8)
    s = st["StressSubtitle"]
    s.font.size, s.font.italic = Pt(11), True
    c = st["StressCaption"]
    c.font.size, c.font.italic = Pt(9), True
    c.paragraph_format.space_after = Pt(10)
    c.paragraph_format.keep_together = True
    e = st["StressEquation"]
    e.font.name, e.font.size = "Cambria Math", Pt(11)
    e.paragraph_format.left_indent = Inches(0.4)
    tt = st["StressTableText"]
    tt.font.size = Pt(8)
    tt.paragraph_format.space_after = Pt(0)
    tt.paragraph_format.line_spacing = 1.0


def add_table(doc, hd, rows, width_in, caption):
    cp = doc.add_paragraph(style="StressCaption")
    runs(cp, caption)
    cp.paragraph_format.keep_with_next = True
    cp.paragraph_format.space_after = Pt(3)
    t = doc.add_table(rows=1, cols=len(hd))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    wts = [min(max(len(str(h)) * 0.5, max(len(str(r[i])) for r in rows) if rows else 4, 5), 40) for i, h in enumerate(hd)]
    ws = [Inches(width_in * w / sum(wts)) for w in wts]
    for i, h in enumerate(hd):
        c = t.rows[0].cells[i]
        c.text = ""
        c.paragraphs[0].style = "StressTableText"
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
            cells[i].paragraphs[0].style = "StressTableText"
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
    fr = fp.add_run("SYRUP JORN/MOAB E = 1 stress calibration · diagnostic fit to all windows, not field validation · page ")
    fr.font.size = Pt(8)
    add_field(fp, "PAGE")
    in_land = False
    for b in B:
        k = b[0]
        if k != "figL" and in_land:
            section(doc, False)
            in_land = False
        if k == "title":
            doc.add_paragraph(b[1], style="StressTitle")
        elif k == "sub":
            doc.add_paragraph(b[1], style="StressSubtitle")
        elif k == "h1":
            doc.add_heading(b[1], level=1)
        elif k == "h2":
            doc.add_heading(b[1], level=2)
        elif k == "p":
            runs(doc.add_paragraph(), b[1])
        elif k == "eq":
            doc.add_paragraph(b[1], style="StressEquation")
        elif k == "bul":
            for x in b[1]:
                runs(doc.add_paragraph(style="List Bullet"), x)
        elif k == "tab":
            hd, rows = T[b[1]]
            if len(hd) > 8:
                section(doc, True)
                add_table(doc, hd, rows, 9.6, b[2])
                section(doc, False)
            else:
                add_table(doc, hd, rows, 6.7, b[2])
        elif k == "fig":
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.keep_with_next = True
            p.add_run().add_picture(str(FIG / b[1]), width=Inches(6.7))
            runs(doc.add_paragraph(style="StressCaption"), b[2])
        elif k == "figL":
            if not in_land:
                section(doc, True)
                in_land = True
            else:
                doc.add_page_break()
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.keep_with_next = True
            p.add_run().add_picture(str(FIG / b[1]), width=Inches(9.2))
            runs(doc.add_paragraph(style="StressCaption"), b[2])
        elif k == "refs":
            for n, c, u in b[1]:
                p = doc.add_paragraph()
                p.paragraph_format.left_indent = Inches(0.25)
                p.paragraph_format.first_line_indent = Inches(-0.25)
                p.add_run(f"[{n}] ").bold = True
                p.add_run(f"{c} {u}").font.size = Pt(9)
    doc.core_properties.title = "SYRUP JORN/MOAB E=1 pooled evaporation-stress calibration"
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
    assert not re.search(r"TODO|TBD|XXX|placeholder|nan\b", text)
    return {"zip_ok": True, "media": len(media), "tables": ntab, "header_rows_repeat": nhdr, "sections": xml.count("<w:sectPr"), "paragraphs": xml.count("<w:p>") + xml.count("<w:p "), "bytes": Path(path).stat().st_size, "sha256": sha(path)}


def to_markdown(B, T):
    out = []
    for b in B:
        k = b[0]
        if k == "title":
            out.append(f"# {b[1]}\n")
        elif k == "sub":
            out.append(f"*{b[1]}*\n")
        elif k in ("h1", "h2"):
            out.append(("## " if k == "h1" else "### ") + b[1] + "\n")
        elif k in ("p", "eq"):
            out.append(b[1] + "\n")
        elif k == "bul":
            out.append("\n".join(f"- {x}" for x in b[1]) + "\n")
        elif k == "tab":
            hd, rows = T[b[1]]
            out.append(f"**{b[2]}**\n\n| " + " | ".join(hd) + " |\n|" + "---|" * len(hd) + "\n" + "\n".join("| " + " | ".join(str(c) for c in r) + " |" for r in rows) + "\n")
        elif k in ("fig", "figL"):
            out.append(f"![{b[2]}](figures/{b[1]})\n")
        elif k == "refs":
            out.append("\n".join(f"[{n}] {c} {u}" for n, c, u in b[1]) + "\n")
    return "\n".join(out)


def main():
    R = load()
    check(R)
    D = derive(R)
    fig_profile(R, D)
    fig_traj(R, "main", "fig2_aligned_main.png", "Main fit: aligned model versus observations (3 windows × 3 sensors)")
    fig_traj(R, "sensitivity_d0", "fig3_aligned_d0.png", "d = 0 diagnostic fit: aligned model versus observations (3 windows × 3 sensors)")
    fig_traj(R, "main", "fig4_raw_vs_aligned.png", "Main fit: raw physical model (dashed) and offset-aligned model (solid)", raw=True)
    fig_depth(R, D)
    T = build_tables(R, D)
    write_tables(T)
    B = blocks(R, D, T)
    to_docx(B, T, DOCX)
    info = verify_docx(DOCX, T, B)
    (HERE / "report.md").write_text(to_markdown(B, T))
    (HERE / "verification.md").write_text("# Verification record\n\nStatic checks of the generated DOCX (zip integrity, one media file per figure, table count equals header-row count, footer page field, no placeholder text). "
                                          "No Office renderer was available, so rendered pagination and figure legibility were not checked by this builder.\n\n```\n" + json.dumps(info, indent=1) + "\n```\n\n"
                                          "Reported values are asserted against the frozen data in `check()`: selected stresses, pooled RMSEs, zero-change baseline, counts (53; 28/11/14), 208 evaluations and one interior minimum per fit, "
                                          "624 objective evaluations and 1872 window runs, and the aligned = raw + offset identity for every sensor.\n")
    man = {"docx": {"file": DOCX.name, **info}, "figures": {p.name: sha(p) for p in sorted(FIG.glob("*.png"))}, "tables": {p.name: sha(p) for p in sorted(TAB.iterdir())},
           "builder": sha(Path(__file__)), "report_md": sha(HERE / "report.md"), "data": {str(p.relative_to(DATA)): sha(p) for p in sorted(DATA.rglob("*.json"))}}
    (HERE / "artifact_manifest.json").write_text(json.dumps(man, indent=1) + "\n")
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
