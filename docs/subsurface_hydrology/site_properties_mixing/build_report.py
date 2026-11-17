"""Builds the JORN/MOAB harmonic-vs-geometric interface-mixing report: editable Markdown, figures, tables, verification note and the Word document.

    PYTHONPATH=<task>/report_tools MPLCONFIGDIR=/tmp/syrup-site-mixing-mpl PYTHONDONTWRITEBYTECODE=1 \
        /home/okin/MAPLE/.venv/bin/python -B build_report.py

Every number is computed here from the sealed JSON in `data/` (and, for the earlier generic-soil trajectories, from the checked-in
`../observed_drydown/data` results); nothing is typed in except the declared constants of the OLD generic reference soil and the
Root-reported figures that are asserted against the data. No simulation is run, no network is used, no raw NEON file is opened.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import zipfile
from itertools import pairwise
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
OLD = HERE.parent / "observed_drydown" / "data"
DOCX = HERE / "SYRUP_JORN_MOAB_interface_comparison_20261117.docx"
FIG.mkdir(exist_ok=True)
TAB.mkdir(exist_ok=True)



def J(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


WINDOWS = ["JORN4_A", "JORN4_B", "MOAB2_C"]
WTITLE = {"JORN4_A": "JORN plot 4, window A (10 d, training)", "JORN4_B": "JORN plot 4, window B (4 d, temporal hold-out)", "MOAB2_C": "MOAB plot 2, window C (6 d, site hold-out)"}
WSHORT = {"JORN4_A": "JORN A", "JORN4_B": "JORN B", "MOAB2_C": "MOAB C"}
SITE = {"JORN4_A": "JORN", "JORN4_B": "JORN", "MOAB2_C": "MOAB"}
LV = ["501", "502", "503"]
D_GRID = [0.0, 0.001, 0.003, 0.01, 0.03, 0.05, 0.1, 0.3, 1.0]
E_GRID = [0.0, 0.1, 0.25, 0.5, 1.0, 1.2]
REF, GEN = "d0.05_E1.2_open", "d0_E0.5_open"
D0E0 = "d0_E0_open"
OLD_PARAMS = {"fc_sat": 0.26, "stress": 0.18, "ks_mm_h": 6.9, "k_if_mm_h": 2.182866}
FACTOR = (1 + 30 / 26) ** -1.5


def sc(d, e):
    return f"d{d:g}_E{e:g}_open"



def kind(w):
    return {"JORN4_A": "training_{}", "JORN4_B": "heldout_{}", "MOAB2_C": "heldout_{}"}[w]


def res_path(mode, w, m):
    k = "training" if w == "JORN4_A" else "heldout"
    return DATA / f"actual_{mode}" / f"{w}_{k}_{m}_result.json"


def cfg_path(mode, w, m):
    k = "training" if w == "JORN4_A" else "heldout"
    return DATA / f"actual_{mode}" / f"{w}_{k}_{m}.json"


def old_path(w):
    return OLD / f"{w}_{'training' if w == 'JORN4_A' else 'heldout'}_result.json"


# ------------------------------------------------------------------ load and re-derive
def layer_scores(theta, obs):
    mses, out = [], {}
    for i, v in enumerate(LV):
        e = np.array([theta[k][i] - o for k, o in enumerate(obs[v]) if o is not None])
        mses.append(float(np.mean(e**2)))
        out[v] = {"n": len(e), "bias": float(e.mean()), "rmse": math.sqrt(mses[-1])}
    out["overall"] = math.sqrt(sum(mses) / 3)
    return out


def integrate_profile(depths, theta, edges):
    """Layer means of the agreed piecewise-linear point profile with constant ends: the EXACT trapezoid over the layer edges and the sensor-depth knots."""
    knots = sorted({*edges, *(d for d in depths if edges[0] < d < edges[-1])})
    vals = np.interp(knots, depths, theta)
    out = []
    for a, b in pairwise(edges):
        pts = [(k, v) for k, v in zip(knots, vals, strict=True) if a <= k <= b]
        out.append(sum(0.5 * (v0 + v1) * (k1 - k0) for (k0, v0), (k1, v1) in pairwise(pts)) / (b - a))
    return out


def load():
    R = {}
    for w in WINDOWS:
        d = {"old": J(old_path(w))}
        for m in ("harmonic", "geometric"):
            d[m] = J(res_path("cpu", w, m))
            d[m + "_cfg"] = J(cfg_path("cpu", w, m))
            d[m + "_cuda"] = J(res_path("cuda", w, m))
        pi = d["harmonic"]["plan"]["prepared_inputs"]
        d["pi"] = pi
        d["obs"] = pi["observed_end_theta_m3m3"]
        d["days"] = d["harmonic"]["plan"]["model_days"]
        d["pers"] = {
            "point": layer_scores([pi["raw_initial_point_theta"]] * d["days"], d["obs"]),
            "layer": layer_scores([pi["integrated_initial_layer_theta"]] * d["days"], d["obs"]),
        }
        for m in ("harmonic", "geometric"):
            for k, run in d[m]["runs"].items():
                ls = layer_scores(run["layer_theta"], d["obs"])
                st = d[m]["scores"][k]
                assert math.isclose(ls["overall"], st["overall_rmse"], rel_tol=1e-12), (w, m, k)
                for v in LV:
                    assert ls[v]["n"] == st["per_layer"][v]["n"], (w, m, k, v)
                    assert math.isclose(ls[v]["rmse"], st["per_layer"][v]["rmse"], rel_tol=1e-12), (w, m, k, v)
                    assert math.isclose(ls[v]["bias"], st["per_layer"][v]["bias"], rel_tol=1e-9, abs_tol=1e-15), (w, m, k, v)
        for key, ref in (("point", "initial_point_values_held"), ("layer", "initial_layer_means_held")):
            sv = d["harmonic"]["persistence"][ref]
            assert math.isclose(d["pers"][key]["overall"], sv["overall_rmse"], rel_tol=1e-12), (w, key)
            for v in LV:
                assert d["pers"][key][v]["n"] == sv["per_layer"][v]["n"] and math.isclose(d["pers"][key][v]["rmse"], sv["per_layer"][v]["rmse"], rel_tol=1e-12)
                assert math.isclose(d["pers"][key][v]["bias"], sv["per_layer"][v]["bias"], rel_tol=1e-9, abs_tol=1e-15)
        # integrated initial layers re-derived from the sensor profile
        edges = d["harmonic_cfg"]["layers"]["edges_m"]
        rederived = integrate_profile(pi["sensor_depths_m"], pi["raw_initial_point_theta"], edges)
        assert all(math.isclose(a, b, rel_tol=1e-9) for a, b in zip(rederived, pi["integrated_initial_layer_theta"], strict=True)), (w, rederived)
        # old generic result agrees with property_effect pairing
        pe = J(DATA / "actual_cpu" / f"{w}_property_effect.json")
        d["pe"] = pe
        for k, v in pe["harmonic"]["matched"].items():
            assert math.isclose(d["old"]["scores"][k]["overall_rmse"], v["historical_overall_rmse"], rel_tol=1e-12), (w, k)
        d["cmp"] = J(DATA / "actual_cpu" / ("training_comparison.json" if w == "JORN4_A" else f"{w}_comparison.json"))
        R[w] = d
    return R


def derive(R):
    S = {}
    tr = R["JORN4_A"]
    cand = [(tr["harmonic"]["scores"][sc(d, e)]["overall_rmse"], e, d) for d in D_GRID for e in E_GRID]
    candg = [(tr["geometric"]["scores"][sc(d, e)]["overall_rmse"], e, d) for d in D_GRID for e in E_GRID]
    S["best_h"], S["best_g"] = min(cand), min(candg)
    assert S["best_h"][1:] == (0.0, 0.0) and S["best_g"][1:] == (0.0, 0.0)
    assert S["best_h"][0] == 0.0068218353393985276 == S["best_g"][0]
    assert S["best_h"][0] == tr["pers"]["layer"]["overall"]
    # whole-grid envelopes
    env = {}
    for w in WINDOWS:
        c = R[w]["cmp"]
        sc_cmp = c["scenario_comparison"] if w == "JORN4_A" else c["scenarios"]
        th = max(v["runs"]["series"]["layer_theta"]["max_abs"] for v in sc_cmp.values())
        wa = max(v["runs"]["series"]["water_m"]["max_abs"] for v in sc_cmp.values())
        rm = max(abs(v["scores"]["overall_rmse"]["diff"]) for v in sc_cmp.values())
        fs = max(abs(v["runs"]["final_storage_m"]["diff"]) for v in sc_cmp.values())
        ev = max(v["runs"]["series"]["evaporation_m"]["max_abs"] for v in sc_cmp.values())
        env[w] = {"theta": th, "water_m": wa, "rmse": rm, "final_storage_m": fs, "evap_m": ev, "n": len(sc_cmp)}
    S["env"] = env
    assert math.isclose(env["JORN4_A"]["theta"], 1.6029376112364213e-05, rel_tol=1e-9)
    assert math.isclose(env["JORN4_B"]["theta"], 1.4869378947367229e-05, rel_tol=1e-9)
    assert math.isclose(env["MOAB2_C"]["theta"], 1.838908954773788e-05, rel_tol=1e-9)
    assert math.isclose(env["JORN4_A"]["rmse"], 1.6459640131208375e-06, rel_tol=1e-9)
    assert math.isclose(env["JORN4_B"]["rmse"], 2.4582494413602496e-06, rel_tol=1e-9)
    assert math.isclose(env["MOAB2_C"]["rmse"], 2.237676102773145e-06, rel_tol=1e-9)
    assert math.isclose(env["MOAB2_C"]["final_storage_m"] * 1000, 0.002754, abs_tol=5e-7)
    # evaporation threshold: first day the top layer reaches residual in the E=0.5 d=0 run
    thr = {}
    for w in WINDOWS:
        th = R[w]["harmonic"]["runs"][GEN]["layer_theta"]
        top = [row[0] for row in th]
        first = next((k + 1 for k, x in enumerate(top) if x <= 0.01 + 1e-12), None)
        old_top = [row[0] for row in R[w]["old"]["runs"][GEN]["layer_theta"]]
        thr[w] = {"first_day_at_residual": first, "final_top": top[-1], "old_final_top": old_top[-1]}
    S["thr"] = thr
    assert thr["JORN4_A"]["first_day_at_residual"] == 3 and thr["JORN4_B"]["first_day_at_residual"] == 1
    assert thr["MOAB2_C"]["first_day_at_residual"] is None and math.isclose(thr["MOAB2_C"]["final_top"], 0.011873, abs_tol=5e-7)
    # no gravity / drainage / return anywhere, all states below FC
    for w in WINDOWS:
        for m in ("harmonic", "geometric"):
            fc = R[w][m + "_cfg"]["layers"]["theta_field_capacity"]
            for run in R[w][m]["runs"].values():
                tj = run["trajectory"]
                assert max(abs(x) for x in tj["drainage_m"]) == 0 and max(abs(x) for x in tj["surface_return_m"]) == 0
                assert max(abs(x) for r in tj["percolation_m"] for x in r) == 0
                assert all(r[i] < fc[i] for r in run["layer_theta"] for i in range(3))
    summ = J(DATA / "root_independent_summary.json")
    S["summ"] = summ
    assert summ["max_backend_trajectory_field_abs"] == 3.469446951953614e-18
    assert summ["maximum_decimal_water_error_m"] == 2.1542145003691e-18
    assert math.isclose(summ["max_native_cumulative_residual_over_bound"], 0.002326539193660974, rel_tol=1e-12)
    return S


# ------------------------------------------------------------------ number formatting
def e3(x):
    return f"{x * 1000:.3f}"


def sci(x, n=3):
    return f"{x:.{n}e}"


def mm(x):
    if x == 0:
        return "0 (exact)"
    v = x * 1000
    return f"{v:.3e}" if abs(v) < 1e-4 else f"{v:.4g}"


# ------------------------------------------------------------------ figures
C_OLD, C_H, C_G, C_SEL = "#7a7a7a", "#c0392b", "#1f5fbf", "#2e8b57"


def fig_profiles(R):
    fig, axs = plt.subplots(3, 3, figsize=(7.3, 8.2), sharex="col")
    for j, w in enumerate(WINDOWS):
        r = R[w]
        n = r["days"]
        t = np.arange(0, n + 1)
        pi = r["pi"]
        for i, lv in enumerate(LV):
            ax = axs[i, j]
            oldr = np.array([pi["integrated_initial_layer_theta"][i]] + [row[i] for row in r["old"]["runs"][REF]["layer_theta"]])
            hr = np.array([pi["integrated_initial_layer_theta"][i]] + [row[i] for row in r["harmonic"]["runs"][REF]["layer_theta"]])
            gr = np.array([pi["integrated_initial_layer_theta"][i]] + [row[i] for row in r["geometric"]["runs"][REF]["layer_theta"]])
            ax.plot(t, oldr, "-", color=C_OLD, lw=1.4, label="old generic soil, reference (d 0.05, E 1.2)")
            ax.plot(t, hr, "-", color=C_H, lw=1.8, label="new site hydraulics, harmonic, reference (0.05, 1.2)")
            ax.plot(t, gr, "--", color=C_G, lw=1.2, label="new site hydraulics, geometric, reference (0.05, 1.2) [overlaps harmonic]")
            ax.plot([0, n], [pi["integrated_initial_layer_theta"][i]] * 2, "-", color=C_SEL, lw=1.6, label="new selected (d 0, E 0) = integrated initial layer held")
            ax.plot([0], [pi["integrated_initial_layer_theta"][i]], "s", color=C_SEL, ms=6, label="integrated initial layer mean")
            ax.plot([0], [pi["raw_initial_point_theta"][i]], "D", color="black", mfc="none", ms=6, label="observed initial sensor value (point)")
            ob = [(k + 1, o) for k, o in enumerate(r["obs"][lv]) if o is not None]
            ax.plot([a for a, _ in ob], [b for _, b in ob], "o", color="black", ms=3.5, ls="none", label="observed end-of-day value (missing days not drawn)")
            if i == 0:
                ax.set_title(f"{WSHORT[w]} ({n} d)", fontsize=9)
            if j == 0:
                ax.set_ylabel(f"layer {i + 1} ({['0-10', '10-20', '20-30'][i]} cm)\nθ (m³ m⁻³)", fontsize=8)
            if i == 2:
                ax.set_xlabel("model day (end of day k)", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.25)
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=7, frameon=False)
    fig.suptitle("Layer moisture: observations, old generic reference soil, new site-derived hydraulics", fontsize=10)
    fig.tight_layout(rect=(0, 0.1, 1, 0.97))
    fig.savefig(FIG / "fig1_layer_profiles.png", dpi=170)
    plt.close(fig)


def fig_delta(R, S):
    fig, axs = plt.subplots(1, 3, figsize=(7.3, 3.2), sharey=False)
    for j, w in enumerate(WINDOWS):
        r = R[w]
        n = r["days"]
        for i in range(3):
            h = np.array([row[i] for row in r["harmonic"]["runs"][REF]["layer_theta"]])
            g = np.array([row[i] for row in r["geometric"]["runs"][REF]["layer_theta"]])
            axs[j].plot(np.arange(1, n + 1), (g - h) * 1e5, "o-", ms=3, label=f"layer {i + 1}")
        axs[j].axhline(0, color="k", lw=0.6)
        axs[j].set_title(WSHORT[w], fontsize=9)
        axs[j].set_xlabel("model day", fontsize=8)
        axs[j].tick_params(labelsize=7)
        axs[j].grid(alpha=0.25)
    axs[0].set_ylabel("geometric − harmonic θ  (10⁻⁵ m³ m⁻³)", fontsize=8)
    axs[0].legend(fontsize=7)
    fig.suptitle("Magnified interface-mixing difference at the fixed nonzero reference (d 0.05, E 1.2); d 0 differences are exactly zero", fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(FIG / "fig2_geometric_minus_harmonic.png", dpi=170)
    plt.close(fig)


def grid(R, key):
    return np.array([[key(d, e) for e in E_GRID] for d in D_GRID])


def heat(ax, z, title, label, fmt, cmap):
    im = ax.imshow(z, cmap=cmap, aspect="auto", origin="lower")
    norm = im.norm
    for i in range(len(D_GRID)):
        for j in range(len(E_GRID)):
            ax.text(j, i, fmt(z[i, j]), ha="center", va="center", fontsize=5.5, color="white" if norm(z[i, j]) > 0.6 else "black")
    ax.set_xticks(range(len(E_GRID)), [f"{e:g}" for e in E_GRID], fontsize=7)
    ax.set_yticks(range(len(D_GRID)), [f"{d:g}" for d in D_GRID], fontsize=7)
    ax.set_xlabel("evaporation multiplier E", fontsize=8)
    ax.set_title(title, fontsize=8)
    cb = plt.colorbar(im, ax=ax)
    cb.set_label(label, fontsize=7)
    cb.ax.tick_params(labelsize=6)


def fig_training(R):
    tr = R["JORN4_A"]
    zh = grid(R, lambda d, e: tr["harmonic"]["scores"][sc(d, e)]["overall_rmse"] * 1e3)
    zd = grid(R, lambda d, e: (tr["geometric"]["scores"][sc(d, e)]["overall_rmse"] - tr["harmonic"]["scores"][sc(d, e)]["overall_rmse"]) * 1e6)
    fig, axs = plt.subplots(1, 2, figsize=(7.3, 4.4))
    heat(axs[0], zh, "Harmonic: primary equal-layer RMSE\n(training, JORN window A)", "10⁻³ m³ m⁻³", lambda v: f"{v:.1f}", "viridis_r")
    axs[0].set_ylabel("exchange coefficient d", fontsize=8)
    heat(axs[1], zd, "Paired score difference, geometric − harmonic", "10⁻⁶ m³ m⁻³", lambda v: f"{v:.2f}", "coolwarm")
    fig.tight_layout()
    fig.savefig(FIG / "fig3_training_grid.png", dpi=170)
    plt.close(fig)


def fig_property(R):
    tr = R["JORN4_A"]
    zn = grid(R, lambda d, e: (tr["harmonic"]["scores"][sc(d, e)]["overall_rmse"] - tr["old"]["scores"][sc(d, e)]["overall_rmse"]) * 1e3)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    heat(ax, zn, "New (harmonic) minus old generic RMSE at identical (d, E) pairs\n(JORN window A; positive = new worse)", "10⁻³ m³ m⁻³", lambda v: f"{v:.1f}", "magma_r")
    ax.set_ylabel("exchange coefficient d", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_new_minus_old.png", dpi=170)
    plt.close(fig)


def fig_evap(R):
    fig, axs = plt.subplots(2, 3, figsize=(7.3, 5.0))
    for j, w in enumerate(WINDOWS):
        r = R[w]
        n = r["days"]
        t = np.arange(0, n + 1)
        for k, (nm, ls) in enumerate(((REF, "-"), (GEN, ":"))):
            for lab, run, col in (("old generic", r["old"]["runs"][nm], C_OLD), ("new harmonic", r["harmonic"]["runs"][nm], C_H), ("new geometric", r["geometric"]["runs"][nm], C_G)):
                ev = np.concatenate([[0], np.cumsum(run["trajectory"]["evaporation_m"])]) * 1000
                st = np.concatenate([[r["pi"]["integrated_initial_layer_water_m"] and sum(r["pi"]["integrated_initial_layer_water_m"])], [sum(x) for x in run["trajectory"]["water_m"]]]) * 1000
                dd = "--" if lab == "new geometric" else "-"
                axs[0, j].plot(t, ev, ls=ls if ls == ":" else dd, color=col, lw=1.3, label=f"{lab}, {'(0.05, 1.2)' if nm == REF else '(0, 0.5)'}")
                axs[1, j].plot(t, st, ls=ls if ls == ":" else dd, color=col, lw=1.3)
        axs[0, j].set_title(WSHORT[w], fontsize=9)
        axs[1, j].set_xlabel("model day", fontsize=8)
        for a in axs[:, j]:
            a.tick_params(labelsize=7)
            a.grid(alpha=0.25)
    axs[0, 0].set_ylabel("cumulative soil evaporation (mm)", fontsize=8)
    axs[1, 0].set_ylabel("total stored water, 0-30 cm (mm)", fontsize=8)
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=6.5, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(FIG / "fig5_evaporation_storage.png", dpi=170)
    plt.close(fig)


def fig_initial(R):
    fig, axs = plt.subplots(1, 3, figsize=(7.3, 3.2), sharey=True)
    for j, w in enumerate(WINDOWS):
        r = R[w]
        pi = r["pi"]
        edges = r["harmonic_cfg"]["layers"]["edges_m"]
        dep = np.array(pi["sensor_depths_m"])
        th = np.array(pi["raw_initial_point_theta"])
        x = np.linspace(0, 0.3, 301)
        axs[j].plot(np.interp(x, dep, th), x * 100, "-", color="black", lw=1, label="piecewise-linear profile, constant ends")
        axs[j].plot(th, dep * 100, "D", mfc="none", color="black", ms=6, label="sensor values (point)")
        for i in range(3):
            axs[j].plot([pi["integrated_initial_layer_theta"][i]] * 2, [edges[i] * 100, edges[i + 1] * 100], "-", color=C_SEL, lw=3, label="layer mean used to start the model" if i == 0 else None)
        axs[j].set_title(WSHORT[w], fontsize=9)
        axs[j].set_xlabel("θ (m³ m⁻³)", fontsize=8)
        axs[j].tick_params(labelsize=7)
        axs[j].grid(alpha=0.25)
    axs[0].invert_yaxis()
    axs[0].set_ylabel("depth (cm)", fontsize=8)
    for a in axs:
        a.xaxis.set_major_locator(plt.MaxNLocator(4))
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(FIG / "fig6_initial_profile.png", dpi=170)
    plt.close(fig)


# ------------------------------------------------------------------ tables
def build_tables(R, S):
    T = {}
    # window table
    rows = []
    for w in WINDOWS:
        r = R[w]
        p = r["harmonic"]["plan"]
        rows.append([WSHORT[w], {"JORN4_A": "training", "JORN4_B": "temporal hold-out", "MOAB2_C": "site hold-out"}[w], str(r["days"]), p["anchor_utc"].replace("T", " ").replace("Z", ""), str(p["valid_joint_targets"]), "/".join(str(p["valid_targets_per_level"][v]) for v in LV), f"{p['latitude_deg'] if 'latitude_deg' in p else p['pet']['latitude_deg']:.3f}"])
    T["windows"] = (["Window", "Role", "Model days", "Anchor (UTC)", "Joint targets", "Valid targets per layer (1/2/3)", "Latitude (°N)"], rows)
    # property table
    rows = []
    for site, w in (("JORN", "JORN4_A"), ("MOAB", "MOAB2_C")):
        L = R[w]["harmonic_cfg"]["provenance"]["layers"]
        lc = R[w]["harmonic_cfg"]["layers"]
        for i, l in enumerate(L):
            rows.append([site, f"{i + 1} ({l['record']})", f"{l['raw_density_g_cm3']:.4f}", f"{l['effective_density_g_cm3']:.4f}", f"{l['raw_air_entry_kpa']:.4f}", f"{l['effective_air_entry_kpa']:.4f}", f"{lc['theta_residual'][i]:.3f}", f"{l['theta1500']:.4f}", f"{l['theta_field_capacity']:.4f}", f"{l['theta_saturation']:.4f}", f"{l['Ks_mm_h']:.2f}"])
    T["props"] = (["Site", "Layer (record)", "Raw ρ (g cm⁻³)", "Effective ρ", "Raw air entry (kPa)", "Effective air entry", "θ residual", "θ1500", "θ FC6", "θ sat.", "Ks (mm h⁻¹)"], rows)
    # interface table
    rows = []
    for site, w in (("JORN", "JORN4_A"), ("MOAB", "MOAB2_C")):
        h = R[w]["harmonic_cfg"]["provenance"]["interface"]
        g = R[w]["geometric_cfg"]["provenance"]["interface"]
        for i in range(2):
            kh, kg = h["interface_conductivity_m_s"][i] * 3.6e6, g["interface_conductivity_m_s"][i] * 3.6e6
            rows.append([site, f"{i + 1}↔{i + 2}", f"{h['local_conductivity_m_s'][i] * 3.6e6:.4f}", f"{h['local_conductivity_m_s'][i + 1] * 3.6e6:.4f}", f"{kh:.4f}", f"{kg:.4f}", f"{(kg / kh - 1) * 100:.3f}"])
    T["iface"] = (["Site", "Interface", "K upper = f·Ks (mm h⁻¹)", "K lower (mm h⁻¹)", "Harmonic K_if (mm h⁻¹)", "Geometric K_if (mm h⁻¹)", "G vs H (%)"], rows)
    # old vs new parameters
    rows = [["Old generic reference soil", "0.26 (FC = saturation)", "0.18", "6.9", "2.1829 (all pairs)"]]
    for site, w in (("JORN", "JORN4_A"), ("MOAB", "MOAB2_C")):
        L = R[w]["harmonic_cfg"]["provenance"]["layers"]
        ks = [l["Ks_mm_h"] for l in L]
        rows.append([f"New {site} (three layers)", " / ".join(f"{l['theta_field_capacity']:.3f}" for l in L) + " (FC6; saturation " + " / ".join(f"{l['theta_saturation']:.3f}" for l in L) + ")", f"{L[0]['theta1500']:.4f} (top θ1500)", " / ".join(f"{k:.1f}" for k in ks) + " (" + f"{min(ks) / 6.9:.1f}–{max(ks) / 6.9:.1f}× old)", " / ".join(f"{x * 3.6e6:.1f}" for x in R[w]["harmonic_cfg"]["layers"]["interface_conductivity_m_s"]) + " (harmonic)"])
    T["oldnew"] = (["Soil", "Field capacity θ", "Evaporation stress scale θ", "Ks (mm h⁻¹)", "Interface K (mm h⁻¹)"], rows)
    # training summary
    tr = R["JORN4_A"]
    rows = []
    for m, b in (("harmonic", S["best_h"]), ("geometric", S["best_g"])):
        rows.append([m.title(), f"d {b[2]:g}, E {b[1]:g}", f"{e3(b[0])}", f"{e3(tr['pers']['point']['overall'])}", f"{e3(tr['pers']['layer']['overall'])}", f"{e3(tr[m]['scores'][REF]['overall_rmse'])}", f"{e3(tr['old']['scores'][REF]['overall_rmse'])}", f"{e3(tr['old']['scores'][GEN]['overall_rmse'])}"])
    T["train"] = (["Method", "Training-selected pair (9×6 grid)", "Selected RMSE", "Point persistence", "Layer persistence", "Reference (0.05, 1.2) RMSE", "Old generic RMSE at reference", "Old generic RMSE at its best (0, 0.5)"], rows)
    # heldout table
    rows = []
    for w in WINDOWS[1:] + ["JORN4_A"]:
        r = R[w]
        for nm, lab in ((D0E0, "(0, 0) selected"), (REF, "(0.05, 1.2) reference"), (GEN, "(0, 0.5) old generic best")):
            if nm not in r["harmonic"]["scores"]:
                continue
            h, g = r["harmonic"]["scores"][nm]["overall_rmse"], r["geometric"]["scores"][nm]["overall_rmse"]
            o = r["old"]["scores"][nm]["overall_rmse"]
            rows.append([WSHORT[w], lab, e3(h), e3(g), f"{(g - h) * 1e6:+.3f}", e3(o), e3(r["pers"]["point"]["overall"]), e3(r["pers"]["layer"]["overall"])])
    T["held"] = (["Window", "Pair (d, E)", "H RMSE", "G RMSE", "G − H (10⁻⁶)", "Old generic RMSE", "Point pers.", "Layer pers."], rows)
    # per-layer table
    rows = []
    for w in WINDOWS:
        r = R[w]
        p, l = r["pers"]["point"], r["pers"]["layer"]
        rows.append([WSHORT[w], "point persistence"] + [x for v in LV for x in (str(p[v]["n"]), f"{p[v]['bias'] * 1e3:+.2f}", f"{p[v]['rmse'] * 1e3:.2f}")])
        rows.append([WSHORT[w], "layer persistence (= d 0, E 0)"] + [x for v in LV for x in (str(l[v]["n"]), f"{l[v]['bias'] * 1e3:+.2f}", f"{l[v]['rmse'] * 1e3:.2f}")])
        for m in ("harmonic", "geometric"):
            s = r[m]["scores"][REF]["per_layer"]
            rows.append([WSHORT[w], f"{m} reference (0.05, 1.2)"] + [x for v in LV for x in (str(s[v]["n"]), f"{s[v]['bias'] * 1e3:+.2f}", f"{s[v]['rmse'] * 1e3:.2f}")])
        s = r["old"]["scores"][REF]["per_layer"]
        rows.append([WSHORT[w], "old generic reference (0.05, 1.2)"] + [x for v in LV for x in (str(s[v]["n"]), f"{s[v]['bias'] * 1e3:+.2f}", f"{s[v]['rmse'] * 1e3:.2f}")])
    T["layers"] = (["Window", "Scenario"] + [f"L{i} {q}" for i in (1, 2, 3) for q in ("N", "bias", "RMSE")], rows)
    # reference differences
    rows = []
    for w in WINDOWS:
        c = R[w]["cmp"]
        v = (c["scenario_comparison"] if w == "JORN4_A" else c["scenarios"])[REF]
        s = v["runs"]["series"]
        rows.append([WSHORT[w], f"{v['scores']['overall_rmse']['diff'] * 1e6:+.3f}", f"{s['layer_theta']['max_abs'] * 1e5:.3f}", f"{s['layer_theta']['max_rel'] * 100:.4f}", mm(s["water_m"]["max_abs"]), mm(s["evaporation_m"]["max_abs"]), mm(abs(v["runs"]["final_storage_m"]["diff"])), mm(v["runs"]["final_storage_m"]["harmonic"])])
    T["refdiff"] = (["Window", "RMSE G − H (10⁻⁶)", "Max |Δθ| (10⁻⁵)", "Max relative Δθ (%)", "Max |Δ layer water| (mm)", "Max |Δ daily evaporation| (mm)", "|Δ final storage| (mm)", "H final storage (mm)"], rows)
    # envelope
    rows = []
    for w in WINDOWS:
        e = S["env"][w]
        rows.append([WSHORT[w], str(e["n"]), f"{e['theta'] * 1e5:.3f}", f"{e['rmse'] * 1e6:.3f}", mm(e["water_m"]), mm(e["evap_m"]), mm(e["final_storage_m"])])
    T["env"] = (["Window", "Scenarios paired", "Max |Δθ| (10⁻⁵ m³ m⁻³)", "Max |Δ primary RMSE| (10⁻⁶)", "Max |Δ layer water| (mm)", "Max |Δ daily evaporation| (mm)", "Max |Δ final storage| (mm)"], rows)
    # evaporation threshold
    rows = []
    for w in WINDOWS:
        t = S["thr"][w]
        pi = R[w]["pi"]
        rows.append([WSHORT[w], f"{pi['integrated_initial_layer_theta'][0]:.4f}", str(t["first_day_at_residual"]) if t["first_day_at_residual"] else "not reached", f"{t['final_top']:.5f}", f"{t['old_final_top']:.5f}"])
    T["thr"] = (["Window", "Initial top-layer θ (layer mean)", "First day top layer at residual 0.01 (new, d 0, E 0.5)", "Final top-layer θ new", "Final top-layer θ old generic"], rows)
    # backend
    cu = J(DATA / "actual_cuda" / "completion.json")
    cp = J(DATA / "actual_cpu" / "completion.json")
    rows = []
    for a, b in zip(cp["rows"], cu["rows"]):
        rows.append([a["config"].replace(".json", ""), str(a["scenarios"]), f"{a['seconds_including_startup']:.2f}", f"{b['seconds_including_startup']:.2f}", sci(b["max_abs_trajectory_difference_from_actual_cpu"], 3)])
    T["backend"] = (["Configuration", "Scenarios", "CPU loop incl. startup (s)", "CUDA loop incl. startup (s)", "Max |CUDA − CPU| trajectory field"], rows)
    return T


def write_tables(T):
    for k, (hd, rows) in T.items():
        with open(TAB / f"{k}.csv", "w", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(hd)
            w.writerows(rows)
    (TAB / "report_tables.json").write_text(json.dumps({k: {"header": h, "rows": r} for k, (h, r) in T.items()}, indent=1))


# ------------------------------------------------------------------ text blocks
REFS = [
    ("Tietjen 2009", "Tietjen B, Zehe E, Jeltsch F (2009). Simulating plant water availability in dry lands under climate change: a generic model of two soil layers. Water Resources Research 45, W01418.", "https://doi.org/10.1029/2007WR006589"),
    ("Tietjen 2010", "Tietjen B, Jeltsch F, Zehe E, Classen N, Groengroeft A, Schiffers K, Oldeland J (2010). Effects of climate change on the coupled dynamics of water and vegetation in drylands. Ecohydrology 3, 226–237.", "https://doi.org/10.1002/eco.70"),
    ("Saxton & Rawls 2006", "Saxton KE, Rawls WJ (2006). Soil water characteristic estimates by texture and organic matter for hydrologic solutions. Soil Science Society of America Journal 70, 1569–1578.", "https://doi.org/10.2136/sssaj2005.0117"),
    ("Hargreaves & Samani 1985", "Hargreaves GH, Samani ZA (1985). Reference crop evapotranspiration from temperature. Applied Engineering in Agriculture 1(2), 96–99.", "https://doi.org/10.13031/2013.26773"),
    ("Baldauf 2023", "Baldauf S, Cantón Y, Tietjen B (2023). Biocrusts intensify water redistribution and improve water availability to dryland vegetation: insights from a spatially-explicit ecohydrological model. Frontiers in Microbiology 14, 1179291 (and the downloaded EcoHyD code release, commit 80b10e6b6b82ad16e9fd41a31901cc3157d880e9, used by SYRUP's reference parameters).", "https://doi.org/10.3389/fmicb.2023.1179291"),
    ("Hoffmann et al. 2002", "Hoffmann JP, Ripich MA, Ellett KM (2002). Characteristics of shallow deposits beneath Rillito Creek, Pima County, Arizona. U.S. Geological Survey Water-Resources Investigations Report 01-4257 (harmonic series rule for flow normal to layers, Eq. 5, p. 19).", "https://doi.org/10.3133/wri014257"),
    ("NEON data", "National Ecological Observatory Network. RELEASE-2026: soil water content and water salinity DP1.00094.001; single aspirated air temperature DP1.00002.001; precipitation DP1.00045.001; megapit soil characterization DP1.00096.001.", "https://data.neonscience.org/data-products/DP1.00094.001/RELEASE-2026"),
    ("NEON notice 2026", "NEON (2026-11-09). Soil water content and water salinity (DP1.00094.001) algorithm updates: planned for RELEASE-2027 (depth workaround, calibration and QA revisions). The frozen RELEASE-2026 files used here are retained; no silent migration.", "https://www.neonscience.org/impact/observatory-blog/soil-water-content-and-water-salinity-dp100094001-algorithm-updates"),
]


def blocks(R, S, T):
    B = []
    A = B.append
    tr = R["JORN4_A"]
    bh = S["best_h"]
    env = S["env"]
    pp, pl = tr["pers"]["point"]["overall"], tr["pers"]["layer"]["overall"]
    sm = S["summ"]
    pf = J(DATA / "root_additional_metrics.json")["physical_CPU_CUDA_max_abs_by_field"]
    evr = ", ".join(f"{R[w]['harmonic']['runs'][REF]['cumulative']['evaporation_m3'] * 1000:.1f}" for w in WINDOWS) + " mm (JORN A, JORN B, MOAB C)"
    ks = {k: [l["Ks_mm_h"] for l in R[w]["harmonic_cfg"]["provenance"]["layers"]] for k, w in (("JORN", "JORN4_A"), ("MOAB", "MOAB2_C"))}
    ksr = f"Ks ≈ {min(ks['JORN']):.0f}–{max(ks['JORN']):.0f} mm h⁻¹ at JORN and {min(ks['MOAB']):.0f}–{max(ks['MOAB']):.0f} mm h⁻¹ at MOAB"
    mult = {}
    for w in WINDOWS:
        th0 = R[w]["pi"]["integrated_initial_layer_theta"][0]
        mult[w] = (min(1.0, (th0 / OLD_PARAMS["stress"]) ** 2), min(1.0, (th0 / R[w]["harmonic_cfg"]["layers"]["theta_evaporation_stress"]) ** 2))
    for w, (o, n) in zip(WINDOWS, ((0.069911, 1.0), (0.034836, 0.723098), (0.131537, 1.0)), strict=True):
        assert math.isclose(mult[w][0], o, abs_tol=1e-6) and math.isclose(mult[w][1], n, abs_tol=1e-6), (w, mult[w])
    A(("title", "Harmonic versus geometric interface conductivity in the SYRUP dry-down model: JORN and MOAB site-derived hydraulics"))
    A(("sub", "Comparison report for three frozen NEON dry-down windows · generated 2026-11-17 from sealed actual runs · diagnostic, not field validation"))
    A(("h1", "1. Summary"))
    A(("p", "SYRUP's layered soil-water model needs one conductivity at each boundary between soil layers. Two averages of the neighbouring layers' conductivities are defensible: the **harmonic** mean, which mirrors resistances in series, and the **geometric** mean. This report runs both through the same native SYRUP dry-down driver on three short NEON soil-moisture windows (JORN, the Jornada Experimental Range, plot 4 twice; MOAB, the Moab site, plot 2 once), using layer properties derived for those two sites, and compares them with each other and with the earlier generic-soil results at identical settings."))
    A(("bul", [
        f"**Harmonic and geometric are practically indistinguishable here.** Over the 56 paired JORN A training scenarios the largest water-content difference is {env['JORN4_A']['theta'] * 1e5:.2f}×10⁻⁵ m³ m⁻³; in the 3-scenario held-out sets it is {env['JORN4_B']['theta'] * 1e5:.2f}×10⁻⁵ (JORN B) and {env['MOAB2_C']['theta'] * 1e5:.2f}×10⁻⁵ (MOAB C). Primary-score differences never exceed {max(e['rmse'] for e in env.values()) * 1e6:.2f}×10⁻⁶ m³ m⁻³. Adjacent layers' saturated conductivities are close ({ksr}), so the two means differ by at most 0.24 % at any interface.",
        f"**Training selects the same pair for both methods: d 0, E 0** (primary RMSE {e3(bh[0])}×10⁻³ m³ m⁻³ on the coarse 9×6 grid). That optimum equals persistence of the integrated initial layer means. For JORN A the raw initial sensor values (point persistence, {e3(pp)}×10⁻³ m³ m⁻³) score better; for MOAB C layer persistence is the better of the two. At d = 0 the interface cannot act, so the held-out matches for the common pair are identical for H and G *by construction*; this does not prove the two laws equivalent. The nonzero reference and the paired training grid therefore carry the comparison.",
        "**The properties and the lower evaporation-stress scale matter far more than the averaging.** At the fixed reference (d 0.05, E 1.2) and at the earlier generic best (d 0, E 0.5), all three new fits score worse than the earlier generic soil at the identical pair (for example MOAB C at the reference: 18.7×10⁻³ against 1.5×10⁻³ m³ m⁻³ before). The new stress scales (top-layer θ1500: 0.0395 JORN, 0.0590 MOAB, against 0.18) let the top layer lose water much faster. The d = 0 controls isolate the evaporation-limit effect from exchange: no exchange, gravity or saturation limit acts there, although the unchanged residual-availability cap still limits E.",
        "**Gravity drainage, surface return and bottom drainage are exactly zero in every run** because every daily state stays below field capacity. The field-capacity and saturation limits therefore never act, open and closed bottoms are identical, and nothing here tests bottom drainage. The interface conductivity does act on the d > 0 exchange.",
        "**The new values are derived surrogates, not measured hydraulic properties, and should not be treated as validated site hydraulics.** Texture is NEON field data, organic carbon is estimated, and hydraulics come from a pedotransfer regression with project-approved bounding floors; only three short windows were used.",
    ]))
    A(("h1", "2. What was run and why"))
    A(("p", "The experiment uses the existing, independently qualified SYRUP dry-down driver unchanged (production source package 37d89f23…, 180 files). It evolves three 10 cm soil layers (0–30 cm) with potential evaporation from a Hargreaves temperature model [Hargreaves & Samani 1985] at each site's latitude, transpiration off, gravity on, no rain, no storm, no vegetation, wind or bed. **Clock.** Model days are not NEON calendar days: day k ends at the window's anchor time plus k × 86 400 s, at the anchor's own clock time (JORN A 22:00 UTC, JORN B 06:30 UTC, MOAB C 03:30 UTC). Each day uses the 48 consecutive passing half-hour weather records of its anchor-aligned 24 h window (for example JORN A day 1 uses 2025-04-08T22:00Z to 2025-04-09T22:00Z), and its PET date is the calendar date of the window midpoint at fixed UTC−7 (April 9 for that example). The observed daily target is the passing 30-minute mean ending exactly at the model day end, not an instantaneous value and not a daily mean. The water model starts from the sensor profile at the anchor. The soil-water and exchange formulation follows Tietjen's two-layer approach [Tietjen 2009; Tietjen 2010] as implemented in the EcoHyD code used for SYRUP's reference parameters [Baldauf 2023]."))
    A(("tab", "windows", "Table 1. Frozen windows (RELEASE-2026 data). Days are anchor-aligned 24 h windows (anchor + k × 86 400 s) with 48 consecutive passing half-hour weather records each; the anchor column gives the clock time of every day end. Missing sensor values stay missing."))
    A(("p", "**Layer interface conductivity.** Each layer i has a fixed redistribution conductivity K_i = f·Ks_i, where Ks_i is the saturated conductivity and f = (1 + 30/26)^−1.5 = 0.31635738822500753 is the EcoHyD factor retained from the current SYRUP implementation. K_i is not a moisture-dependent conductivity and not a wetting-front suction. For a pair of adjacent layers, half-layer distances to the shared edge give weights w_i = h_i/(h_i + h_j) and w_j = h_j/(h_i + h_j), with h the layer thickness taken from the actual layer edges (all 0.1 m here, so both weights are 0.5)."))
    A(("eq", "Harmonic:   K_if = 1 / Σ(w·/K·) = 1 / (w_i/K_i + w_j/K_j)"))
    A(("eq", "Geometric:  K_if = exp(w_i ln K_i + w_j ln K_j)"))
    A(("p", "For equal layer thicknesses these reduce to 2K_iK_j/(K_i + K_j) and √(K_iK_j). When the two conductivities are identical both return K exactly, which preserves the earlier homogeneous-reference identity. Zero, negative, non-finite or malformed inputs are refused before anything runs. The harmonic form is the series-resistance (layers in series) rule for flow normal to layers [Hoffmann et al. 2002, Eq. 5, p. 19] and is the better-motivated physical choice for vertical layering. That is a motivation for the *averaging only*: it does not validate SYRUP's moisture-gradient exchange law or Richards-equation flow, and the windows used here cannot select either mean empirically because adjacent Ks values are close."))
    A(("p", "**What the interface conductivity does.** The exchange between neighbouring layers is a pair relaxation in which the donor layer depth enters as d·K_if·h_donor, with d the dimensionless exchange coefficient; the donor and capacity limits of the existing exact-pair relaxation are unchanged. The gravity/field-capacity cascade does not use K_if. At d = 0 the exchange vanishes, so every d = 0 scenario must be, and is, identical for H and G (checked for all 7 d = 0 training controls and for every held-out d = 0 match)."))
    A(("h1", "3. Site-derived layer properties"))
    A(("p", "Both sites use three 10 cm layers with edges 0, 0.1, 0.2, 0.3 m. Soil texture and bulk density come from the NEON megapit measurements [NEON data], used as site proxies for the sensor plots; the estimated hydraulic values are not co-located hydraulic measurements. organic carbon is estimated, and organic matter is taken as 1.724 × estimated OC (g kg⁻¹)/10 percent, a provisional conversion. Saturated conductivity, retention and field capacity at 6 kPa (FC6) are Saxton–Rawls 2006 pedotransfer regression outputs [Saxton & Rawls 2006] computed from layer-averaged (dry-mass-mixed) texture and organic matter **before** the floors are applied. The floors are project-approved bounded assumptions for this sensitivity study, not Saxton–Rawls recommendations. The air entry is floored at zero, max(predicted, 0), and for JORN the effective bulk density is max(raw, 1 g cm⁻³, 0.901 × the regression's normal density). The air entry entered FC6 only; no derived air entry is passed to the native provider as a measured override and the provider's default recipe is not adopted. Residual water content is 0.01 for all layers. MOAB uses a 6 cm A plus 4 cm Bw dry-mass mixture for layer 1 and Bw for layers 2 and 3, with no density change."))
    A(("tab", "props", "Table 2. Raw and effective inputs and the derived hydraulic table (all derived surrogates; raw measurements remain in the canonical receipt). θ1500 is the predicted water content at 1500 kPa; FC6 and saturation are volumetric. The air entry is 0 where the raw regression value is negative."))
    A(("p", "The raw JORN Bw density (0.9035 g cm⁻³) is low but is not shown to be wrong; the density floor and the zero air-entry floor are bounding assumptions that let the surrogate be evaluated, not corrections of the measurements. SYRUP's native property provider stays strict and unchanged, and these values are prescribed inputs, not a provider recipe."))
    A(("tab", "iface", "Table 3. Interface conductivities for the two means. Geometric differs from harmonic by 0.001–0.24 % and exactly 0 for MOAB's lower pair (identical layers)."))
    A(("h2", "How these differ from the earlier generic soil"))
    A(("p", "The earlier generic reference soil took field capacity equal to saturation (0.26), a wilting point and evaporation-stress scale of 0.18, Ks of 6.9 mm h⁻¹, and an interface conductivity of 2.182866 mm h⁻¹ (= f × 6.9). The new soils have conductivities roughly 8–18 times larger and, importantly, a *much lower* evaporation-stress scale, taken here as the top-layer predicted θ1500 (0.0395 JORN, 0.0590 MOAB; a declared assumption) and held fixed within a run. Lower-layer wilting values are informative but do not enter surface evaporation."))
    A(("tab", "oldnew", "Table 4. Old generic reference and new site-derived parameters."))
    A(("h1", "4. Experiment design"))
    A(("p", "Training (JORN A, both methods) runs the same 9 × 6 grid of exchange coefficient d ∈ {0, 0.001, 0.003, 0.01, 0.03, 0.05, 0.1, 0.3, 1} and evaporation multiplier E ∈ {0, 0.1, 0.25, 0.5, 1, 1.2} with an open bottom, plus two closed-bottom controls (d 0, E 0 and d 0.05, E 1.2): 56 scenarios per method. The method selecting the best pair uses only training: the lowest primary equal-layer RMSE, √(mean of per-layer mean-square errors), with ties to smaller E, then smaller d. Held-out windows (JORN B at the same site, MOAB C at a different site) were then run for the de-duplicated union of the selected pairs, the unfitted reference (0.05, 1.2), the earlier generic best (0, 0.5) and the d = 0 / E = 0 controls: three distinct scenarios, run under both methods. No held-out refit and no grid refinement, extra windows or property re-tuning followed the outcomes. The d = 0, E = 0 optimum is the best of a discrete 9 × 6 grid only; it is not evidence that zero evaporation is physical or a global optimum."))
    A(("p", "**Initial profile.** Each window starts from the agreed piecewise-linear integration of the sensor profile with constant values beyond the end sensors, averaged over each layer. A single sensor at 5.8 cm can sit above or below the 0–10 cm layer mean when the profile has a gradient, so the starting layer mean and the first sensor reading differ; no new operator was introduced and the observations are scored against the layer states as before. Persistence controls hold either the integrated layer means (layer persistence) or the raw sensor values (point persistence) constant."))
    A(("fig", "fig6_initial_profile.png", "Figure 1. Initial sensor profile, its piecewise-linear integration with constant ends, and the resulting layer means that start the model."))
    A(("h1", "5. Results"))
    A(("h2", "5.1 Training and selection"))
    A(("tab", "train", "Table 5. Training selection, persistence controls and old/new comparison (JORN A; primary equal-layer RMSE in 10⁻³ m³ m⁻³). Both methods select the same pair and score identically."))
    A(("p", f"The selected d = 0, E = 0 run keeps the integrated initial layer means for ten days; its score ({e3(bh[0])}×10⁻³ m³ m⁻³) is therefore exactly the layer-persistence score ({e3(pl)}×10⁻³), while holding the raw sensor values scores better ({e3(pp)}×10⁻³). The dry-down model with active evaporation scores worse than either control on this window."))
    A(("fig", "fig3_training_grid.png", "Figure 2. Left: harmonic primary RMSE over the 9×6 training grid. Right: geometric − harmonic score at each matching (d, E) pair, in 10⁻⁶ m³ m⁻³ (zero in the d = 0 row)."))
    A(("h2", "5.2 Interface mixing at the nonzero reference and over the whole grid"))
    A(("tab", "refdiff", "Table 6. Geometric − harmonic differences at the fixed nonzero reference (d 0.05, E 1.2). Relative differences are on the harmonic value; no zero denominators were hidden."))
    A(("tab", "env", "Table 7. Largest geometric − harmonic differences over every paired scenario: all 56 for JORN A (the training grid and controls) and the 3 distinct scenarios for each held-out window; the held-out rows are not training-grid values."))
    A(("p", f"Over the 56 paired JORN A training scenarios the largest θ difference is {env['JORN4_A']['theta'] * 1e5:.3f}×10⁻⁵ m³ m⁻³; over the 3 held-out scenarios it is {env['JORN4_B']['theta'] * 1e5:.3f}×10⁻⁵ (JORN B) and {env['MOAB2_C']['theta'] * 1e5:.3f}×10⁻⁵ (MOAB C). The largest primary-score differences are {env['JORN4_A']['rmse'] * 1e6:.3f}, {env['JORN4_B']['rmse'] * 1e6:.3f} and {env['MOAB2_C']['rmse'] * 1e6:.3f}×10⁻⁶ m³ m⁻³ (JORN A, JORN B, MOAB C). The largest difference in final stored water is {env['MOAB2_C']['final_storage_m'] * 1000:.6f} mm (MOAB C), against reference evaporation of {evr}. Every d = 0 difference is exactly zero. These are differences in this model, not statements about measurement detectability: no sensor precision was assessed, and no significance is claimed from three short windows."))
    A(("fig", "fig2_geometric_minus_harmonic.png", "Figure 3. Magnified geometric − harmonic layer moisture at the nonzero reference, by window and layer."))
    A(("h2", "5.3 Held-out transfer and the earlier generic soil"))
    A(("tab", "held", "Table 8. Overall RMSE (10⁻³ m³ m⁻³) for the key pairs, with the earlier generic-soil score at the *identical* pair. G − H in 10⁻⁶. Primary held-out matches use the common H-selected pair (0, 0), where H = G by construction."))
    A(("p", "At the training-selected pair the held-out scores equal the layer-persistence scores of each window and the two methods agree exactly. At the unfitted reference and at the earlier generic best (0, 0.5) the new site-derived properties score worse than the earlier generic soil at the identical pair in all three windows, by more than a factor of 10 at the MOAB reference, where the earlier generic reference scored well while evaporating less water. These two controls are the only pairs I state this for. The statement is scoped: the same (d, E) settings transferred poorly to the derived properties in these windows. The earlier generic soil is not validated truth, so the comparison does not say which soil is closer to the real hydraulics."))
    A(("fig", "fig1_layer_profiles.png", "Figure 4. Layer moisture time series. Black circles: observed end-of-day sensor values (missing days are not drawn or connected); open diamond: observed initial point value; green square and line: integrated initial layer mean and the held d 0, E 0 run; grey: old generic reference; red/blue: new harmonic and geometric at the reference (the two overlap)."))
    A(("fig", "fig4_new_minus_old.png", "Figure 5. New (harmonic) minus old generic overall RMSE at every identical (d, E) pair, JORN A. The (d 0, E 0) cell is exactly zero because both soils then simply hold the initial layer means."))
    A(("tab", "layers", "Table 9. Per-layer sample size, bias (model − observation, 10⁻³ m³ m⁻³) and RMSE (10⁻³). The primary score is the equal-layer RMSE √(mean of layer MSEs); the pooled alternative (all layer errors together) is stored in the JSON results and differs slightly."))
    A(("h2", "5.4 Why the evaporation stress scale dominates"))
    A(("p", f"For the flat bare-soil test the actual daily soil evaporation is E = min(max(W_top − W_res, 0), E_factor × PET × min(1, (θ_top/θ_stress)²)), with W_top and W_res the top-layer water and residual water (m), PET in m, and θ_stress the stress scale. The potential rate applies in full at or above θ_stress and falls quadratically below it; no water is removed below residual. θ1500 is used here as the stress scale, not as a hard threshold where evaporation stops, and that choice is a declared assumption, not a direct field measurement. With the earlier generic scale of 0.18 the initial top-layer multiplier min(1, (θ/θ_stress)²) is {mult['JORN4_A'][0]:.6f} (JORN A), {mult['JORN4_B'][0]:.6f} (JORN B) and {mult['MOAB2_C'][0]:.6f} (MOAB C); with the new scales it is {mult['JORN4_A'][1]:.6f}, {mult['JORN4_B'][1]:.6f} and {mult['MOAB2_C'][1]:.6f}, so the new soils start evaporating at much more of the potential rate. In the d = 0, E = 0.5 control (no exchange, gravity or saturation limit; the residual-availability cap still limits E) the new top layer reaches the 0.01 residual on day 3 in JORN A and day 1 in JORN B, and in MOAB C it is still {S['thr']['MOAB2_C']['final_top']:.6f} on day 6 (old generic top layer at the end of the windows: {S['thr']['JORN4_A']['old_final_top']:.4f}, {S['thr']['JORN4_B']['old_final_top']:.4f} and {S['thr']['MOAB2_C']['old_final_top']:.4f}). This supports the stress-scale interpretation; for d > 0 I do not attribute the difference to a single cause."))
    A(("tab", "thr", "Table 10. Top-layer behaviour in the d 0, E 0.5 control (derived from the stored layer states)."))
    A(("fig", "fig5_evaporation_storage.png", "Figure 6. Cumulative soil evaporation and total stored water (0–30 cm) for the reference and the old generic best, old versus new, by window."))
    A(("h2", "5.5 Gravity, bottom boundary and states below field capacity"))
    A(("p", "In all 62 CPU scenarios per method (124 in total) the daily layer states stay below the new field-capacity values, so percolation, surface return and bottom drainage are exactly zero throughout. The field-capacity and saturation limits never act, and the open and closed-bottom controls are therefore identical here. The gravity step is an instantaneous field-capacity cascade independent of the interface conductivity, while the interface conductivity does act on the d > 0 exchange. **Bottom drainage is not validated.** The runs include no rainfall, infiltration or storm, and the full rainfall/infiltration path remains untested with these properties."))
    A(("h1", "6. Interpretation and limits"))
    A(("bul", [
        "**Averaging choice.** The two means differ by at most 0.24 % in interface conductivity and by at most ~2×10⁻⁵ m³ m⁻³ in water content over these windows. The harmonic mean is preferable on physical grounds for vertical layers, but these homogeneous adjacent Ks values and short windows cannot discriminate between them empirically. Because they are close, this result is not a reason to change any runtime default or recommended setting.",
        "**Parameter transfer.** The derived properties are regression surrogates for two sites with estimated organic carbon and project-approved bounding floors (not Saxton–Rawls recommendations). At the fixed reference (d 0.05, E 1.2) and the earlier generic best (d 0, E 0.5), all three new fits score worse than the earlier generic soil at the identical pair (for example MOAB reference 0.018671 against 0.001520, harmonic). I state this only for those two controls; the whole grid was not summarized that way. The selected d 0, E 0 is the best of a coarse, prespecified grid (E = 0, 0.1, 0.25, …) and does not exclude a physical fit at E between 0 and 0.1. Further tuning would need new hold-out windows. The earlier generic soil is not validated truth, and the new estimates should not be read as validated site hydraulic values.",
        "**Initial-state mismatch.** The layer means that start the model can differ from the first sensor reading when the profile has a gradient, and the layer-persistence control equals the best-scoring model. The model is not outperforming holding the starting state in these windows.",
        "**Scope.** Three short windows (3–8 joint scored days), one 10-day training window, no sensitivity or refit beyond the frozen grid, no storm or transpiration, and no field validation.",
    ]))
    A(("h2", "Possible follow-up (not run)"))
    A(("p", "Sensitivity of the evaporation stress scale and effective density, calibration against field profiles with measured hydraulic properties, observation of evaporation versus transpiration separately, and additional independent hold-out windows would address the limits above. None of these were run for this report, and no new model runs were performed to write it."))
    A(("h1", "Appendix A. Qualification, identities and reproduction"))
    A(("h2", "A.1 Backend and independent checks"))
    A(("p", f"Actual CPU runs (NumPy and Numba) covered 124 scenarios (62 per method: 56 training + 3 + 3 held-out); actual CUDA on GPU 0 (NVIDIA GeForce GTX 1080 Ti, CuPy 14.2.0, driver 13000, runtime 12090) repeated the same scenarios, with NumPy, Numba and CUDA all compared pairwise and a column-state midpoint reconstruction on each backend (not a native lifecycle checkpoint). A separate generic62 repeat of the earlier generic reference configurations also passed. The maximum backend trajectory difference over the new comparisons is {sci(sm['max_backend_trajectory_field_abs'], 3)}; it is the largest of the mixed trajectory fields (the per-step ledger residual, in m³), not a unitless physical error. By physical field the largest CPU/CUDA differences are {pf['water_m']:.3e} m (layer water) and {pf['layer_theta']:.3e} m³ m⁻³ (θ) (`data/root_additional_metrics.json`). The d = 0 runs and the held-out windows are exactly equal CPU/CUDA; this is not claimed to be bitwise for every scenario. An independent 70-digit Decimal check of the declared dry-column equations gave a maximum water error {sm['maximum_decimal_water_error_m']:.3e} m, and the largest native cumulative residual over its bound was {sm['max_native_cumulative_residual_over_bound']:.6f}. The Decimal check is an independent diagnostic, not a formal global error proof."))
    A(("tab", "backend", "Table 11. Single-column whole validation loops including startup and compilation. The CPU loop runs 2 backends (NumPy, Numba) plus the reconstruction; the GPU loop runs 3 backends (NumPy, Numba, CUDA) plus the reconstruction. These are different workloads, not a CPU/GPU performance benchmark. The last column is the largest mixed trajectory field difference (the ledger residual field is in m³), not a unitless physical error."))
    A(("h2", "A.2 Source and package identities"))
    A(("bul", [
        "SYRUP production package (180 files): sha256 37d89f23dbf60626c85f85ea5dd6d1734d3bc7a4aa61d8f2fdfc29f237bcfbd0, source commit bb28af6c58c2892cb3c065d73d33b88285c498f0 (analysis Git base 9adab66e190887a60627dae8a07d58022b32446f), unchanged before and after every run.",
        "MAPLE (412 files): sha256 fbc504bbcd1ec0c02e6a96089134a06187532455ef7ba5e6f146801c8313c02e, commit 9f466d2cfcd52991b24496cf67f4562d6268afa9. LEAF package 38009872f8969f86b56c720adc1b5d9ffefc554fb9cacf90855d999db604530f (28 files). External helper `benchmarks.spatial_mapping` from the qualified LEAF tree (47 260 bytes, sha256 333921bb645d984fa8e7449619d03114262ee333c7fea61a22493a878fd9ccc5). EcoHyD reference code (not executed here): commit 80b10e6b6b82ad16e9fd41a31901cc3157d880e9. All of these are also in `data/qualified_source_identities.json`.",
        "Canonical property receipt root_relative_density_probe.json, sha256 " + sha(DATA / "root_relative_density_probe.json") + ".",
        "All 12 actual configurations (6 CPU, 6 CUDA) were independently validated against the frozen parent bytes (root_config_binding_disposition.json). A review found that an orchestrator expected-hash check was vacuous in the first launch; it was fixed for future runs and a post-hoc source audit proved the outputs unchanged. This is recorded, not a model defect.",
        "Full identities are in the data files: `data/source_guard_cpu/sources_before.json` and `data/source_guard_cuda/sources_before.json` (package sha256 and file counts for maple_syrup, maple and LEAF 38009872f8969f86b56c720adc1b5d9ffefc554fb9cacf90855d999db604530f), `data/frozen_helper_manifest.json` (per-file helper sha256), and the config/result hashes in `data/actual_cpu/completion.json` and `data/actual_cuda/completion.json`.",
        "Original parameter-audit JSON and scripts for raw texture/OM provenance: data/property_audit/.",
    ]))
    A(("h2", "A.3 Reproduction"))
    A(("code", "cd analysis_candidate/docs/subsurface_hydrology/site_properties_mixing\nPYTHONPATH=<task>/report_tools MPLCONFIGDIR=/tmp/syrup-site-mixing-mpl PYTHONDONTWRITEBYTECODE=1 \\\n  /home/okin/MAPLE/.venv/bin/python -B build_report.py"))
    A(("p", "The script reads only data/ and ../observed_drydown/data, re-derives every score, the selection, the integrated initial layers and the envelopes (asserting them against the sealed values), and rewrites report.md, tables/, figures/, verification.md and the DOCX. The model runs themselves are not reproduced by this script."))
    A(("h1", "References"))
    A(("refs", REFS))
    return B


def fill(B, T):
    return B


# ------------------------------------------------------------------ renderers
def md_table(hd, rows):
    hd = [str(h).replace("|", "\\|") for h in hd]
    s = "| " + " | ".join(hd) + " |\n|" + "|".join("---" for _ in hd) + "|\n"
    return s + "\n".join("| " + " | ".join(str(c).replace("|", "\\|") for c in r) + " |" for r in rows) + "\n"


def to_markdown(B, T):
    out = []
    for b in B:
        k = b[0]
        if k == "title":
            out.append(f"# {b[1]}\n")
        elif k == "sub":
            out.append(f"*{b[1]}*\n")
        elif k == "h1":
            out.append(f"## {b[1]}\n")
        elif k == "h2":
            out.append(f"### {b[1]}\n")
        elif k == "p":
            out.append(b[1] + "\n")
        elif k == "eq":
            out.append("    " + b[1] + "\n")
        elif k == "code":
            out.append("```\n" + b[1] + "\n```\n")
        elif k == "bul":
            out.append("\n".join(f"- {x}" for x in b[1]) + "\n")
        elif k == "tab":
            out.append(f"**{b[2]}**\n\n" + md_table(*T[b[1]]))
        elif k == "fig":
            out.append(f"![{b[2]}](figures/{b[1]})\n\n*{b[2]}*\n")
        elif k == "refs":
            out.append("\n".join(f"- **{n}** — {c} <{u}>" for n, c, u in b[1]) + "\n")
    return "\n".join(out)


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


def runs(par, text, size=None, style=None):
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
    for nm, base in (("SiteMixTitle", "Title"), ("SiteMixSubtitle", "Subtitle"), ("SiteMixCaption", "Normal"), ("SiteMixCode", "Normal"), ("SiteMixEquation", "Normal"), ("SiteMixTableText", "Normal")):
        s = st.add_style(nm, WD_STYLE_TYPE.PARAGRAPH)
        s.base_style = st[base]
    t = st["SiteMixTitle"]
    t.font.name, t.font.size, t.font.bold = "Calibri", Pt(22), True
    t.font.color.rgb = RGBColor.from_string("1F3A5F")
    t.paragraph_format.space_after = Pt(8)
    s = st["SiteMixSubtitle"]
    s.font.size, s.font.italic = Pt(11), True
    c = st["SiteMixCaption"]
    c.font.size, c.font.italic = Pt(9), True
    c.paragraph_format.space_after = Pt(10)
    c.paragraph_format.keep_together = True
    k = st["SiteMixCode"]
    k.font.name, k.font.size = "Consolas", Pt(8.5)
    e = st["SiteMixEquation"]
    e.font.name, e.font.size = "Cambria Math", Pt(11)
    e.paragraph_format.left_indent = Inches(0.4)
    tt = st["SiteMixTableText"]
    tt.font.size = Pt(8)
    tt.paragraph_format.space_after = Pt(0)
    tt.paragraph_format.line_spacing = 1.0


def set_cell_width(cell, w):
    cell.width = w


def add_table(doc, hd, rows, width_in, caption):
    cp = doc.add_paragraph(style="SiteMixCaption")
    runs(cp, caption)
    cp.paragraph_format.keep_with_next = True
    cp.paragraph_format.space_after = Pt(3)
    t = doc.add_table(rows=1, cols=len(hd))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    wts = [max(len(str(h)) * 0.6, max(len(str(r[i])) for r in rows) if rows else 4, 5) for i, h in enumerate(hd)]
    tot = sum(wts)
    ws = [Inches(width_in * w / tot) for w in wts]
    for i, h in enumerate(hd):
        c = t.rows[0].cells[i]
        c.text = ""
        c.paragraphs[0].style = "SiteMixTableText"
        r = c.paragraphs[0].add_run(str(h))
        r.bold = True
        shade(c, "D9E2F3")
        c.width = ws[i]
    trPr = t.rows[0]._tr.get_or_add_trPr()
    th = OxmlElement("w:tblHeader")
    th.set(qn("w:val"), "true")
    trPr.append(th)
    for ri, row in enumerate(rows):
        cells = t.add_row().cells
        cant = OxmlElement("w:cantSplit")
        t.rows[-1]._tr.get_or_add_trPr().append(cant)
        for i, v in enumerate(row):
            cells[i].text = ""
            cells[i].paragraphs[0].style = "SiteMixTableText"
            cells[i].paragraphs[0].add_run(str(v))
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
    fr = fp.add_run("SYRUP JORN/MOAB interface-mixing comparison · diagnostic dry-down results, not field validation · page ")
    fr.font.size = Pt(8)
    add_field(fp, "PAGE")
    width = 6.7
    for b in B:
        k = b[0]
        if k == "title":
            doc.add_paragraph(b[1], style="SiteMixTitle")
        elif k == "sub":
            doc.add_paragraph(b[1], style="SiteMixSubtitle")
        elif k == "h1":
            doc.add_heading(b[1], level=1)
        elif k == "h2":
            doc.add_heading(b[1], level=2)
        elif k == "p":
            runs(doc.add_paragraph(), b[1])
        elif k == "eq":
            doc.add_paragraph(b[1], style="SiteMixEquation")
        elif k == "code":
            doc.add_paragraph(b[1], style="SiteMixCode")
        elif k == "bul":
            for x in b[1]:
                runs(doc.add_paragraph(style="List Bullet"), x)
        elif k == "tab":
            hd, rows = T[b[1]]
            wide = len(hd) > 8
            if wide:
                ns = doc.add_section(WD_SECTION.NEW_PAGE)
                ns.orientation = WD_ORIENT.LANDSCAPE
                ns.page_width, ns.page_height = Inches(11), Inches(8.5)
                ns.left_margin = ns.right_margin = Inches(0.7)
                add_table(doc, hd, rows, 9.5, b[2])
                ps = doc.add_section(WD_SECTION.NEW_PAGE)
                ps.orientation = WD_ORIENT.PORTRAIT
                ps.page_width, ps.page_height = Inches(8.5), Inches(11)
                ps.left_margin = ps.right_margin = Inches(0.9)
            else:
                add_table(doc, hd, rows, width, b[2])
        elif k == "fig":
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.keep_with_next = True
            wid = 6.7 if b[1] not in ("fig4_new_minus_old.png",) else 4.4
            p.add_run().add_picture(str(FIG / b[1]), width=Inches(wid))
            cp = doc.add_paragraph(style="SiteMixCaption")
            runs(cp, b[2])
        elif k == "refs":
            for n, c, u in b[1]:
                p = doc.add_paragraph()
                p.paragraph_format.left_indent = Inches(0.25)
                p.paragraph_format.first_line_indent = Inches(-0.25)
                r = p.add_run(f"[{n}] ")
                r.bold = True
                p.add_run(f"{c} {u}").font.size = Pt(9)
    doc.core_properties.title = "SYRUP JORN/MOAB interface-mixing comparison"
    doc.core_properties.author = "SYRUP project"
    doc.save(path)


# ------------------------------------------------------------------ DOCX static verification and verification.md
def verify_docx(path, T, B):
    z = zipfile.ZipFile(path)
    bad = z.testzip()
    names = z.namelist()
    media = [n for n in names if n.startswith("word/media/")]
    xml = z.read("word/document.xml").decode()
    ntab = xml.count("<w:tbl>")
    nhdr = xml.count("<w:tblHeader")
    nsect = xml.count("<w:sectPr")
    nfig = sum(1 for b in B if b[0] == "fig")
    ntabs = sum(1 for b in B if b[0] == "tab")
    foot = any("footer" in n for n in names)
    pagef = "PAGE" in "".join(z.read(n).decode(errors="ignore") for n in names if "footer" in n)
    assert bad is None and len(media) == nfig and ntab == ntabs == nhdr and foot and pagef
    assert not re.search(r"TODO|TBD|XXX|placeholder", re.sub(r"<[^>]+>", " ", xml))
    return {"zip_ok": True, "media": len(media), "tables": ntab, "header_rows_repeat": nhdr, "sections": nsect, "bytes": Path(path).stat().st_size, "sha256": sha(path)}


def write_verification(S, info):
    sm = S["summ"]
    t = f"""# Verification record — JORN/MOAB interface-mixing report

Generated by `build_report.py` on the sealed data in `data/`. Nothing here is a new model run.

## Model/test evidence inherited from the actual runs (Root-reported, sealed in `data/`)
* Root helper acceptance (`data/root_helper_acceptance.json`) and the post-lint rerun (`data/root_postlint_tests_runs/out`): 125 helper tests passed, 2 GPU-only tests skipped. The post-lint equivalence run (`data/root_postlint_equivalence_runs/`, return code 0) supports that the configs, selection, comparisons and property-effect files are byte-identical after the lint-only helper edit. These are inherited Root records, not something this build re-ran.
* Genuine actual runs on every backend: NumPy and Numba on CPU (56 + 56 training scenarios, 3 × 4 held-out), and actual CUDA on GPU 0 (NVIDIA GeForce GTX 1080 Ti) repeating the same 124 scenarios, with NumPy/Numba/CUDA compared pairwise, plus the generic62 repeat of the earlier reference configurations. All source packages (SYRUP 180 files, sha256 37d89f23…; MAPLE 412 files fbc504bb…) were identical before and after every run (`data/source_guard_cpu`, `data/source_guard_cuda`).
* Maximum backend trajectory error over the new comparisons: {sm['max_backend_trajectory_field_abs']!r} (d = 0 results and held-out windows are exactly equal CPU/CUDA).
* Maximum independent 70-digit Decimal water error: {sm['maximum_decimal_water_error_m']!r} m (an independent diagnostic of the declared dry-column equations, not a formal global error proof).
* Maximum native cumulative residual over its bound: {sm['max_native_cumulative_residual_over_bound']!r}.
* The midpoint reconstruction is a per-backend column-state reconstruction, **not** native lifecycle checkpoint acceptance. No checkpoint, scaling or field-validation claim is made. Wall times in the tables are whole validation loops including startup and are not kernel benchmarks.
* All 12 actual configurations were validated against the frozen parent bytes (`data/root_config_binding_disposition.json`); a vacuous expected-hash check in the orchestrator was fixed for future runs and a post-hoc audit proved outputs unchanged.

## Report-build checks (performed by this build)
* Table values are read from the sealed JSON. The builder re-derives and asserts against the saved results: for every scenario in both methods, the primary equal-layer RMSE and the per-layer N, bias and RMSE; the point and layer persistence overall and per-layer values; the training selection; the integrated initial layer means (exact piecewise-linear trapezoid at the layer edges and sensor-depth knots, relative tolerance 1e-9); the old-versus-new RMSE pairing; and the initial evaporation multipliers. Root separately checks all per-layer and cumulative values independently; no claim is made beyond these builder assertions.
* The report asserts the Root-supplied figures against the data: training/held-out largest θ differences 1.6029e-05 / 1.4869e-05 / 1.8389e-05 m³ m⁻³; largest primary-score differences 1.6460e-06 / 2.4582e-06 / 2.2377e-06; largest final-storage difference (MOAB) 0.002754 mm; all layer states below field capacity with zero percolation, drainage and surface return in every run.
* DOCX static check: ZIP integrity OK; {info['media']} embedded figures (equal to the figure blocks); {info['tables']} tables, each with a repeating header row ({info['header_rows_repeat']}); {info['sections']} section definitions (portrait body, landscape pages for wide tables); footer with a PAGE field; no placeholder text. File {info['bytes']} bytes, sha256 `{info['sha256']}`.
* Ruff was run only on `build_report.py`. **No Office paginated render was available**, so the page count and the visual layout were not inspected in Word/LibreOffice. The intended length is about 10–16 pages, which is an estimate only.

## Limits
Diagnostic dry-down comparison on three short NEON windows. Not field validation, not a hydraulic measurement of JORN/MOAB, not a native lifecycle checkpoint, scaling or rainfall/infiltration test; publication status is tracked separately and this build does not establish acceptance or publication.
Possible future steps (none run here): evaporation-stress-scale and density sensitivity, field-profile calibration, observed evaporation versus transpiration, and additional hold-out windows.
"""
    (HERE / "verification.md").write_text(t)


def write_readme(info):
    (HERE / "README.md").write_text(f"""# JORN/MOAB harmonic vs geometric interface-mixing report (2026-11-17)

* `SYRUP_JORN_MOAB_interface_comparison_20261117.docx` — the Word report ({info['bytes']} bytes).
* `report.md` — the editable Markdown version (same text, tables and figures).
* `build_report.py` — regenerates everything below from `data/` (and `../observed_drydown/data` for the earlier generic trajectories).
* `verification.md` — what was verified and what was not.
* `figures/` — PNG figures used in the report; `tables/` — every table as CSV plus `report_tables.json`.
* `data/`, `config_templates/`, `protocol.md` — sealed inputs, read only.

Rebuild (no install, no network, no models):

```
PYTHONPATH=<task>/report_tools MPLCONFIGDIR=/tmp/syrup-site-mixing-mpl PYTHONDONTWRITEBYTECODE=1 \\
  /home/okin/MAPLE/.venv/bin/python -B build_report.py
```

Scope: diagnostic dry-down comparison, not field validation or a native checkpoint/scaling claim. Publication status is tracked separately.
""")


def main():
    R = load()
    S = derive(R)
    fig_profiles(R)
    fig_delta(R, S)
    fig_training(R)
    fig_property(R)
    fig_evap(R)
    fig_initial(R)
    T = build_tables(R, S)
    write_tables(T)
    B = blocks(R, S, T)
    (HERE / "report.md").write_text(to_markdown(B, T))
    to_docx(B, T, DOCX)
    info = verify_docx(DOCX, T, B)
    write_verification(S, info)
    write_readme(info)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
