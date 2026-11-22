"""Builds the JORN/MOAB dry-period fixed-stress comparison report: figures, CSV tables, editable Markdown, verification note, manifest and the Word document.

    PYTHONPATH=<tasks>/site_hydraulics_interface_comparison_20261117/report_tools MPLCONFIGDIR=/tmp/syrup-dry-report PYTHONDONTWRITEBYTECODE=1 /home/okin/MAPLE/.venv/bin/python -B build_report.py

DOCUMENTS ONLY. It refuses to start unless Root's `root_report_ready.json` says PASS and `root_run_seal.json` exists (no result file is read before that). It re-hashes every sealed file before and after the build and
recomputes scores (every valid observation weighted individually), group scores, budgets, the zero-gravity check, the unchanged historical prepared inputs, the preset/case/source identities and the controls independently
from the sealed result files, comparing them with Root's audit. It runs no model, fits nothing and uses no network or GPU. Declared input: DRY_TASK (this task folder; default = the folder four levels above this file).
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import shutil
import sys
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
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
TASK = Path(os.environ.get("DRY_TASK", str(HERE.parents[3])))
CAND = HERE.parents[3] if (HERE.parents[3] / "cases").exists() else TASK / "analysis_candidate"
FIG, TAB, DATA = HERE / "figures", HERE / "tables", HERE / "data"
DOCX = HERE / "SYRUP_JORN_MOAB_dry_period_stress_comparison_20261122.docx"
WINDOWS = ["JORN4_A", "JORN4_B", "MOAB2_C"]
STRESS = {"s1": 0.2017, "s1p5": 0.30255}
KEYS = [f"{w}_{s}" for w in WINDOWS for s in STRESS]
WTEXT = {"JORN4_A": "JORN plot 004, 2025-04-08 to 04-18 (10 days)", "JORN4_B": "JORN plot 004, 2025-04-26 to 04-30 (4 days)", "MOAB2_C": "MOAB plot 002, 2025-05-22 to 05-28 (6 days)"}
WSHORT = {"JORN4_A": "JORN A", "JORN4_B": "JORN B", "MOAB2_C": "MOAB C"}
NDAYS = {"JORN4_A": 10, "JORN4_B": 4, "MOAB2_C": 6}
SV = ["501", "502", "503"]
DEPTH = {"501": "upper sensor (layer 0-10 cm)", "502": "middle sensor (layer 10-20 cm)", "503": "deep sensor (layer 20-30 cm)"}
SC = {"s1": "#1F5FA8", "s1p5": "#C2185B"}
SM = {"s1": "-", "s1p5": "--"}
SL = {"s1": "stress 0.2017", "s1p5": "stress 0.30255"}
SYR, MAP = "4e38a472e268a56491b788d5b1271cc3c3da23bb1cc3221a8c59d90f3cc864d2", "fbc504bbcd1ec0c02e6a96089134a06187532455ef7ba5e6f146801c8313c02e"
HIST = CAND / "docs/subsurface_hydrology/joint_d_stress_calibration/data/actual_cpu"
D_SEL = 0.0010239461376024856
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


def require_ready():
    rr = TASK / "root_report_ready.json"
    if not rr.exists() or not (TASK / "root_run_seal.json").exists():
        sys.exit("NOT READY: root_report_ready.json / root_run_seal.json missing; no result file was read.")
    ready = J(rr)
    if ready.get("status") != "PASS":
        sys.exit(f"NOT READY: root_report_ready.json status is {ready.get('status')!r}; no result file was read.")
    return ready


def seal_check():
    seal = J(TASK / "root_run_seal.json")["files"]
    for p, rec in seal.items():
        b = Path(p).read_bytes()
        assert hashlib.sha256(b).hexdigest() == rec["sha256"] and len(b) == rec["bytes"], p
    return seal


def load():
    ready = require_ready()
    seal = seal_check()
    need = ["root_result_audit.json", "root_warm_timing.json", "root_parameter_verification.json", "root_moab_fc_reference.json", "config_paths.json"]
    R = {"ready": ready, "seal": seal, "idx": J(TASK / "config_paths.json")}
    for n in need:
        assert str(TASK / n) in seal, n
        R[n.split(".")[0]] = J(TASK / n)
    assert list(R["idx"]["cases"]) == KEYS
    for k in KEYS:
        q = TASK / "study" / k
        R[k] = {"res": J(q / "actual/result.json"), "cfg": J(q / "case.json"), "rt": J(q / "runtime.json"), "dec": J(q / "root_decimal_verification.json"), "wall": J(q / "actual_runs/result.json"),
                "src": (J(q / "actual_runs/sources_before.json"), J(q / "actual_runs/sources_after.json"))}
        for rel in ("actual/result.json", "case.json", "runtime.json", "root_decimal_verification.json", "actual_runs/result.json"):
            assert str(q / rel) in seal, (k, rel)
    R["preset_raw"] = J(CAND / "cases/subsurface_validation/jorn_defaults.json")
    return R


def stats(e):
    e = np.asarray(e, dtype=float)
    return {"n": int(e.size), "rmse": 100 * math.sqrt(float(np.mean(e**2))), "bias": 100 * float(e.mean())}


def prep(R):
    for k in KEYS:
        res = R[k]["res"]
        run = next(iter(res["runs"].values()))
        pi = res["plan"]["prepared_inputs"]
        R[k]["run"], R[k]["pi"] = run, pi
        R[k]["theta"] = np.array(run["layer_theta"])
        R[k]["m0"] = np.array(pi["integrated_initial_layer_theta"])
        R[k]["o0"] = np.array(pi["raw_initial_point_theta"])
        R[k]["obs"] = {v: np.array([np.nan if x is None else x for x in pi["observed_end_theta_m3m3"][v]]) for v in SV}
        R[k]["n"] = len(R[k]["theta"])
        assert res["plan"]["probe_layer_index"] == [0, 1, 2]
    return R


def errors(R, k, j, kind):
    th, ob, m0, o0 = R[k]["theta"][:, j], R[k]["obs"][SV[j]], R[k]["m0"][j], R[k]["o0"][j]
    ok = np.isfinite(ob)
    if kind == "raw":
        return th[ok] - ob[ok]
    if kind == "aligned":
        return (th[ok] - m0) - (ob[ok] - o0)
    return o0 - ob[ok]


def group_errors(R, keys, kind, layers):
    return np.concatenate([errors(R, k, j, kind) for k in keys for j in layers])


def scopes():
    """scope name -> (window list, layers) as in Root's audit; evaluated for one stress label."""
    out = {}
    for lay, tag in ((range(3), "all_layers"), (range(2), "upper_two")):
        out[f"all_sites_{tag}"] = (WINDOWS, lay)
        out[f"JORN_{tag}"] = (["JORN4_A", "JORN4_B"], lay)
        out[f"MOAB_{tag}"] = (["MOAB2_C"], lay)
        for w in WINDOWS:
            out[f"{w}_{tag}"] = ([w], lay)
    return out


def group_scores(R):
    G = {}
    for s in STRESS:
        for name, (ws, lay) in scopes().items():
            keys = [f"{w}_{s}" for w in ws]
            G[(s, name)] = {kind: stats(group_errors(R, keys, kind, lay)) for kind in ("raw", "aligned", "persistence")}
    return G


def case_budget(R, k):
    tr = R[k]["run"]["trajectory"]
    perc = np.array(tr["percolation_m"]) * 1000
    dif = np.array(tr["diffusion_m"]) * 1000
    return {"initial": 1000 * R[k]["run"]["initial_storage_m"], "evap": 1000 * float(np.sum(tr["evaporation_m"])), "final": 1000 * R[k]["run"]["final_storage_m"], "drain": 1000 * float(np.sum(tr["drainage_m"])),
            "surface": 1000 * float(np.sum(tr["surface_return_m"])), "grav": perc.sum(0).tolist(), "grav_abs_max": float(np.abs(perc).max()), "grad": dif.sum(0).tolist(), "pet": float(np.sum(R[k]["res"]["plan"]["pet_mm_day"]))}


def review(R):
    """Independent recomputation against Root's audit, decimal controls, parameter verification and the frozen historical inputs; every assertion aborts the build."""
    A = {r["key"]: r for r in R["root_result_audit"]["records"]}
    assert R["root_result_audit"]["status"] == "PASS" and [r["key"] for r in R["root_result_audit"]["records"]] == KEYS
    assert R["root_parameter_verification"]["status"] == "PASS" and R["root_warm_timing"]["status"] == "PASS"
    pre = R["preset_raw"]
    q = pre["parameters"]
    assert q == {"theta_field_capacity_all_layers": 0.1, "theta_evaporation_stress": 0.30255, "evaporation_factor": 1.0, "diffusion_coefficient": D_SEL, "conductivity_multiplier": 1.0, "interface_conductivity_method": "harmonic"}
    assert R["idx"]["preset"]["sha256"] == sha(CAND / "cases/subsurface_validation/jorn_defaults.json") and R["root_parameter_verification"]["preset_sha256"] == R["idx"]["preset"]["sha256"]
    moab_ref = R["root_moab_fc_reference"]["MOAB_implied_density_adjusted_theta33"]
    B, S = {}, {}
    for k in KEYS:
        w, s = k.rsplit("_", 1)
        cfg, res, run, pi = R[k]["cfg"], R[k]["res"], R[k]["run"], R[k]["pi"]
        # ---- configuration really is the declared one
        sc = cfg["scenarios"][0]
        assert abs(cfg["layers"]["theta_evaporation_stress"] - STRESS[s]) < 1e-15 and sc["diffusion_coefficient"] == D_SEL and sc["evaporation_factor"] == 1.0 and sc["bottom"] == "open" and sc["potential_et"] == "weather"
        if w.startswith("JORN"):
            assert cfg["layers"]["theta_field_capacity"] == [0.1, 0.1, 0.1]
        else:
            assert all(abs(a - b) < 1e-15 for a, b in zip(cfg["layers"]["theta_field_capacity"], moab_ref)) and cfg["layers"]["theta_field_capacity"][0] > 0.13
        assert cfg["pins"]["expected_sources"]["maple_syrup"] == {"package_sha256": SYR, "files": 180} and cfg["pins"]["expected_sources"]["maple"] == {"package_sha256": MAP, "files": 412}
        assert res["config_sha256"] == sha(TASK / "study" / k / "case.json") == R["idx"]["cases"][k]["sha256"]
        # ---- sources before/after (driver receipt and launcher record)
        for rec in (res["sources_before"], res["sources_after"]):
            assert (rec["maple_syrup"]["package_sha256"], rec["maple_syrup"]["files"]) == (SYR, 180) and (rec["maple"]["package_sha256"], rec["maple"]["files"]) == (MAP, 412)
        for blk in R[k]["src"]:
            assert {x["module"]: (x["actual_sha256"], x["actual_files"]) for x in blk} == {"maple_syrup": (SYR, 180), "maple": (MAP, 412)}
        # ---- inputs are exactly the historical ones
        assert pi == J(HIST / "prepared_inputs" / f"{w}.json"), (k, "prepared inputs differ from the historical ones")
        n = R[k]["n"]
        assert n == NDAYS[w] and run["trajectory"]["t_s"] == [float(86400 * (i + 1)) for i in range(n)]
        # ---- budgets and the zero-gravity check
        b = case_budget(R, k)
        B[k] = b
        a = A[k]
        assert math.isclose(b["initial"], a["initial_storage_mm"], rel_tol=1e-12) and math.isclose(b["final"], a["final_storage_mm"], rel_tol=1e-9) and math.isclose(b["evap"], a["evaporation_mm"], rel_tol=1e-9)
        assert b["drain"] == 0.0 and b["surface"] == 0.0 and b["grav_abs_max"] == 0.0 and a["bottom_drainage_mm"] == 0.0 and all(x == 0.0 for x in a["gravity_transfer_mm"])
        assert all(math.isclose(x, y, rel_tol=1e-9, abs_tol=1e-12) for x, y in zip(b["grad"], a["net_gradient_transfer_mm_down_positive"]))
        assert abs(b["initial"] - b["evap"] - b["final"]) < 1e-9 * max(1.0, b["initial"])       # no added water, no drainage: storage change = - evaporation
        tr = run["trajectory"]
        assert all(abs(r_) <= bd for r_, bd in zip(tr["step_residual_m3"], tr["step_bound_m3"])) and abs(run["cumulative_ledger"]["residual_m3"]) <= run["cumulative_ledger"]["bound_m3"]
        cum = run["cumulative"]
        assert cum["added_m3"] == 0 and cum["surface_return_m3"] == 0 and cum["drainage_m3"] == 0
        # ---- per-layer and case-pooled scores (every valid observation weighted individually)
        S[k] = {}
        for j, v in enumerate(SV):
            S[k][v] = {kind: stats(errors(R, k, j, kind)) for kind in ("raw", "aligned", "persistence")}
            for kind in ("raw", "aligned", "persistence"):
                got, want = S[k][v][kind], a["scores"][v][kind]
                assert got["n"] == want["n"] and math.isclose(got["rmse"], want["rmse_pp"], rel_tol=1e-9, abs_tol=1e-12) and math.isclose(got["bias"], want["bias_pp"], rel_tol=1e-9, abs_tol=1e-12), (k, v, kind)
        S[k]["pooled"] = {kind: stats(group_errors(R, [k], kind, range(3))) for kind in ("raw", "aligned", "persistence")}
        for kind in ("raw", "aligned", "persistence"):
            assert S[k]["pooled"][kind]["n"] == a["pooled"][kind]["n"] and math.isclose(S[k]["pooled"][kind]["rmse"], a["pooled"][kind]["rmse_pp"], rel_tol=1e-9)
        # ---- controls
        assert run["restart"]["whole_trajectory_and_cumulative_identical"] is True and run["restart"]["after_day"] >= 1
        assert sorted(run["restart"]["backends"]) == ["numba", "numpy"] and a["restart"]["whole_trajectory_and_cumulative_identical"] is True
        dec = R[k]["dec"]
        assert dec["status"] == "PASS" and dec["decimal_digits"] == 90 and dec["days"] == n and dec["bottom_drainage_mm"] == 0.0 and all(x == 0.0 for x in dec["gravity_transfer_mm"]) and math.isclose(dec["evaporation_mm"], b["evap"], rel_tol=1e-9)
        assert R[k]["wall"]["returncode"] == 0
    G = group_scores(R)
    for (s, name), kinds in G.items():
        want = R["root_result_audit"]["groups"][str(STRESS[s])][name]
        for kind in ("raw", "aligned", "persistence"):
            assert kinds[kind]["n"] == want[kind]["n"] and math.isclose(kinds[kind]["rmse"], want[kind]["rmse_pp"], rel_tol=1e-9, abs_tol=1e-12) and math.isclose(kinds[kind]["bias"], want[kind]["bias_pp"], rel_tol=1e-9, abs_tol=1e-12), (s, name, kind)
    for s in STRESS:
        n = lambda nm, s=s: G[(s, nm)]["aligned"]["n"]
        assert (n("all_sites_all_layers"), n("JORN_all_layers"), n("MOAB_all_layers"), n("all_sites_upper_two"), n("JORN_upper_two"), n("MOAB_upper_two")) == (53, 39, 14, 36, 26, 10)
    assert R["root_result_audit"]["all_gravity_and_bottom_drainage_zero"] is True and R["root_result_audit"]["t0_excluded"] is True
    wt = {r["key"]: r for r in R["root_warm_timing"]["rows"]}
    assert list(wt) == KEYS and all(r["exact_trajectory_to_primary"] for r in wt.values())
    return B, S, G


# ------------------------------------------------------------------------------------------------ figures
def style(ax):
    ax.grid(alpha=0.25)
    ax.tick_params(labelsize=7.5)


def fig_grid(R, aligned, name, title):
    fig, axs = plt.subplots(3, 3, figsize=(7.4, 7.8), sharex="row")
    for r, w in enumerate(WINDOWS):
        for j, v in enumerate(SV):
            ax = axs[r, j]
            n = R[f"{w}_s1"]["n"]
            for s in STRESS:
                k = f"{w}_{s}"
                m0, o0 = R[k]["m0"][j], R[k]["o0"][j]
                y = np.r_[m0, R[k]["theta"][:, j]] * 100 + ((o0 - m0) * 100 if aligned else 0)
                ax.plot(range(n + 1), y, color=SC[s], ls=SM[s], lw=1.5, label=SL[s])
            k = f"{w}_s1"
            m0, o0 = R[k]["m0"][j], R[k]["o0"][j]
            ob = R[k]["obs"][v]
            ax.scatter([0], [(o0 if aligned else m0) * 100], marker="^", color="#2E8B57", s=46, zorder=5, label="initial model layer mean" + (" (shifted onto the sensor)" if aligned else ""))
            ax.scatter([0], [o0 * 100], facecolors="none", edgecolors="black", s=34, lw=0.9, zorder=6, label="initial sensor point value")
            ok = np.isfinite(ob)
            ax.scatter(np.arange(1, n + 1)[ok], ob[ok] * 100, facecolors="none", edgecolors="black", s=28, lw=0.9, zorder=6, label="sensor point value (gaps not filled)")
            if r == 0:
                ax.set_title(DEPTH[v], fontsize=8)
            if j == 0:
                ax.set_ylabel(f"{WSHORT[w]}\nwater content (vol. %)", fontsize=8)
            if r == 2:
                ax.set_xlabel("days since the start of the window", fontsize=8)
            style(ax)
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=7, frameon=False)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    fig.savefig(FIG / name, dpi=150)
    plt.close(fig)


UP_ORDER = [("JORN4_A_upper_two", "JORN A"), ("JORN4_B_upper_two", "JORN B"), ("JORN_upper_two", "JORN\npooled"), ("MOAB2_C_upper_two", "MOAB C"), ("MOAB_upper_two", "MOAB\npooled"), ("all_sites_upper_two", "combined")]


def fig_scores(G):
    fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.6))
    xs = np.arange(len(UP_ORDER))
    w = 0.38
    for ax, kind, ttl in ((axs[0], "raw", "RAW (model layer mean vs sensor point)"), (axs[1], "aligned", "ALIGNED (day-0 offset removed)")):
        for i, s in enumerate(STRESS):
            ax.bar(xs + (i - 0.5) * w, [G[(s, nm)][kind]["rmse"] for nm, _ in UP_ORDER], w, color=SC[s], label=SL[s])
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{lab}\nn = {G[('s1', nm)][kind]['n']}" for nm, lab in UP_ORDER], fontsize=6)
        ax.set_ylabel("upper-two RMSE (percentage points)", fontsize=8)
        ax.set_title(ttl, fontsize=8.5)
        style(ax)
    axs[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_upper_two_scores.png", dpi=150)
    plt.close(fig)


def fig_budget(B):
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.4))
    xs = np.arange(len(WINDOWS))
    w = 0.38
    for ax, key, lab in ((axs[0], "evap", "actual evaporation (mm)"), (axs[1], "final", "final storage (mm)")):
        for i, s in enumerate(STRESS):
            ax.bar(xs + (i - 0.5) * w, [B[f"{wd}_{s}"][key] for wd in WINDOWS], w, color=SC[s], label=SL[s])
        if key == "final":
            ax.scatter(xs, [B[f"{wd}_s1"]["initial"] for wd in WINDOWS], marker="_", s=300, color="black", zorder=5, label="initial storage")
        ax.set_xticks(xs)
        ax.set_xticklabels([WSHORT[wd] for wd in WINDOWS], fontsize=8)
        ax.set_ylabel(lab, fontsize=8)
        ax.legend(fontsize=6.5, frameon=False)
        style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_budgets.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ tables
def g(x, n=3):
    return f"{x:.{n}f}"


def tables(R, B, S, G):
    T = {}
    fcs = lambda k: " / ".join(g(100 * x, 2) for x in R[k]["cfg"]["layers"]["theta_field_capacity"])
    T["cases"] = (["Case", "Window", "Field capacity per layer (vol. %)", "Stress", "E", "d"],
                  [[k, WTEXT[k.rsplit("_", 1)[0]], fcs(k) if k.startswith("MOAB") else "10.00 / 10.00 / 10.00", g(R[k]["cfg"]["layers"]["theta_evaporation_stress"], 5), "1", f"{D_SEL:.10f}"] for k in KEYS])
    rows = [("JORN4_A_upper_two", "JORN A (Apr 8-18)"), ("JORN4_B_upper_two", "JORN B (Apr 26-30)"), ("JORN_upper_two", "JORN pooled"), ("MOAB2_C_upper_two", "MOAB C (May 22-28)"), ("MOAB_upper_two", "MOAB pooled"), ("all_sites_upper_two", "Combined JORN + MOAB")]
    T["upper_two"] = (["Upper two layers", "n", "RAW, stress 0.2017", "RAW, stress 0.30255", "Aligned, stress 0.2017", "Aligned, stress 0.30255"],
                      [[lab, G[("s1", nm)]["raw"]["n"]] + [g(G[(s, nm)][kind]["rmse"]) for kind in ("raw", "aligned") for s in STRESS] for nm, lab in rows])
    T["upper_two_bias"] = (["Upper two layers: bias (percentage points)", "n", "RAW, stress 0.2017", "RAW, stress 0.30255", "Aligned, stress 0.2017", "Aligned, stress 0.30255"],
                           [[lab, G[("s1", nm)]["raw"]["n"]] + [f"{G[(s, nm)][kind]['bias']:+.3f}" for kind in ("raw", "aligned") for s in STRESS] for nm, lab in rows])
    rows3 = [(nm.replace("upper_two", "all_layers"), lab) for nm, lab in rows]
    T["all_layers"] = (["All three layers (context)", "n", "RAW, stress 0.2017", "RAW, stress 0.30255", "Aligned, stress 0.2017", "Aligned, stress 0.30255"],
                       [[lab, G[("s1", nm)]["raw"]["n"]] + [g(G[(s, nm)][kind]["rmse"]) for kind in ("raw", "aligned") for s in STRESS] for nm, lab in rows3])
    T["no_change"] = (["No-change reference (day-0 sensor value held), RMSE", "n upper two", "Upper two", "n all three", "All three"],
                      [[lab, G[("s1", nm)]["persistence"]["n"], g(G[("s1", nm)]["persistence"]["rmse"]), G[("s1", nm3)]["persistence"]["n"], g(G[("s1", nm3)]["persistence"]["rmse"])] for (nm, lab), (nm3, _) in zip(rows, rows3)])
    drows = []
    for w in WINDOWS:
        for v in SV:
            for s in STRESS:
                k = f"{w}_{s}"
                drows.append([k, v, S[k][v]["raw"]["n"], S[k][v]["raw"]["rmse"], S[k][v]["raw"]["bias"], S[k][v]["aligned"]["rmse"], S[k][v]["aligned"]["bias"], S[k][v]["persistence"]["rmse"]])
    T["_depth_scores"] = (["case", "sensor", "n", "raw_rmse_pp", "raw_bias_pp", "aligned_rmse_pp", "aligned_bias_pp", "no_change_rmse_pp"], drows)
    T["budget"] = (["Case", "Initial storage (mm)", "Evaporation (mm)", "Final storage (mm)", "Upper interface net exchange (mm)", "Lower interface net exchange (mm)"],
                   [[k, g(B[k]["initial"], 3), g(B[k]["evap"], 3), g(B[k]["final"], 3), f"{B[k]['grad'][0]:+.3f}", f"{B[k]['grad'][1]:+.3f}"] for k in KEYS])
    T["evap_effect"] = (["Window", "PET sum (mm)", "E, stress 0.2017 (mm)", "E, stress 0.30255 (mm)", "Ratio of the two", "Actual E / PET at 0.30255 (%)"],
                        [[WSHORT[w], g(B[f"{w}_s1"]["pet"], 2), g(B[f"{w}_s1"]["evap"], 3), g(B[f"{w}_s1p5"]["evap"], 3), g(B[f"{w}_s1p5"]["evap"] / B[f"{w}_s1"]["evap"], 3), g(100 * B[f"{w}_s1p5"]["evap"] / B[f"{w}_s1"]["pet"], 1)] for w in WINDOWS])
    wt = {r["key"]: r for r in R["root_warm_timing"]["rows"]}
    T["timing"] = (["Case", "Days", "Fresh driver bundle (s)", "Whole fresh command (s)", "Warm native column call (s)"],
                   [[k, R[k]["n"], g(R[k]["rt"]["driver_call_wall_s"], 2), g(R[k]["wall"]["seconds"], 2), g(wt[k]["warm_native_column_call_s"], 4)] for k in KEYS])
    A = {r["key"]: r for r in R["root_result_audit"]["records"]}
    T["checks"] = (["Case", "Numba vs NumPy, max abs difference", "Restart reconstruction", "Ledger residual / bound (m³)", "Layer budget error (m)", "90-digit control, max error (m)"],
                   [[k, f"{max(v for d_ in R[k]['run']['backend_agreement_max_abs'].values() for v in d_.values()):.1e}", "identical (same process)",
                     f"{abs(R[k]['run']['cumulative_ledger']['residual_m3']):.1e} / {R[k]['run']['cumulative_ledger']['bound_m3']:.2e}", f"{A[k]['max_layer_budget_error_m']:.1e}", f"{R[k]['dec']['max_water_error_m']:.1e}"] for k in KEYS])
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
        (4, "NEON (RELEASE-2026). Soil water content and water salinity, DP1.00094.001 (Jornada plot 004 and Moab plot 002, sensors 501-503).", "https://data.neonscience.org/data-products/DP1.00094.001/RELEASE-2026"),
        (5, "NEON (RELEASE-2026). Precipitation, DP1.00045.001 (used to define the rain-free windows).", "https://data.neonscience.org/data-products/DP1.00045.001/RELEASE-2026"),
        (6, "NEON (RELEASE-2026). Single aspirated air temperature, DP1.00002.001 (air temperature only; no soil temperature enters the evaporation demand).", "https://data.neonscience.org/data-products/DP1.00002.001/RELEASE-2026"),
        (7, "NEON (RELEASE-2026). Soil physical and chemical properties, Megapit, DP1.00096.001 (Jornada pit of 2013 and Moab pit of 2014; MOAB pit 1, collected 2014-02-24): texture, organic carbon and bulk density behind the layer properties. A spatial proxy for the plots, not demonstrably co-located.", "https://data.neonscience.org/data-products/DP1.00096.001/RELEASE-2026"),
        (8, "SYRUP project. Joint fit of the diffusion multiplier d and the evaporation-stress threshold at E = 1 (docs/subsurface_hydrology/joint_d_stress_calibration/report.md); the SR33 field-capacity, evaporation and uniform field-capacity sensitivity reports of 2026-11-20 (SYRUP_JORN_160day_SR33_K_sensitivity, ..._SR33_1x_evaporation_sensitivity and ..._uniform_FC_sensitivity).", "")]


def lower_of(G, nm, kind):
    a, b = G[("s1", nm)][kind]["rmse"], G[("s1p5", nm)][kind]["rmse"]
    return ("0.30255" if b < a else "0.2017"), a, b


def blocks(R, B, S, G):
    pre = R["preset_raw"]
    nm_j, nm_m, nm_c = "JORN_upper_two", "MOAB_upper_two", "all_sites_upper_two"
    wl = {w: lower_of(G, f"{w}_upper_two", "aligned")[0] for w in WINDOWS}
    n_lower_b = sum(v == "0.30255" for v in wl.values())
    wl_raw = {w: lower_of(G, f"{w}_upper_two", "raw")[0] for w in WINDOWS}

    def pair(nm, kind):
        w_, a, b = lower_of(G, nm, kind)
        return f"{a:.3f} at stress 0.2017 and {b:.3f} at 0.30255 (lower at {w_})"

    ratios = {w: B[f"{w}_s1p5"]["evap"] / B[f"{w}_s1"]["evap"] for w in WINDOWS}
    dfin = {w: B[f"{w}_s1p5"]["final"] - B[f"{w}_s1"]["final"] for w in WINDOWS}
    assert all(r_ > 1 / 2.25 and r_ < 1.0 for r_ in ratios.values()) and all(v > 0 for v in dfin.values())      # statements below about the ratio and the final storage are checked against the data
    assert all(R[f"{w}_{x}"]["theta"].max() < 0.2017 for w in WINDOWS for x in STRESS)
    assert all((R[k]["theta"].max(0) < np.array(R[k]["cfg"]["layers"]["theta_field_capacity"])).all() and (R[k]["m0"] < np.array(R[k]["cfg"]["layers"]["theta_field_capacity"])).all() for k in KEYS)   # every layer stays below its field capacity
    moab_fc = [100 * x for x in R["MOAB2_C_s1"]["cfg"]["layers"]["theta_field_capacity"]]
    ini = {w: [100 * x for x in R[f"{w}_s1"]["m0"]] for w in WINDOWS}
    return [
        ("title", "Fixed evaporation stress 0.2017 against 0.30255 in the earlier JORN and MOAB dry windows"),
        ("sub", "Six native runs of the three rain-free 2025 windows (JORN A and B, MOAB C) at two fixed stress values, with the user-selected JORN defaults. A comparison, not a calibration, and not an independent validation."),
        ("h1", "1. Summary"),
        ("bul", ["**The user's JORN selection stands.** For future JORN runs the user chose a field capacity of 10 vol. % in all three layers and an evaporation stress of 0.30255 (with E = 1, SR33 1x conductivities with harmonic interfaces and d = 0.0010239461376024856). It is recorded in the versioned preset cases/subsurface_validation/jorn_defaults.json and in a new AGENTS paragraph; the comparison helper loads the file, the generator builds the JORN configurations from it and the run-time check validates the JORN field capacity, d and E against it and accepts only the preset stress or the declared comparison stress 0.2017 (the explicit exception used here), refusing a JORN configuration that disagrees. It is a JORN case-level visual choice, not a new calibration, not a global model default and not a MOAB default, and frozen older configurations and restart identities are not migrated. It persists whatever this comparison shows; nothing here suggests reversing it.",
                f"**Upper two layers (the primary focus).** Aligned RMSE (percentage points): JORN pooled (n = 26) {pair(nm_j, 'aligned')}; MOAB (n = 10) {pair(nm_m, 'aligned')}; combined (n = 36) {pair(nm_c, 'aligned')}. RAW RMSE: JORN {pair(nm_j, 'raw')}; MOAB {pair(nm_m, 'raw')}; combined {pair(nm_c, 'raw')}. By window the aligned upper-two RMSE is lower at stress 0.30255 in {n_lower_b} of 3 windows (JORN A: {wl['JORN4_A']}; JORN B: {wl['JORN4_B']}; MOAB C: {wl['MOAB2_C']}).",
                f"**All three layers are kept as context.** Aligned RMSE over all 39 JORN values {pair('JORN_all_layers', 'aligned')}; over the 14 MOAB values {pair('MOAB_all_layers', 'aligned')}; over all 53 values {pair('all_sites_all_layers', 'aligned')}. The third (deep) layer is scored and retained in these numbers; it is not investigated here (see Section 5).",
                f"**The field capacity cannot be tested by these windows.** In all six cases the native columns show exactly zero gravity transfer between layers and zero bottom drainage (every layer stays below its field capacity: the initial layer contents are {min(min(v) for v in ini.values()):.1f}-{max(max(v) for v in ini.values()):.1f} vol. %). The field capacity is therefore not identifiable from these dry windows, and the dry-window agreement does not validate the 10 vol. % choice.",
                f"**Water.** At a fixed water content the higher stress reduces the evaporation, but the cumulative effect is smaller because the layer stays wetter and evaporates more later: the ratio of cumulative evaporation at 0.30255 to that at 0.2017 is {min(ratios.values()):.3f}-{max(ratios.values()):.3f} across the windows (the fixed-moisture ratio would be 0.444). No water is added, no water leaves through the bottom or the surface, and the internal gradient exchanges are not losses.",
                "**What this is not.** The earlier 2025 windows were used to fit d and the stress (jointly, at E = 1, with T = 0 and the old field capacity), so this is not held-out validation. T = 0 and rain-free weather are not proof that no transpiration occurred, and the earlier fit may have absorbed missing transpiration. No refit, transpiration or new physics was added."]),
        ("h1", "2. What was run"),
        ("p", "The three earlier dry windows (Table 1 of the joint d/stress report [8]) are unchanged: JORN A (plot 004, 10 days from 2025-04-08), JORN B (plot 004, 4 days from 2025-04-26) and MOAB C (plot 002, 6 days from 2025-05-22). The observed NEON soil water [4], rain-free definition [5], air temperature [6] and Hargreaves-Samani potential evaporation [3], the integrated initial profile and the QA of the 53 valid observations (JORN 39, MOAB 14; the first instant of each window is excluded and gaps stay gaps) are exactly the historical ones; the prepared inputs of each result equal the stored historical prepared inputs. Each case is a one-square-metre column of three 10 cm layers with an open bottom, gravity on, one 86,400 s step per day, no rain, no transpiration, no vegetation, wind or bed, and E = 1."),
        ("p", f"Only these things differ from the historical configurations: the field capacity, the stress, the scenario d (0.05 to {D_SEL:.10f}), the SYRUP package pin (the old 37d89f23 pin is still refused unmodified) and names and provenance. For JORN the field capacity is the preset's 10 vol. % in all layers; the saturation, residual content, Ks and interface conductivities are the accepted JORN values and were not recomputed. MOAB keeps its own field capacity, {moab_fc[0]:.2f} / {moab_fc[1]:.2f} / {moab_fc[2]:.2f} vol. %, from the current native Saxton-Rawls provider [2] applied to the original MOAB texture, organic matter and measured density [7] (a spatial proxy for the 2025 plot); it is neither JORN's 10 vol. % nor the old 6 kPa value. The two stress values are the exact rounded 0.2017 (the earlier fitted value was 0.20170581117963332) and 0.30255 = 1.5 x 0.2017. Each evaluation runs the Numba and the NumPy backend and a same-process reconstruction of the column from its public state at a midpoint (24 native window evaluations in all). No Green-Ampt storm calculation applies in a rain-free column."),
        ("tab", "cases", "Table 1. The six cases."),
        ("h1", "3. The user's selection and what it means"),
        ("p", f"The preset (schema syrup.jorn_defaults/1, version 1, SHA-256 {sha(CAND / 'cases/subsurface_validation/jorn_defaults.json')[:16]}...) records the selection with its source (accepted study commit {pre['source']['accepted_study_commit'][:8]}, tracked case file SHA-256 {pre['source']['case_file_sha256'][:16]}...), the d provenance (the earlier joint fit) and the statement that the choice is not a calibration or refit. It also records that the unresolved model-versus-sensor discrepancy of the 20-30 cm layer at the January 2024 storm is deferred by the user's decision, and that T = 0 is not required for future LEAF coupling (it holds only in the present controlled experiments). The stress threshold is the water content at which the full potential rate is allowed; below it the demand is multiplied by (water content / threshold)², and a separate residual content stops withdrawal. It is a proxy for the rising resistance of drying soil, not a measured property [1]."),
        ("h1", "4. Upper two layers: JORN, MOAB and combined"),
        ("p", "Scores are RMSE in percentage points (100 x m³/m³) with every valid observation weighted individually (pooled groups are not averages of per-layer scores). RAW compares the model layer mean with the sensor point value. Aligned subtracts one constant day-0 offset per sensor (the integrated initial layer mean minus the initial point value); it shifts the plotted model only, and the physical water is never altered. In the aligned view the green initial model triangle and the open black initial sensor circle coincide by construction."),
        ("tab", "upper_two", "Table 2a. Upper two layers: RMSE (percentage points) at the two stress values. n = number of sensor values (the day-0 instant is excluded). MOAB C and MOAB pooled are the same 10 targets because there is only one MOAB window; the pooled row is not additional data. The combined count of 36 is exact."),
        ("tab", "upper_two_bias", "Table 2b. Upper two layers: bias (model minus sensor, percentage points). A positive bias means the model is wetter than the sensor."),
        ("fig", "fig1_raw_grid.png", "Figure 1. RAW water content, three windows (rows) by three sensors (columns): model layer mean at stress 0.2017 (solid blue) and 0.30255 (dashed magenta), sensor point values (open black circles; gaps are not filled), the initial sensor value and the initial model layer mean (green triangle). Field capacities (JORN 10 vol. %, MOAB 13.9-15.6 vol. %) lie above the plotted range and are not drawn."),
        ("fig", "fig2_aligned_grid.png", "Figure 2. The same runs in the aligned view: the plotted model is shifted by the day-0 offset so that its initial value coincides with the sensor's."),
        ("fig", "fig3_upper_two_scores.png", "Figure 3. Upper-two RMSE by window and pooled group (RAW left, aligned right); n is the number of sensor values."),
        ("p", f"Whether a window prefers one stress over the other depends on the scoring and on the window: the aligned upper-two RMSE is lower at stress 0.30255 in {n_lower_b} of 3 windows, and the RAW upper-two RMSE in {sum(v == '0.30255' for v in wl_raw.values())} of 3. The pooled and combined groups (Table 2a) are the summary; with 10, 26 or 36 values from three short, correlated windows no significance is claimed, and the earlier fit used these windows, so the comparison cannot validate either stress."),
        ("h1", "5. All three layers (context)"),
        ("tab", "all_layers", "Table 3a. All three layers: RMSE (percentage points) (JORN 39, MOAB 14, total 53 values). MOAB C and MOAB pooled are the same 14 targets (one MOAB window); the combined count of 53 is exact."),
        ("tab", "no_change", "Table 3b. No-change reference: the day-0 sensor value held (RMSE, percentage points). The per-depth scores for every case are in the CSV tables."),
        ("p", "The deep (third) layer is scored and retained in these numbers. By the user's decision, the unresolved model-versus-sensor discrepancy of the 20-30 cm layer is deferred: no sensor, wetting-front or support-cause investigation, censoring or recalibration is made or implied here, and no sensor fault is declared."),
        ("h1", "6. Water budgets"),
        ("tab", "budget", "Table 4. Water budget per case. No water is added, and the bottom drainage and surface return are exactly zero in all six cases (checked from the trajectories); the two interface columns are net gradient exchanges (downward positive, upward negative) between layers: internal transfers, not losses, and not added to the storage change. Gravity transfer between layers is also exactly zero in every case."),
        ("tab", "evap_effect", "Table 5. Effect of the stress on the cumulative evaporation."),
        ("p", f"With the stress raised 1.5 times, the evaporation at a fixed water content is multiplied by 1/1.5² = 0.444 (the quadratic branch applies, since every layer stays below both stress values). The cumulative evaporation changes less: the ratios in Table 5 are {min(ratios.values()):.3f}-{max(ratios.values()):.3f}, because the top layer stays wetter and loses more later. The final storage is higher at the higher stress by {min(dfin.values()):.3f}-{max(dfin.values()):.3f} mm. These numbers come from the sealed trajectories, not from fixed ratios."),
        ("fig", "fig4_budgets.png", "Figure 4. Actual evaporation and final storage per window at the two stress values (the black bars mark the initial storage)."),
        ("h1", "7. Controls, identities and run time"),
        ("tab", "checks", "Table 6. Per case: Numba against NumPy on the whole trajectory (maximum absolute difference of any recorded quantity), the midpoint public-state reconstruction, the native ledger and layer budget, and Root's independent 90-digit control."),
        ("bul", [f"All files of Root's seal ({len(R['seal'])}) were re-hashed before and after this build. From the sealed result files I recomputed every score (also the group scores, with each valid observation weighted individually), the budgets, the zero gravity transfer and zero bottom drainage, the unchanged historical prepared inputs, the preset, case and source values, and the controls; all agree with Root's audit (relative 1e-9).",
                 "Sources: SYRUP package 180 files (4e38a472...) and MAPLE package 412 files (fbc504bb...) before and after every run; the unmodified historical 37d89f23 pin is still refused.",
                 "The restart control reconstructs the column from its public state (water and time) at a midpoint in the same process and reproduces the whole trajectory exactly. It is not a fresh-process test and not a native checkpoint; no coupled, GPU or scaling claim is made."]),
        ("tab", "timing", "Table 7. Timing, separated and labelled. The fresh driver bundle is one driver call per case and includes input and source checks, JIT compilation, four native window evaluations (Numba, NumPy and the two restart-control runs), scoring and output; the whole fresh command adds interpreter start and the source-verified bootstrap. The warm native column call is Root's timed call of the native column for one window (operator preparation, arrays and daily updates; no input preparation, scoring or output) on the Numba backend after a warm-up. Not kernel-only timing and not a controlled machine benchmark; one CPU column. Differences between cases reflect the shared host and the first-run conditions, not the model."),
        ("h1", "8. Limits"),
        ("bul", ["A comparison of two fixed values, with no refit. The earlier joint fit of d and the stress used these same windows, so the comparison is not held-out validation, and d is conditional on E = 1, T = 0 and the old field capacity.",
                 "The model has no transpiration, and rain-free windows do not prove that none occurred; the earlier fit may have absorbed missing transpiration. This report does not claim transpiration or plant activity was measured.",
                 "The field capacity cannot be tested by these dry windows (gravity transfer is zero in all cases). The user's JORN selection does not depend on this comparison and persists regardless of the ranking.",
                 "Three short windows, point sensors against 10 cm layer means, the soil pits (JORN 2013, MOAB 2014) as a spatial proxy, not demonstrably co-located, and no uncertainty bound for support. The unresolved 20-30 cm January 2024 discrepancy is deferred by the user and not investigated here."]),
        ("h1", "References"), ("refs", REFS),
        ("h1", "Appendix. Identity (compact)"),
        ("p", "Sources: SYRUP package 180 files, SHA-256 " + SYR + "; MAPLE package 412 files, SHA-256 " + MAP + ". Preset SHA-256 " + sha(CAND / "cases/subsurface_validation/jorn_defaults.json") + ". Case-file hashes: " + "; ".join(f"{k} {sha(TASK / 'study' / k / 'case.json')[:12]}" for k in KEYS)
         + ". The hash of every sealed file is in tables/provenance_seal.csv and artifact_manifest.json; full trajectories, per-depth scores, budgets and the preset values are in the CSV tables. Raw NEON files are not copied.")]
def to_docx(Bk, T, path):
    doc = Document()
    setup_styles(doc)
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Inches(8.5), Inches(11)
    sec.left_margin = sec.right_margin = Inches(0.9)
    sec.top_margin = sec.bottom_margin = Inches(0.8)
    fp = sec.footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    fr = fp.add_run("SYRUP JORN/MOAB dry windows: fixed stress 0.2017 vs 0.30255 · comparison, not a recalibration · page ")
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
            p.add_run().add_picture(str(FIG / b[1]), width=Inches(6.4 if b[1] in ("fig1_raw_grid.png", "fig2_aligned_grid.png") else 6.2))
            runs(doc.add_paragraph(style="RepCaption"), b[2])
        elif k == "refs":
            for n, c, u in b[1]:
                p = doc.add_paragraph()
                p.paragraph_format.left_indent = Inches(0.25)
                p.paragraph_format.first_line_indent = Inches(-0.25)
                p.paragraph_format.space_after = Pt(2)
                p.add_run(f"[{n}] ").bold = True
                p.add_run(f"{c} {u}".strip()).font.size = Pt(9)
    doc.core_properties.title = "SYRUP JORN/MOAB dry-period stress comparison"
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
    B, S, G = review(R)
    for p in list(TAB.glob("*.csv")) + list(DATA.glob("*")):
        p.unlink()
    fig_grid(R, False, "fig1_raw_grid.png", "RAW water content in the three dry windows at two fixed stress values")
    fig_grid(R, True, "fig2_aligned_grid.png", "Aligned view: the plotted model starts on the sensor's initial value")
    fig_scores(G)
    fig_budget(B)
    T = tables(R, B, S, G)
    for name, (hd, rows) in T.items():
        csvw(TAB / f"{name.lstrip('_')}.csv", hd, rows)
    csvw(TAB / "provenance_seal.csv", ["file", "sha256", "bytes"], [[k, v["sha256"], v["bytes"]] for k, v in R["seal"].items()])
    csvw(TAB / "case_hashes.csv", ["case", "config_sha256_in_result", "case_json_sha256"], [[k, R[k]["res"]["config_sha256"], sha(TASK / "study" / k / "case.json")] for k in KEYS])
    csvw(TAB / "preset_values.csv", ["parameter", "value"], [[k, v] for k, v in R["preset_raw"]["parameters"].items()] + [["preset_sha256", sha(CAND / "cases/subsurface_validation/jorn_defaults.json")]])
    traj = []
    for k in KEYS:
        tr = R[k]["run"]["trajectory"]
        for i in range(R[k]["n"]):
            traj.append([k, i + 1, tr["t_s"][i]] + [R[k]["theta"][i, j] for j in range(3)] + [R[k]["obs"][v][i] if np.isfinite(R[k]["obs"][v][i]) else "" for v in SV] + [tr["evaporation_m"][i], tr["drainage_m"][i]] + list(tr["percolation_m"][i]) + list(tr["diffusion_m"][i]))
    csvw(TAB / "daily_trajectories.csv", ["case", "day", "t_s", "theta_l1", "theta_l2", "theta_l3", "obs_501", "obs_502", "obs_503", "evaporation_m3", "bottom_drainage_m3", "gravity_l1_l2_m", "gravity_l2_l3_m", "diffusion_l1_l2_m", "diffusion_l2_l3_m"], traj)
    csvw(TAB / "group_scores_all.csv", ["stress", "scope", "scoring", "n", "rmse_pp", "bias_pp"], [[SL[s], nm, kind, v["n"], v["rmse"], v["bias"]] for (s, nm), kinds in G.items() for kind, v in kinds.items()])
    csvw(TAB / "driver_native_scores.csv", ["case", "driver_overall_rmse_mean_of_layer_mse_m3m3", "driver_pooled_rmse_alternative_m3m3", "label"],
         [[k, next(iter(R[k]["res"]["scores"].values()))["overall_rmse"], next(iter(R[k]["res"]["scores"].values()))["pooled_rmse_alternative"], "native driver metrics (per-layer-weighted overall RMSE differs from the pooled scores of this report)"] for k in KEYS])
    for k in KEYS:
        shutil.copyfile(TASK / "study" / k / "case.json", DATA / f"case_{k}.json")
        shutil.copyfile(TASK / "study" / k / "root_decimal_verification.json", DATA / f"root_decimal_verification_{k}.json")
        shutil.copyfile(TASK / "study" / k / "runtime.json", DATA / f"runtime_{k}.json")
    for name in ("root_result_audit.json", "root_warm_timing.json", "root_parameter_verification.json", "root_moab_fc_reference.json", "root_report_ready.json", "config_paths.json"):
        shutil.copyfile(TASK / name, DATA / name)
    shutil.copyfile(CAND / "cases/subsurface_validation/jorn_defaults.json", DATA / "jorn_defaults.json")
    blk = blocks(R, B, S, G)
    to_docx(blk, T_public(T), DOCX)
    info = verify_docx(DOCX, blk)
    seal_check()                                                    # re-hash after consuming
    (HERE / "report.md").write_text(to_markdown(blk, T_public(T)), newline="\n")
    (HERE / "verification.md").write_text("# Verification record\n\nBefore building, `root_report_ready.json` was required to say PASS and every file of `root_run_seal.json` was re-hashed; they were re-hashed again after the document was built. From the sealed result files of the six cases the builder recomputed "
                                          "per-layer and case-pooled RAW, aligned and no-change scores and bias, all group scores (JORN, MOAB and combined, upper two layers and all three layers, every valid observation weighted individually; counts 26/10/36 and 39/14/53), the budgets, the zero gravity transfer and zero "
                                          "bottom drainage/surface return, the unchanged historical prepared inputs (equal to the stored historical ones), the preset, case and source identities (SYRUP 4e38a472.../180 and MAPLE fbc504bb.../412 before and after), the MOAB own field capacity, the "
                                          "Numba/NumPy agreement, the same-process public-state restart control and Root's 90-digit controls; all agree with Root's audit files. Static DOCX checks: zip integrity, one media file per figure, table count equals repeated-header count, footer page field, no placeholder text. "
                                          "No Office renderer was available, so rendered pagination and legibility were not checked (figures were viewed as PNG).\n\nThe restart control is a same-process public-constructor reconstruction, not a fresh-process test or a native checkpoint.\n\n```\n" + json.dumps(info, indent=1) + "\n```\n", newline="\n")
    man_files = {}
    for sub in (FIG, TAB, DATA):
        for p in sorted(sub.iterdir()):
            if p.is_file() and p.suffix in (".png", ".csv", ".json", ".md"):
                man_files[f"{sub.name}/{p.name}"] = sha(p)
    man = {"docx": {"file": DOCX.name, **info}, "files": man_files, "builder": sha(Path(__file__)), "report_md": sha(HERE / "report.md"), "verification_md": sha(HERE / "verification.md"), "seal_files_rechecked": len(R["seal"]),
           "excluded": "cache directories and ignored files (.ruff_cache, __pycache__, temp) are never listed"}
    (HERE / "artifact_manifest.json").write_text(json.dumps(man, indent=1) + "\n", newline="\n")
    print(json.dumps(info, indent=1))


def T_public(T):
    return {k: v for k, v in T.items() if not k.startswith("_")}


if __name__ == "__main__":
    main()
