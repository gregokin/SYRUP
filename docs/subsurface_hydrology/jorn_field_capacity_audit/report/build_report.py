"""Builds the JORN field-capacity worked audit report: independent recomputation, figure, CSV/JSON tables, editable Markdown, verification note, manifest and the Word document.

    PYTHONPATH=<tasks>/site_hydraulics_interface_comparison_20261117/report_tools MPLCONFIGDIR=/tmp/syrup-fc-report PYTHONDONTWRITEBYTECODE=1 /home/okin/MAPLE/.venv/bin/python -B build_report.py

DOCUMENTS ONLY. It reads Root's audit (root_audit.json, root_layers.csv), re-hashes every raw input / receipt / source file listed in root_audit.json['source_sha256'], recomputes everything ITSELF in 60-digit Decimal arithmetic from the raw NEON CSVs
(its own transcription of Saxton & Rawls 2006 Table 1; no model or SYRUP import) and asserts agreement with Root, the archived receipt and the actual original case_numba.json before any number is printed. No model run, network or GPU.
Declared inputs: JORN_FC_TASK (this task folder; default parent of this report folder) and JORN_FC_ROOT (the SYRUP checkout holding the hashed sources; default three levels above the task). Not portable without them.
"""
from __future__ import annotations

import csv
import glob
import hashlib
import json
import os
import re
import shutil
import zipfile
from decimal import Decimal as D
from decimal import getcontext
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
from docx.shared import Inches, Pt, RGBColor

getcontext().prec = 60
HERE = Path(__file__).resolve().parent
TASK = Path(os.environ.get("JORN_FC_TASK", str(HERE.parent)))
ROOT = Path(os.environ.get("JORN_FC_ROOT", str(TASK.parents[2])))
FIG, TAB, DATA = HERE / "figures", HERE / "tables", HERE / "data"
DOCX = HERE / "SYRUP_JORN_field_capacity_worked_audit_20261120.docx"
RAW = ROOT / "agent_handoffs/tasks/jorn_three_layer_20261102/downloads/DP1.00096.001/2013-03"
CONFIG = ROOT / "agent_handoffs/tasks/jorn_160_day_run_20261119/case_numba.json"
SIZES = ["sandVeryFineContent", "sandFineContent", "sandMediumContent", "sandCoarseContent", "sandVeryCoarseContent", "siltTotal", "clayTotal"]
COL = ["#B5651D", "#1F5FA8", "#7B3FA0"]
for d in (FIG, TAB, DATA):
    d.mkdir(exist_ok=True)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def csvw(path, header, rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


# ------------------------------------------------------------------------------------------------ identity
def identity():
    audit = json.loads((TASK / "root_audit.json").read_text())
    checked = {}
    for rel, want in audit["source_sha256"].items():
        p = ROOT / rel
        assert p.is_file(), f"missing {rel}"
        assert sha(p) == want, f"{rel} differs from root_audit.json"
        checked[rel] = want
    assert audit["status"] == "PASS" and audit["precision_decimal_digits"] == 80
    txt = ROOT / "agent_handoffs/tasks/neon_texture_failure_audit_20261117/probes/pdf_raw_text.txt"          # text extraction of the hashed Saxton-Rawls PDF
    body = txt.read_text()
    assert "0.63/0.052" in body and "0.36/0.063" in body and "0.78/2.94" in body, "Table 1 R2/Se evidence not found"
    checked[str(txt.relative_to(ROOT))] = sha(txt)
    for name in ("root_audit.json", "root_layers.csv", "root_audit.py"):
        shutil.copyfile(TASK / name, DATA / name)
        assert sha(DATA / name) == sha(TASK / name)
    return audit, checked


# ------------------------------------------------------------------------------------------------ independent recomputation (own transcription of S&R 2006 Table 1)
def raw_rows(kind):
    f = glob.glob(str(RAW / f"*mgp_{kind}.*.csv"))[0]
    with open(f, newline="") as fh:
        return {r["horizonName"]: r for r in csv.DictReader(fh)}


def sr(S, C, OM, rawrho):
    """Saxton & Rawls (2006) Table 1 Eq 2, 3, 4, 5, 6, 8, 9 and Eq 12 at 6 kPa; S, C fractions; OM percent; densities g/cm3. Steps returned for the worked example."""
    s = {}
    terms = [("-0.251 S", D("-0.251") * S), ("+0.195 C", D("0.195") * C), ("+0.011 OM", D("0.011") * OM), ("+0.006 S OM", D("0.006") * S * OM), ("-0.027 C OM", D("-0.027") * C * OM), ("+0.452 S C", D("0.452") * S * C), ("+0.299", D("0.299"))]
    s["t33_terms"] = terms
    s["t33"] = sum(v for _, v in terms)
    s["theta33N"] = s["t33"] + (D("1.283") * s["t33"] ** 2 - D("0.374") * s["t33"] - D("0.015"))
    gt = [("+0.278 S", D("0.278") * S), ("+0.034 C", D("0.034") * C), ("+0.022 OM", D("0.022") * OM), ("-0.018 S OM", D("-0.018") * S * OM), ("-0.027 C OM", D("-0.027") * C * OM), ("-0.584 S C", D("-0.584") * S * C), ("+0.078", D("0.078"))]
    s["tg_terms"] = gt
    s["tg"] = sum(v for _, v in gt)
    s["gap"] = s["tg"] + (D("0.636") * s["tg"] - D("0.107"))
    s["thetaSN"] = s["theta33N"] + s["gap"] - D("0.097") * S + D("0.043")
    s["rhoN"] = (1 - s["thetaSN"]) * D("2.65")
    s["rho_raw"] = rawrho
    s["rho"] = max(rawrho, D(1), D("0.901") * s["rhoN"])
    s["thetaS"] = 1 - s["rho"] / D("2.65")
    s["theta33"] = s["theta33N"] - D("0.2") * (s["thetaSN"] - s["thetaS"])
    at = [("-21.67 S", D("-21.67") * S), ("-27.93 C", D("-27.93") * C), ("-81.97 gap", D("-81.97") * s["gap"]), ("+71.12 S gap", D("71.12") * S * s["gap"]), ("+8.29 C gap", D("8.29") * C * s["gap"]), ("+14.05 S C", D("14.05") * S * C), ("+27.16", D("27.16"))]
    s["at_terms"] = at
    s["psi_et"] = sum(v for _, v in at)
    s["psi_raw"] = s["psi_et"] + (D("0.02") * s["psi_et"] ** 2 - D("0.113") * s["psi_et"] - D("0.70"))
    s["psi_used"] = max(D(0), s["psi_raw"])
    s["theta6"] = s["theta33"] + D(27) * (s["thetaS"] - s["theta33"]) / (D(33) - s["psi_used"])
    s["theta6_rawair"] = s["theta33"] + D(27) * (s["thetaS"] - s["theta33"]) / (D(33) - s["psi_raw"])
    s["theta6_normal_zero_air"] = s["theta33N"] + D(27) * (s["thetaSN"] - s["theta33N"]) / (D(33) - s["psi_used"])
    s["fraction"] = D(27) / (D(33) - s["psi_used"])
    return s


def recompute(audit):
    b, k = raw_rows("perbiogeosample"), raw_rows("perbulksample")
    rA, rB = D(k["A"]["bulkDensExclCoarseFrag"]), D(k["Bw"]["bulkDensExclCoarseFrag"])
    w = D("0.8") * rA / (D("0.8") * rA + D("0.2") * rB)
    layers = []
    for wt, rho in ((D(1), rA), (w, D("0.8") * rA + D("0.2") * rB), (D(0), rB)):
        v = {x: wt * D(b["A"][x]) + (1 - wt) * D(b["Bw"][x]) for x in SIZES + ["estimatedOC"]}
        sand = sum(v[x] for x in SIZES[:5])
        assert sand + v["siltTotal"] + v["clayTotal"] == 100
        om = v["estimatedOC"] * D("1.724") / 10
        layers.append({"w": wt, "psd": v, "sand": sand, "OC": v["estimatedOC"], "OM": om, "calc": sr(sand / 100, v["clayTotal"] / 100, om, rho)})
    # agreement with Root (80 digits), the archived receipt and the ACTUAL original case file
    cfg = json.loads(CONFIG.read_text())
    for L, rl in zip(layers, audit["layers"]):
        c = rl["calculations"]
        for mine, key in (("theta33", "theta33"), ("theta6", "theta6"), ("thetaS", "theta_sat"), ("rho", "density_used"), ("psi_raw", "air_raw"), ("theta33N", "theta33_normal"), ("gap", "gap_normal"), ("thetaSN", "saturation_normal"),
                          ("rhoN", "density_normal"), ("theta6_rawair", "theta6_raw_air_diagnostic_not_adopted"), ("theta6_normal_zero_air", "theta6_normal_density_zero_air_diagnostic_not_adopted")):
            assert abs(L["calc"][mine] - D(c[key])) < D("1e-45"), (mine, L["calc"][mine], c[key])
        assert abs(L["OM"] - D(rl["OM_percent"])) < D("1e-45") and abs(L["psd"]["sandVeryFineContent"] - D(rl["texture_percent"]["sandVeryFineContent"])) < D("1e-45")
    for i, L in enumerate(layers):
        assert abs(float(L["calc"]["theta6"]) - cfg["layers"]["theta_field_capacity"][i]) < 2e-13 and abs(float(L["calc"]["thetaS"]) - cfg["layers"]["theta_saturation"][i]) < 2e-13
    assert abs(float(w) - float(audit["middle_A_mass_weight"])) < 1e-15
    # published Table 3 sandy examples (OM 2.5 %, normal density): FC(33 kPa) about 10 % and 12 %
    t3 = {name: float(sr(D(S), D(C), D("2.5"), D("1.5"))["theta33N"]) * 100 for name, (S, C) in {"sand (S 88 %, C 5 %)": ("0.88", "0.05"), "loamy sand (S 80 %, C 5 %)": ("0.80", "0.05")}.items()}
    assert round(t3["sand (S 88 %, C 5 %)"]) == 10 and round(t3["loamy sand (S 80 %, C 5 %)"]) == 12
    return layers, w, cfg, t3, b, k


# ------------------------------------------------------------------------------------------------ formatting helpers
def pc(x, n=2):
    return f"{float(x) * 100:.{n}f}"


def g(x, n=4):
    return f"{float(x):.{n}f}"


def sg(x, n=6):
    return f"{float(x):.{n}g}"


# ------------------------------------------------------------------------------------------------ figure
def figure(layers):
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    psi = np.linspace(0, 33, 200)
    ax.axhspan(10, 15, color="#DDEEDD", alpha=0.7, lw=0, label="user's expected field-capacity range, 10-15 vol. % (illustrative, not a measurement)")
    names = ["Layer 1 (0-10 cm)", "Layer 2 (10-20 cm)", "Layer 3 (20-30 cm)"]
    for L, c, n in zip(layers, COL, names):
        t33, ts = float(L["calc"]["theta33"]) * 100, float(L["calc"]["thetaS"]) * 100
        ax.plot(psi, t33 + (33 - psi) * (ts - t33) / 33, color=c, lw=1.6, label=f"{n}: computed wet branch")
        ax.scatter([33, 6, 0], [t33, float(L["calc"]["theta6"]) * 100, ts], color=c, s=34, zorder=5, edgecolors="black", linewidths=0.6)
    ax.axvline(6, color="#555555", ls="--", lw=1)
    ax.axvline(33, color="#555555", ls=":", lw=1)
    ax.text(6.4, 22, "6 kPa: project\nfield-capacity pressure", fontsize=8, ha="right", va="center")
    ax.text(32.4, 22, "33 kPa: S&R\nfield capacity", fontsize=8, ha="left", va="center")
    ax.text(2.0, 49.0, "0 kPa: saturation", fontsize=8, ha="right", va="center")
    ax.set_xlim(-0.5, 34)
    ax.set_ylim(5, 51)
    ax.invert_xaxis()
    ax.set_xlabel("soil suction (kPa); suction +33 kPa = matric potential -33 kPa", fontsize=9)
    ax.set_ylabel("volumetric water content (vol. %)", fontsize=9)
    ax.set_title("COMPUTED Saxton-Rawls wet-end curves with air entry floored at 0 kPa (no measured 6 kPa value)", fontsize=9.5)
    ax.grid(alpha=0.25)
    ax.tick_params(labelsize=8)
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, fontsize=7.5, loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig(FIG / "fig1_wet_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ tables
def tables(layers, w, b, k, t3):
    T = {}
    A, Bw = b["A"], b["Bw"]
    names = [("sandVeryFineContent", "Very fine sand, 0.047-0.105 mm"), ("sandFineContent", "Fine sand, 0.105-0.25 mm"), ("sandMediumContent", "Medium sand, 0.25-0.5 mm"), ("sandCoarseContent", "Coarse sand, 0.5-1 mm"),
             ("sandVeryCoarseContent", "Very coarse sand, 1-2 mm")]
    rows = []
    for key, lab in names:
        rows.append([lab, A[key], Bw[key]] + [f"{float(L['psd'][key]):.3f}" for L in layers])
    rows.append(["Total sand, 0.047-2 mm", A["sandTotal"], Bw["sandTotal"]] + [f"{float(L['sand']):.3f}" for L in layers])
    rows.append(["Silt, 0.002-0.047 mm", A["siltTotal"], Bw["siltTotal"]] + [f"{float(L['psd']['siltTotal']):.3f}" for L in layers])
    rows.append(["Clay, below 0.002 mm", A["clayTotal"], Bw["clayTotal"]] + [f"{float(L['psd']['clayTotal']):.3f}" for L in layers])
    rows.append(["Sum of all fractions", "100.0", "100.0"] + ["100.000"] * 3)
    rows.append(["Very fine + fine sand", f"{float(D(A['sandVeryFineContent']) + D(A['sandFineContent'])):.1f}", f"{float(D(Bw['sandVeryFineContent']) + D(Bw['sandFineContent'])):.1f}"]
                + [f"{float(L['psd']['sandVeryFineContent'] + L['psd']['sandFineContent']):.3f}" for L in layers])
    T["psd_full"] = (["Size fraction (% of the <2 mm mass)", "Measured A, 0-18 cm", "Measured Bw, 18-46 cm", "Layer 1, 0-10 cm", "Layer 2, 10-20 cm (derived)", "Layer 3, 20-30 cm"], rows)      # CSV only
    T["psd"] = (["Size fraction (% of the fine-earth mass)", "Layer 1, 0-10 cm (= measured A sample)", "Layer 2, 10-20 cm (A-Bw mixture)", "Layer 3, 20-30 cm (= measured Bw sample)"], [[r[0], r[3], r[4], r[5]] for r in rows])
    oc = [L["OC"] for L in layers]
    T["inputs"] = (["Input", "Layer 1", "Layer 2 (derived)", "Layer 3"], [
        ["Source", "A horizon sample", "0.8627 A + 0.1373 Bw by mass", "Bw horizon sample"], ["A-horizon mass weight in the mix", "1", f"{float(w):.6f}", "0"],
        ["Estimated organic carbon (g/kg)", f"{float(oc[0]):.4f}", f"{float(oc[1]):.4f}", f"{float(oc[2]):.4f}"], ["Organic matter OM = 1.724 x OC / 10 (%)", f"{float(layers[0]['OM']):.4f}", f"{float(layers[1]['OM']):.4f}", f"{float(layers[2]['OM']):.4f}"],
        ["Raw bulk density (measured or mixed) (g/cm³)", "1.4196", f"{float(layers[1]['calc']['rho_raw']):.5f} (0.8 A + 0.2 Bw)", "0.9035"],
        ["Density used after floors (g/cm³)", g(layers[0]["calc"]["rho"]), g(layers[1]["calc"]["rho"]), g(layers[2]["calc"]["rho"])], ["Normal density from Eq. 6 (g/cm³)", g(layers[0]["calc"]["rhoN"]), g(layers[1]["calc"]["rhoN"]), g(layers[2]["calc"]["rhoN"])],
        ["Density sample metadata", f"ID 0-10 cm; centre {k['A']['bulkDensCenterDepth']} cm; bottom depth {k['A']['bulkDensBottomDepth']} cm", "mixture", f"ID 23-33 cm; centre {k['Bw']['bulkDensCenterDepth']} cm"]])
    L1 = layers[0]["calc"]
    S = layers[0]["sand"] / 100
    x = lambda terms: " ".join(f"{float(v):+.10f}" for _, v in terms)
    steps = [
        ["1", "Eq. 2 first solution", "-0.251 S + 0.195 C + 0.011 OM + 0.006 S OM - 0.027 C OM + 0.452 S C + 0.299", x(L1["t33_terms"]), sg(L1["t33"])],
        ["2", "Eq. 2 normal-density 33 kPa", "θ₃₃N = t + (1.283 t² - 0.374 t - 0.015)", f"{sg(L1['t33'])} + (1.283 x {float(L1['t33']) ** 2:.6f} - 0.374 x {sg(L1['t33'])} - 0.015)", sg(L1["theta33N"])],
        ["3", "Eq. 3 first solution", "0.278 S + 0.034 C + 0.022 OM - 0.018 S OM - 0.027 C OM - 0.584 S C + 0.078", x(L1["tg_terms"]), sg(L1["tg"])],
        ["4", "Eq. 3 gap θ(S-33)", "gap = t + (0.636 t - 0.107)", f"{sg(L1['tg'])} + (0.636 x {sg(L1['tg'])} - 0.107)", sg(L1["gap"])],
        ["5", "Eq. 5 normal saturation", "θSN = θ₃₃N + gap - 0.097 S + 0.043", f"{sg(L1['theta33N'])} + {sg(L1['gap'])} - 0.097 x {float(S):.4f} + 0.043", sg(L1["thetaSN"])],
        ["6", "Eq. 6 normal density", "ρN = (1 - θSN) x 2.65", f"(1 - {sg(L1['thetaSN'])}) x 2.65", f"{sg(L1['rhoN'])} g/cm³"],
        ["7", "Project density floors", "ρ = max(raw, 1.000, 0.901 ρN)", f"max(1.4196, 1.000, {float(D('0.901') * L1['rhoN']):.6f})", f"{sg(L1['rho'])} g/cm³"],
        ["8", "Eq. 8 saturation", "θS = 1 - ρ / 2.65", f"1 - {sg(L1['rho'])} / 2.65", sg(L1["thetaS"])],
        ["9", "Eq. 9 field capacity at 33 kPa", "θ₃₃ = θ₃₃N - 0.2 (θSN - θS)", f"{sg(L1['theta33N'])} - 0.2 x ({sg(L1['thetaSN'])} - {sg(L1['thetaS'])})", sg(L1["theta33"])],
        ["10", "Eq. 4 first solution", "-21.67 S - 27.93 C - 81.97 gap + 71.12 S gap + 8.29 C gap + 14.05 S C + 27.16", x(L1["at_terms"]), sg(L1["psi_et"])],
        ["11", "Eq. 4 air entry (kPa)", "ψe = t + (0.02 t² - 0.113 t - 0.70)", f"{sg(L1['psi_et'])} + (0.02 x {float(L1['psi_et']) ** 2:.6f} - 0.113 x {sg(L1['psi_et'])} - 0.70)", f"{sg(L1['psi_raw'])} kPa"],
        ["12", "Project floor on air entry", "ψe,used = max(0, ψe)", f"max(0, {sg(L1['psi_raw'])})", "0 kPa"],
        ["13", "Eq. 12 at 6 kPa (linear wet branch)", "θ₆ = θ₃₃ + (33 - 6)(θS - θ₃₃) / (33 - ψe,used)", f"{sg(L1['theta33'])} + 27 x ({sg(L1['thetaS'])} - {sg(L1['theta33'])}) / 33", sg(L1["theta6"])]]
    T["steps"] = (["Step", "Quantity (S&R Table 1)", "Formula", "Numbers for layer 1 (S, C fractions; OM %)", "Result"], steps)       # CSV only
    termlines = {"1": L1["t33_terms"], "3": L1["tg_terms"], "10": L1["at_terms"]}
    sb = []
    for st in steps:
        if st[0] in termlines:
            lines = [f"{lab} = {float(v):+.10f}" for lab, v in termlines[st[0]]] + [f"sum of the seven terms = {st[4]} (the printed terms are rounded to 10 decimals; the sum is the full-precision result)"]
        else:
            lines = [st[3]]
        sb.append((st[0], st[1], st[2], lines, st[4]))
    T["_steps"] = sb
    cs = [L["calc"] for L in layers]
    T["results"] = (["Quantity", "Layer 1", "Layer 2", "Layer 3"], [
        ["\u03b8\u2083\u2083 at normal density (vol. %)"] + [pc(c["theta33N"]) for c in cs], ["\u03b8\u2083\u2083 used, density-adjusted (vol. %)"] + [pc(c["theta33"]) for c in cs], ["\u03b8S saturation (vol. %)"] + [pc(c["thetaS"]) for c in cs],
        ["Air entry, raw (kPa)"] + [f"{float(c['psi_raw']):.3f}" for c in cs], ["Air entry used after the zero floor (kPa)"] + ["0"] * 3, ["\u03b8\u2086 = SYRUP field capacity (vol. %)"] + [pc(c["theta6"]) for c in cs],
        ["Share of the way from \u03b8\u2083\u2083 to \u03b8S"] + [f"{float(c['fraction']) * 100:.2f}%" for c in cs], ["Water added above \u03b8\u2083\u2083 (mm per 10 cm)"] + [f"{float((c['theta6'] - c['theta33']) * 100):.2f}" for c in cs]])
    T["sens"] = (["Quantity", "Layer 1", "Layer 2", "Layer 3"], [
        ["\u03b8\u2086 as used (vol. %)"] + [pc(c["theta6"]) for c in cs],
        ["Diagnostic A: \u03b8\u2086 with the raw NEGATIVE air entry (non-physical, not a proposed input)"] + [pc(c["theta6_rawair"]) for c in cs], ["Difference A - used (points)"] + [f"{float((c['theta6_rawair'] - c['theta6']) * 100):+.2f}" for c in cs],
        ["Raw bulk density (measured or mixed) (g/cm\u00b3)"] + [g(c["rho_raw"]) for c in cs], ["Density used (g/cm\u00b3)"] + [g(c["rho"]) for c in cs], ["Is a density floor active?"] + ["yes" if c["rho"] != c["rho_raw"] else "no" for c in cs],
        ["Diagnostic B: \u03b8\u2086 with the normal density substituted for the density used (air entry kept at 0)"] + [pc(c["theta6_normal_zero_air"]) for c in cs], ["Difference B - used (points)"] + [f"{float((c['theta6_normal_zero_air'] - c['theta6']) * 100):+.2f}" for c in cs]])
    c1 = layers[0]["calc"]
    T["storage"] = (["Hypothetical 10 cm layer 1 (not a model run)", "Volumetric water (vol. %)", "Stored water (mm)"], [["Water at S&R field capacity θ₃₃", pc(c1["theta33"]), f"{float(c1['theta33']) * 100:.2f}"],
                                                                                                         ["An example wetted layer", "15.00", "15.00"], ["Water at SYRUP field capacity θ₆", pc(c1["theta6"]), f"{float(c1['theta6']) * 100:.2f}"],
                                                                                                         ["Example minus θ₃₃ (would exceed 33 kPa field capacity)", "5.25", f"{15 - float(c1['theta33']) * 100:.2f}"], ["θ₆ minus example (still below SYRUP field capacity)", f"{float(c1['theta6']) * 100 - 15:.2f}", f"{float(c1['theta6']) * 100 - 15:.2f}"]])
    T["controls"] = (["Published-example check (OM 2.5 %, normal density)", "Recomputed θ₃₃ (vol. %)", "S&R Table 3 value (about)"], [[n, f"{v:.2f}", "10" if n.startswith("sand") else "12"] for n, v in t3.items()])
    return T


# ------------------------------------------------------------------------------------------------ text
REFS = [(1, "Saxton, K. E., & Rawls, W. J. (2006). Soil water characteristic estimates by texture and organic matter for hydrologic solutions. Soil Science Society of America Journal 70, 1569-1578. Table 1 (Eq. 2-12), Table 2 (field capacity = 33 kPa), Table 3 (sandy examples).", "https://doi.org/10.2136/sssaj2005.0117 ; https://www.atmos.illinois.edu/~sshu3/model/saxton2006.pdf"),
        (2, "Tietjen, B., Zehe, E., & Jeltsch, F. (2009). Simulating plant water availability in dry lands under climate change: a generic model of two soil layers. Water Resources Research 45, W01418. Section 3.1 (field capacity, similar-site field values, about -6 kPa).", "https://doi.org/10.1029/2007WR006589 ; https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2007WR006589"),
        (3, "NEON (RELEASE-2026). Soil physical and chemical properties, Megapit, DP1.00096.001 (Jornada pit, 2013-03: per-biogeochemistry-sample, per-horizon and per-bulk-sample tables).", "https://data.neonscience.org/data-products/DP1.00096.001/RELEASE-2026"),
        (4, "Earlier SYRUP reports cited by file name only: SYRUP_JORN_160day_rainfall_comparison_20261119.docx and SYRUP_JORN_160day_K_Ks_multipliers_2x_3x_4x_20261120.docx.", "")]


def blocks(layers, w, t3, audit):
    c = [L["calc"] for L in layers]
    added = "/".join(f"{float((x['theta6'] - x['theta33']) * 100):.1f}" for x in c)
    df3 = float(c[2]["rho_raw"] / c[2]["rhoN"])
    return [
        ("title", "Where does the 40% field capacity come from? A worked audit for the JORN soil layers"),
        ("sub", "Independent recomputation of the Saxton-Rawls inputs and arithmetic behind the field capacity used in the JORN 160-day runs. Arithmetic correctness is separated from whether the value is a credible site field capacity."),
        ("h1", "1. Short conclusion"),
        ("bul", ["**The arithmetic is correct.** I recomputed every number from the raw NEON files in 60-digit arithmetic with my own transcription of the equations. The results match Root's 80-digit audit, the archived receipt and the field capacities in the actual original run file to better than 1e-45 (to 2e-13 for the run file, which stores doubles).",
                 f"**Two different pressures are involved.** At the standard 33 kPa suction of Saxton and Rawls [1] the three layers hold {pc(c[0]['theta33'])}, {pc(c[1]['theta33'])} and {pc(c[2]['theta33'])} vol. % of water, about 10-11%; the first two values are slightly below 10%, close to the user's illustrative 10-15 vol. % field-capacity range (an expectation for sandy soil, not a measurement and not an exact match). SYRUP uses the water content at 6 kPa, following the field-capacity pressure of Tietjen et al. [2], and the Saxton-Rawls wet branch gives {pc(c[0]['theta6'])}, {pc(c[1]['theta6'])} and {pc(c[2]['theta6'])} vol. % there.",
                 f"**The 6 kPa value is our hybrid choice, not a measured or published site value.** Tietjen et al. take field capacity from similar-site field values, not from Saxton-Rawls equations. The Saxton-Rawls wet branch is a straight line between 33 kPa and saturation; with the air entry floored at zero, 6 kPa lies {float(c[0]['fraction']) * 100:.1f}% of the way to saturation. No measured 6 kPa retention value was used in, or found among, the field inputs I audited.",
                 "**The value is therefore arithmetically right but scientifically unsuitable without further qualification.** Particle-size data alone cannot validate the wet end of the curve. I do not defend 40% as a real field capacity for this sandy soil, and nothing here is adopted or rerun."]),
        ("h1", "2. Inputs actually used"),
        ("p", "The source is the NEON Jornada megapit (pit 1, sampled 2013-03) [3], two horizons: A, 0-18 cm, and Bw, 18-46 cm. They are spatial proxies for the 2024 sensor plot, not proven co-located measurements. Layer 1 uses the A sample, layer 3 the Bw sample, and layer 2 (10-20 cm) mixes 8 cm of A with 2 cm of Bw. The mixing is done on the raw measured properties, before any floors: the texture and carbon are weighted by dry mass (A weight " + f"{float(w):.4f}" + " = 0.8 × ρA / (0.8 × ρA + 0.2 × ρBw)) and the density is the thickness-weighted mean."),
        ("p", "Table 1 gives the complete particle-size distribution that NEON measured: five sand classes plus silt and clay, as percentages of the mass of the fine earth (material finer than 2 mm), not as volume percentages and not as the MAPLE grain classes. Fine and very fine sand dominate (52.2% in the A sample, 50.9% in Bw), but roughly 40% of the sand (34.5 of 86.7 points in A, 32.8 of 83.7 in Bw) is medium to very coarse. The Saxton-Rawls equations, however, use only the totals of sand, silt and clay; the sand sub-classes do not enter them."),
        ("tab", "psd", "Table 1. Particle-size distribution, % of fine-earth mass. Layers 1 and 3 equal the measured NEON A and Bw samples; layer 2 is a derived mixture (the raw A and Bw columns are in tables/psd_full.csv)."),
        ("p", f"Table 2 lists the other inputs. Organic matter is not measured: NEON reports an estimated organic carbon of 2.0 g/kg (A) and 1.0 g/kg (Bw), converted here with OM = 1.724 × OC / 10 per cent (a provisional conversion). The measured bulk densities are 1.4196 g/cm³ (A) and 0.9035 g/cm³ (Bw). The Bw value is far below the range the equations were built for (density factor {df3:.2f} against 0.9-1.3), so the study floors the density at max(raw, 1.000, 0.901 × normal density), applied after mixing; this raises layer 3 to 1.4346 g/cm³. The metadata of the A density sample is inconsistent (sample ID 0-10 cm and centre 5 cm, but a recorded bottom depth of 110 cm) and the Bw sample comes from 23-33 cm; I preserved both as recorded and did not correct them. The bulk density is NEON's value excluding coarse fragments."),
        ("tab", "inputs", "Table 2. Other inputs (carbon, organic matter, density) and their provenance."),
        ("h1", "3. Worked calculation for the top layer"),
        ("p", "The equations are Table 1 of Saxton and Rawls [1]; S and C are the sand and clay fractions (0-1) and OM is organic matter in per cent. Suction ψ is a positive number here: a suction of 33 kPa is a matric potential of -33 kPa. The steps below reproduce layer 1 (S = 0.867, C = 0.071, OM = 0.3448%, raw density 1.4196 g/cm³); every term of the three long sums is listed on its own line, rounded to 10 decimals, so the result can be rebuilt with a calculator. Displayed intermediates are rounded for reading; every later result is computed from the full-precision values and shown to six significant figures. Water contents are fractions (multiply by 100 for vol. %)."),
        ("steps", "Worked calculation for layer 1 (six significant figures; the same steps are in tables/steps.csv and full precision in tables/steps_layer1.csv)."),
        ("p", f"Reading the result. The regression for field capacity at 33 kPa gives {pc(c[0]['theta33'])} vol. %. Saxton and Rawls define field capacity as the water content at 33 kPa (their Table 2) [1]. Our step 13 evaluates the same printed wet branch at 6 kPa, an interpolation between the 33 kPa value and saturation. Because the study floors the air entry (a raw value of {float(c[0]['psi_raw']):.3f} kPa is negative and the provider refuses negative air entry), the linear branch runs from θ₃₃ at 33 kPa straight to saturation at 0 kPa, and the point at 6 kPa is 27/33 = 81.8% of the way up: θ₆ = {pc(c[0]['theta33'])} + 0.818 × ({pc(c[0]['thetaS'])} - {pc(c[0]['theta33'])}) = {pc(c[0]['theta6'])} vol. %."),
        ("h1", "4. Results for all three layers"),
        ("tab", "results", "Table 3. Computed hydraulic values (vol. % unless stated). θ₆ is the field capacity written into the original run file."),
        ("fig", "fig1_wet_curves.png", "Figure 1. The computed Saxton-Rawls wet-end curves for the three layers (lines) with the values at 33 kPa, 6 kPa and 0 kPa (dots). These are calculated, not observed. The green band is the user's expected field-capacity range of 10-15 vol. %, shown for orientation only (illustrative, not a measurement); the 33 kPa values lie just below or inside its lower edge."),
        ("p", "The check against published numbers is reassuring about the arithmetic but not about the 6 kPa choice: recomputing the published sandy examples of Saxton and Rawls (their Table 3, OM 2.5%, normal density) with my code gives a 33 kPa field capacity of " + f"{t3['sand (S 88 %, C 5 %)']:.2f} and {t3['loamy sand (S 80 %, C 5 %)']:.2f}" + " vol. %, in line with their listed values of about 10 and 12. The pedotransfer regression itself is uncertain: the paper's Table 1 lists R\u00b2 0.63 and a standard error of 0.052 for the 33 kPa water content (read from the text extraction of the local PDF copy), a large error for a value near 0.10."),
        ("h1", "5. Why the value is so high and what it implies"),
        ("bul", [f"**Pressure and curve shape dominate.** The step from 33 kPa to 6 kPa is an interpolation along the printed Eq. 12 wet branch, a straight line between the 33 kPa value and saturation, and it adds {added} mm (layers 1/2/3) of water per 10 cm layer above the 33 kPa value. Saxton and Rawls themselves describe this linear branch as a practical substitute for the often poorly represented low-tension range, and their power-law Eq. 11 applies only from 33 to 1500 kPa; extending Eq. 11 below 33 kPa would be an extrapolation, which I did not do, and I do not offer it as a validated replacement.",
                 "**Particle size alone cannot validate the wet end.** The size classes constrain the texture class but do not by themselves fix the water held at 6 kPa. No measured 6 kPa retention value was used in, or found among, the field inputs I audited (the NEON saturated-extract water fields are mass-based extract measurements, not volumetric water at 6 kPa).",
                 "**The air-entry and density choices are secondary.** Table 4 shows that the zero floor on the air entry changes θ₆ by about 0.5 points (the comparison uses a negative, non-physical air entry and is not a proposed input). Substituting the normal density for the density used, with the air entry kept at zero, lowers θ₆ by about 5 points and still leaves about 35% (layer 1 uses its measured density, so no floor is active there). The curve shape and the pressure choice explain almost all of the difference from 10 vol. %."]),
        ("tab", "sens", "Table 4. Sensitivity of θ₆ to the air-entry and density choices (diagnostics only, not adopted)."),
        ("p", "**Implications for the model runs.** The original 160-day case file contains exactly these field capacities, and the conductivity multipliers (1.2, 2, 3, 4) left them unchanged. In the native model gravity drainage acts only on water above field capacity, so it could not start while the modelled water content stayed far below about 39.5-39.9 vol. %, even at four times the conductivity; water still moved between layers by the moisture-gradient exchange, so the soil was not static. Table 5 is an explicitly hypothetical comparison, not a run: a 10 cm layer holding 15 vol. % (15 mm) would be above the 33 kPa field capacity by about 5 mm, but far below the SYRUP value. If the field capacity were changed later, the diffusion coefficient and evaporation-stress values that were fitted under the present one would have to be reconsidered; the old results should be kept, not rewritten."),
        ("tab", "storage", "Table 5. Hypothetical storage comparison for layer 1 (illustration only)."),
        ("h1", "6. Suggested later choices (not adopted)"),
        ("bul", ["Offer the 33 kPa value, which is the Saxton-Rawls definition and lies close to the user's expected range, as a transparent sensitivity or control option for the field capacity of the pedotransfer-derived layers. It is not validated at this site and not shown to be right at 6 kPa or for sandy soil in general; the existing d and stress fit is conditional on the present value and would have to be redone under it.",
                 "Keep a 6 kPa pressure only together with a qualified site retention curve or site-specific field capacity values (as Tietjen et al. use), and state that source explicitly; a qualified site curve or a field drainage assessment can then choose the operational field capacity.",
                 "Estimate a site field capacity from the sensors themselves (for example the water content where drainage becomes slow after a wetting event) and label it as a site estimate, not a measurement of a pressure.",
                 "Treat the density and air-entry choices as separate, smaller questions; they do not explain the high value. Nothing here is adopted."]),
        ("h1", "7. Reproducibility and limits"),
        ("p", "Every raw NEON file, the archived receipt, the original run file and the Saxton-Rawls PDF listed in root_audit.json were re-hashed before use (tables/provenance.csv); Root's JSON/CSV copies are in data/ with their hashes in artifact_manifest.json. The builder recomputes all values itself and asserts agreement. Limits: the megapit is a 2013 proxy for 2024 sensors; organic carbon is estimated; the Tietjen paragraph reference and the PDF equation wording were checked by Root and by the PDF text, not re-derived here; no measured 6 kPa retention value was used or found in the audited inputs; and no model was run. No Word or LibreOffice renderer was available, so page layout and equation appearance in Word have not been visually checked."),
        ("h1", "References"), ("refs", REFS)]


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
    for nm, base in (("RepTitle", "Title"), ("RepSubtitle", "Subtitle"), ("RepCaption", "Normal"), ("RepTable", "Normal"), ("RepEq", "Normal")):
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
    eq = st["RepEq"]
    eq.font.name, eq.font.size = "Cambria Math", Pt(10)
    eq.paragraph_format.left_indent = Inches(0.35)
    eq.paragraph_format.space_after = Pt(1)
    tt = st["RepTable"]
    tt.font.size = Pt(8)
    tt.paragraph_format.space_after = Pt(0)
    tt.paragraph_format.line_spacing = 1.0


def add_table(doc, hd, rows, width_in, caption, wide_col=None):
    cp = doc.add_paragraph(style="RepCaption")
    runs(cp, caption)
    cp.paragraph_format.keep_with_next = True
    cp.paragraph_format.space_after = Pt(3)
    t = doc.add_table(rows=1, cols=len(hd))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    wts = [min(max(len(str(h)) * 0.35, max(len(str(r[i])) for r in rows), 5), 30 if wide_col != i else 46) for i, h in enumerate(hd)]
    ws = [Inches(width_in * x / sum(wts)) for x in wts]
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
    sec.left_margin = sec.right_margin = Inches(0.85)
    sec.top_margin = sec.bottom_margin = Inches(0.8)
    fp = sec.footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    fr = fp.add_run("SYRUP JORN field-capacity worked audit · arithmetic audit, not a site validation · page ")
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
        elif k == "steps":
            for n, title, formula, lines, result in T["_steps"]:
                hp = doc.add_paragraph()
                hp.paragraph_format.keep_with_next = True
                hp.paragraph_format.space_after = Pt(1)
                hp.paragraph_format.space_before = Pt(5)
                hp.add_run(f"Step {n}. {title}").bold = True
                fp2 = doc.add_paragraph(style="RepEq")
                fp2.paragraph_format.keep_with_next = True
                fp2.add_run(formula)
                for ln in lines:
                    lp = doc.add_paragraph(style="RepEq")
                    lp.paragraph_format.keep_with_next = True
                    lp.add_run(ln).font.size = Pt(9)
                rp = doc.add_paragraph()
                rp.paragraph_format.left_indent = Inches(0.35)
                rp.paragraph_format.space_after = Pt(2)
                rp.add_run("Result: ").bold = True
                rp.add_run(result)
            cp2 = doc.add_paragraph(style="RepCaption")
            runs(cp2, b[1])
        elif k == "tab":
            hd, rows = T[b[1]]
            add_table(doc, hd, rows, 6.8, b[2])
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
    doc.core_properties.title = "SYRUP JORN field-capacity worked audit"
    doc.core_properties.author = "SYRUP project"
    doc.save(path)


def verify_docx(path, B):
    z = zipfile.ZipFile(path)
    names = z.namelist()
    media = [n for n in names if n.startswith("word/media/")]
    xml = z.read("word/document.xml").decode()
    ntab, nhdr = xml.count("<w:tbl>"), xml.count("<w:tblHeader")
    nfig, ntabs = sum(1 for b in B if b[0] == "fig"), sum(1 for b in B if b[0] == "tab")
    assert sum(1 for b in B if b[0] == "steps") == 1
    pagef = "PAGE" in "".join(z.read(n).decode(errors="ignore") for n in names if "footer" in n)
    text = re.sub(r"<[^>]+>", " ", xml)
    assert z.testzip() is None and len(media) == nfig == 1 and ntab == ntabs == nhdr and pagef
    assert not re.search(r"TODO|TBD|XXX|placeholder|DRAFT|\bnan\b|\\\(|\$\$|\\frac", text)
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
        elif k == "steps":
            for n, title, formula, lines, result in T["_steps"]:
                out.append(f"**Step {n}. {title}**\n\n    {formula}\n" + "".join(f"    {ln}\n" for ln in lines) + f"\n*Result:* {result}\n")
            out.append(f"*{b[1]}*\n")
        elif k == "tab":
            hd, rows = T[b[1]]
            out.append(f"**{b[2]}**\n\n| " + " | ".join(hd) + " |\n|" + "---|" * len(hd) + "\n" + "\n".join("| " + " | ".join(str(c) for c in r) + " |" for r in rows) + "\n")
        elif k == "fig":
            out.append(f"![{b[2]}](figures/{b[1]})\n")
        elif k == "refs":
            out.append("\n".join(f"[{n}] {c} {u}".strip() for n, c, u in b[1]) + "\n")
    return "\n".join(out)


def main():
    audit, checked = identity()
    layers, w, _cfg, t3, b, k = recompute(audit)
    for p in list(TAB.glob("*.csv")):
        p.unlink()
    figure(layers)
    T = tables(layers, w, b, k, t3)
    for name, val in T.items():
        if not name.startswith("_"):
            csvw(TAB / f"{name}.csv", val[0], val[1])
    csvw(TAB / "steps_layer1.csv", ["quantity", "full_precision_value"], [[kk, str(vv)] for kk, vv in layers[0]["calc"].items() if not kk.endswith("_terms")])
    csvw(TAB / "provenance.csv", ["file_relative_to_SYRUP_root", "sha256"], [[r, h] for r, h in checked.items()])
    (DATA / "independent_recalculation.json").write_text(json.dumps({"precision_decimal_digits": 60, "agreement_with_root_abs_tolerance": "1e-45", "layers": [{"layer": i + 1, "A_mass_weight": str(L["w"]), "OM_percent": str(L["OM"]), "sand_percent": str(L["sand"]),
                                                                                                                                     "results": {kk: str(vv) for kk, vv in L["calc"].items() if not kk.endswith("_terms")}} for i, L in enumerate(layers)],
                                                                          "published_table3_controls_theta33N_percent": t3}, indent=1) + "\n", newline="\n")
    blk = blocks(layers, w, t3, audit)
    to_docx(blk, T, DOCX)
    info = verify_docx(DOCX, blk)
    (HERE / "report.md").write_text(to_markdown(blk, T), newline="\n")
    (HERE / "verification.md").write_text("# Verification record\n\nBefore building: every file in root_audit.json['source_sha256'] (raw NEON CSVs, archived receipt, native provider source, original case_numba.json, Saxton-Rawls PDF, published reports) was re-hashed; the builder then recomputed all three layers "
                                          "in 60-digit Decimal arithmetic from the raw NEON CSVs with its own transcription of Saxton-Rawls Table 1 and asserted agreement with Root's 80-digit values (1e-45), the layer mixture, the archived receipt and the original run file (2e-13), "
                                          "and recomputed the published Table 3 sandy examples. The quoted R2 0.63 / standard error 0.052 for the 33 kPa regression was read from the text extraction of the hashed Saxton-Rawls PDF (probes/pdf_raw_text.txt, Table 1 row for theta33: \"0.63/0.052\"; also 0.36/0.063 and 0.78/2.94 present) and asserted by the builder. Static DOCX checks: zip integrity, one media file, table count equals header-row count, footer page field, no placeholder or raw-LaTeX text. No Word/LibreOffice renderer was available: "
                                          "page layout and equation appearance were not checked visually.\n\n```\n" + json.dumps(info, indent=1) + "\n```\n", newline="\n")
    man = {"docx": {"file": DOCX.name, **info}, "figures": {p.name: sha(p) for p in sorted(FIG.glob("*.png"))}, "tables": {p.name: sha(p) for p in sorted(TAB.iterdir())}, "data": {p.name: sha(p) for p in sorted(DATA.iterdir())},
           "builder": sha(Path(__file__)), "report_md": sha(HERE / "report.md"), "origin_hashes_of_root_files": {n: sha(TASK / n) for n in ("root_audit.json", "root_layers.csv", "root_audit.py")}, "hashed_sources": checked}
    (HERE / "artifact_manifest.json").write_text(json.dumps(man, indent=1) + "\n", newline="\n")
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
