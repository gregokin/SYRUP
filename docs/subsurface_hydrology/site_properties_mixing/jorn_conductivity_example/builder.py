"""Build the JORN conductivity worked example (DOCX with Word OMML equations, markdown, numbers.json, provenance.json).

Pure arithmetic (Decimal, 50 digits, plus a float replay of the accepted recipe). No model import, no model run, no GPU, no Git.
Reads accepted sources only to hash them / locate line numbers and to cross-check numbers. Writes ONLY into this directory and the DOCX one level up.
"""

import hashlib
import json
import math
import re
import zipfile
from decimal import Decimal as D
from decimal import localcontext
from pathlib import Path
from xml.sax.saxutils import escape

from docx import Document
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from lxml import etree

HERE = Path(__file__).resolve().parent
SM = HERE.parent                       # .../site_properties_mixing
ROOT = SM.parents[2]                   # candidate root (docs_candidate)
DOCX = SM / "SYRUP_JORN_conductivity_worked_example_20261117.docx"
TASK = ROOT.parent                     # task directory
PDF = Path("/home/okin/SYRUP/agent_handoffs/tasks/neon_texture_failure_audit_20261117/saxton_rawls_2006.pdf")
PDF_SHA = "24038366b95d508de140786b5b71fb634164dc9f89d1de158944c445c77186e5"
ECOHYD_SRC = Path("/home/okin/SYRUP/reference_models/ecohyd/github_baldauf_2023/source/model/src/WaterCell.cpp")
MNS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
WNS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def near(a, b, tol=1e-9):
    assert abs(float(a) - float(b)) <= tol * max(1.0, abs(float(b))), (a, b)


# ---------------------------------------------------------------- arithmetic
def soil(S, C, OM, rho):
    """Saxton-Rawls 2006 Eq 1,2,3,5,6,7,8,9,15,16,18 with the SYRUP effective-density rule; rho in g/cm3, OM in percent."""
    with localcontext() as ctx:
        ctx.prec = 50
        S, C, OM, rho = D(S), D(C), D(OM), D(rho)
        terms = {}
        for key, k in (("eq1", (-D("0.024"), D("0.487"), D("0.006"), D("0.005"), -D("0.013"), D("0.068"), D("0.031"))),
                       ("eq2", (-D("0.251"), D("0.195"), D("0.011"), D("0.006"), -D("0.027"), D("0.452"), D("0.299"))),
                       ("eq3", (D("0.278"), D("0.034"), D("0.022"), -D("0.018"), -D("0.027"), -D("0.584"), D("0.078")))):
            vars_ = (S, C, OM, S * OM, C * OM, S * C, D(1))
            terms[key] = [(c, v, c * v) for c, v in zip(k, vars_)]
        t1500 = sum(t[2] for t in terms["eq1"])
        t33 = sum(t[2] for t in terms["eq2"])
        tg = sum(t[2] for t in terms["eq3"])
        r = {"S": S, "C": C, "OM": OM, "rho_raw": rho, "terms": terms, "t1500": t1500, "t33": t33, "tg": tg}
        r["theta1500"] = D("1.14") * t1500 - D("0.02")
        r["theta33N"] = t33 + D("1.283") * t33 * t33 - D("0.374") * t33 - D("0.015")
        r["g"] = D("1.636") * tg - D("0.107")
        r["thetaSN"] = r["theta33N"] + r["g"] - D("0.097") * S + D("0.043")
        r["rhoN"] = (1 - r["thetaSN"]) * D("2.65")
        r["rho_eff"] = max(rho, D(1), D("0.901") * r["rhoN"])
        r["DF"] = r["rho_eff"] / r["rhoN"]
        r["thetaSDF"] = 1 - r["rho_eff"] / D("2.65")
        r["theta33DF"] = r["theta33N"] - D("0.2") * (r["thetaSN"] - r["thetaSDF"])
        r["gap"] = r["thetaSDF"] - r["theta33DF"]
        r["lnratio"] = (D(1500) / D(33)).ln()
        r["lnth"] = (r["theta33DF"] / r["theta1500"]).ln()
        r["B"] = r["lnratio"] / r["lnth"]
        r["lam"] = 1 / r["B"]
        r["expo"] = 3 - r["lam"]
        r["Ks_mm_h"] = D(1930) * (r["gap"].ln() * r["expo"]).exp()
        r["Ks_m_s"] = r["Ks_mm_h"] / D(3600000)
        return r


def float_replay(S, C, OM, rho):
    """Float replay in the accepted code's operation order (soil_texture_properties.reference_block)."""
    p = lambda c: c[0] * S + c[1] * C + c[2] * OM + c[3] * S * OM + c[4] * C * OM + c[5] * S * C + c[6]
    t = p((-0.024, 0.487, 0.006, 0.005, -0.013, 0.068, 0.031)); wp = t + 0.14 * t - 0.02
    t = p((-0.251, 0.195, 0.011, 0.006, -0.027, 0.452, 0.299)); fcn = t + 1.283 * t * t - 0.374 * t - 0.015
    t = p((0.278, 0.034, 0.022, -0.018, -0.027, -0.584, 0.078)); g = t + 0.636 * t - 0.107
    satn = fcn + g + -0.097 * S + 0.043
    rhon = (1.0 - satn) * 2.65
    rho_eff = max(rho, 1.0, 0.901 * rhon)
    sat = 1.0 - rho_eff / 2.65
    fc = fcn - 0.2 * (satn - sat)
    gap = sat - fc
    B = (math.log(1500.0) - math.log(33.0)) / (math.log(fc) - math.log(wp))
    return {"theta1500": wp, "theta33": fc, "thetaS": sat, "rhoN": rhon, "DF": rho_eff / rhon, "Ks_mm_h": 1930.0 * gap ** (3.0 - 1.0 / B)}


def derive():
    R = {}
    with localcontext() as ctx:
        ctx.prec = 50
        R["A"] = A = soil("0.867", "0.071", "0.3448", "1.4196")
        wA = D("0.8") * D("1.4196") / (D("0.8") * D("1.4196") + D("0.2") * D("0.9035"))
        mix = lambda a, b: wA * D(a) + (1 - wA) * D(b)
        R["wA"] = wA
        R["mixed"] = {"S": mix("0.867", "0.837"), "Si": mix("0.062", "0.079"), "C": mix("0.071", "0.084"), "OM": mix("0.3448", "0.1724")}
        R["rho2_raw"] = D("0.8") * D("1.4196") + D("0.2") * D("0.9035")
        M = R["mixed"]
        R["L2"] = L2 = soil(M["S"], M["C"], M["OM"], R["rho2_raw"])
        R["f"] = f = (1 + D(30) / D(26)) ** D("-1.5")
        R["Ku_ratio"] = 1 / (1 + D(30) / D(26)) ** 3
        R["K1"], R["K2"] = f * A["Ks_mm_h"], f * L2["Ks_mm_h"]
        w1 = w2 = D("0.5")
        R["H"] = 1 / (w1 / R["K1"] + w2 / R["K2"])
        R["G"] = (w1 * R["K1"].ln() + w2 * R["K2"].ln()).exp()
        k1, k2 = D("38.2"), D("3.82")                    # hypothetical contrast, arithmetic illustration only
        R["hyp"] = {"K1": k1, "K2": k2, "H": 1 / (D("0.5") / k1 + D("0.5") / k2), "G": (D("0.5") * k1.ln() + D("0.5") * k2.ln()).exp()}
    return R


def checks(R):
    """Assert the numbers given in the assignment and the accepted probe/control files."""
    A, L2 = R["A"], R["L2"]
    near(A["theta1500"], "0.039508152304", 1e-11); near(A["theta33DF"], "0.09746408113338648", 1e-12)
    near(A["thetaSDF"], "0.4643018867924528", 1e-12); near(A["rhoN"], "1.5698170962598574", 1e-12)
    near(A["DF"], "0.9043091729490303", 1e-12); near(A["Ks_mm_h"], "120.78651558947541", 1e-11)
    near(R["f"], "0.31635738822500753", 1e-14); near(L2["Ks_mm_h"], "119.70776661032677", 1e-11)
    near(L2["rho_eff"], "1.4172961608026395", 1e-12); near(R["K1"], "38.2117", 1e-6); near(R["K2"], "37.8704", 1e-5)
    near(R["H"], "38.0403061077", 1e-9); near(R["G"], "38.0406888019", 1e-9)
    near(A["g"], "0.3625882482976", 1e-12)
    assert abs(A["g"] - (A["thetaSDF"] - A["theta33DF"])) > D("0.001")        # g is NOT final thetaS - final theta33
    fa = float_replay(0.867, 0.071, 0.3448, 1.4196)
    for k, v in (("theta1500", "theta1500"), ("theta33", "theta33DF"), ("thetaS", "thetaSDF"), ("Ks_mm_h", "Ks_mm_h"), ("DF", "DF"), ("rhoN", "rhoN")):
        near(fa[k], A[v], 1e-12)
    pr = json.loads((SM / "data" / "root_relative_density_probe.json").read_text())
    rec = {r["name"]: r for r in pr["records"]}
    for nm, X in (("JORN_A", A), ("JORN_10_20_dry_mass", L2)):
        near(rec[nm]["Ks_mm_h"], X["Ks_mm_h"], 1e-11); near(rec[nm]["theta1500"], X["theta1500"], 1e-11)
        near(rec[nm]["theta33"], X["theta33DF"], 1e-11); near(rec[nm]["theta_saturation"], X["thetaSDF"], 1e-11)
    near(rec["JORN_A"]["raw_air_entry_kpa"], -0.6079086840620981)
    near(rec["JORN_A"]["FC6_with_zero_air_floor"], 0.39760410394534984)
    ctrl = TASK / "root_numeric_control.json"
    if ctrl.exists():                                                   # earlier independent Decimal control (read only)
        c = json.loads(ctrl.read_text())
        near(c["interface_harmonic_mm_h"], R["H"], 1e-12); near(c["interface_geometric_mm_h"], R["G"], 1e-12)
        near(c["layers"][1]["Ks_mm_h"], L2["Ks_mm_h"], 1e-12)
    return pr


# ---------------------------------------------------------------- math DSL: each node is (omml, text)
def R_(t, plain=False):
    rpr = '<m:rPr><m:sty m:val="p"/></m:rPr>' if plain else ""
    return f'<m:r>{rpr}<m:t xml:space="preserve">{escape(t)}</m:t></m:r>'


def X(*p):
    xs, ts = [], []
    for x in p:
        if isinstance(x, str):
            xs.append(R_(x)); ts.append(x)
        else:
            xs.append(x[0]); ts.append(x[1])
    return "".join(xs), "".join(ts)


def N(s):                                    # a number or operator run, upright
    return R_(s, True), s


def SUB(b, s):
    b, s = X(b), X(s)
    return f"<m:sSub><m:e>{b[0]}</m:e><m:sub>{s[0]}</m:sub></m:sSub>", f"{b[1]}_{s[1]}"


def SUP(b, s):
    b, s = X(b), X(s)
    return f"<m:sSup><m:e>{b[0]}</m:e><m:sup>{s[0]}</m:sup></m:sSup>", f"{b[1]}^({s[1]})"


def FR(n, d):
    n, d = X(n), X(d)
    return f"<m:f><m:num>{n[0]}</m:num><m:den>{d[0]}</m:den></m:f>", f"({n[1]})/({d[1]})"


def PAR(*x):
    x = X(*x)
    return f"<m:d><m:e>{x[0]}</m:e></m:d>", f"({x[1]})"


def LN(*x):
    x = X(*x)
    return ('<m:func><m:fName><m:r><m:rPr><m:sty m:val="p"/></m:rPr><m:t>ln</m:t></m:r></m:fName>' f"<m:e>{x[0]}</m:e></m:func>"), f"ln{x[1]}"


def EXP(*x):                                 # e^{...}
    return SUP("e", X(*x))


def SQRT(*x):
    x = X(*x)
    return f'<m:rad><m:radPr><m:degHide m:val="1"/></m:radPr><m:deg/><m:e>{x[0]}</m:e></m:rad>', f"sqrt({x[1]})"


AL = ('<m:r><m:t>&amp;</m:t></m:r>', "")          # alignment point inside an equation array


def EQA(*rows):
    """Aligned multi-line display equation (OMML equation array); rows are nodes, '&' marks the alignment column."""
    return "<m:eqArr>" + "".join(f"<m:e>{r[0]}</m:e>" for r in rows) + "</m:eqArr>", "\n    ".join(r[1] for r in rows)


th = lambda sub: SUB("θ", sub)


def omml(node, display=True):
    inner = f"<m:oMath>{node[0]}</m:oMath>"
    return f'<m:oMathPara xmlns:m="{MNS}">{inner}</m:oMathPara>' if display else inner.replace("<m:oMath>", f'<m:oMath xmlns:m="{MNS}">', 1)


# ---------------------------------------------------------------- number formatting
def fx(x, n):
    return f"{D(x):.{n}f}"


def sg(x, n):                                 # n significant digits
    x = D(x)
    e = x.adjusted()
    return f"{x:.{max(n - 1 - e, 0)}f}"


def sci(x, n=4):
    x = D(x)
    e = x.adjusted()
    return f"{x / D(10) ** e:.{n - 1}f}×10^{e}"


def sci_node(x, n=4):
    x = D(x)
    e = x.adjusted()
    return X(N(f"{x / D(10) ** e:.{n - 1}f}×"), SUP(N("10"), N(str(e))))


# ---------------------------------------------------------------- content
def content(R):
    A, L2 = R["A"], R["L2"]
    B = []
    add = B.append
    t1, t2 = A["t1500"], A["t33"]

    add(("title", "How SYRUP gets JORN's conductivities"))
    add(("sub", "Saxton–Rawls Ks, the EcoHyD within-layer adjustment, and layer-to-layer averaging, with one worked example · JORN 0–10 cm A horizon and 10–20 cm layer · documentation only, no model run · 2026-11-17"))

    add(("h1", "1. What this note answers"))
    add(("p", ("Three different conductivities appear in the SYRUP JORN work, and they are easy to mix up. They are computed in sequence, and each is a distinct physical quantity: "
              "(a) the saturated conductivity Ks of each layer, from the Saxton and Rawls (2006) pedotransfer equations; "
              "(b) a representative unsaturated conductivity K inside each layer, from the Tietjen/EcoHyD rule K = f·Ks; and "
              "(c) one conductivity per interface between adjacent layers, from a SYRUP experiment rule (weighted harmonic or geometric mean). "
              "Only (a) is Saxton–Rawls. Step (b) is Tietjen's, step (c) is neither.")))
    add(("bul", [
        f"**Surface Ks:** {sg(A['Ks_mm_h'], 8)} mm/h = {sci(A['Ks_m_s'], 5)} m/s for the A horizon (0–10 cm).",
        f"**Within-layer K:** f = {sg(R['f'], 8)}, so K₁ = {sg(R['K1'], 6)} mm/h (layer 1) and K₂ = {sg(R['K2'], 6)} mm/h (layer 2).",
        f"**Interface:** harmonic {sg(R['H'], 9)} mm/h, geometric {sg(R['G'], 9)} mm/h. They are nearly identical because the two neighbours are nearly identical.",
        "**Texture alone is not what this example uses.** The density-adjusted Ks below uses texture plus an organic-matter value plus a reported bulk density. Without a reported density one could use the regression's own normal density, but that is a different assumption and a different Ks. An interface value further needs the neighbouring layer's properties and the layer geometry."]))

    add(("h1", "2. Where the Saxton–Rawls equations are"))
    add(("p", ("Saxton KE, Rawls WJ (2006), Soil Water Characteristic Estimates by Texture and Organic Matter for Hydrologic Solutions, Soil Sci. Soc. Am. J. 70:1569–1578, DOI 10.2136/sssaj2005.0117. "
              "The equations used here are in the paper's Table 1 (printed page 1571, PDF page 3). The table below maps each equation number to its role and to the line in SYRUP's accepted recipe that evaluates it. "
              "Nothing in this note reproduces the paper's figures or text; equations are restated with their coefficients, as the code uses them.")))
    add(("tab", "map", "Table 1. Formula map. Eq numbers are the paper's. “Code” is soil_texture_properties.py in this candidate."))
    add(("p", "Two further paper equations are deliberately outside the Ks chain: Eq 4 (air-entry tension) and Eq 17 (moisture-dependent conductivity K(θ)). Section 6 explains why."))

    add(("h1", "3. Inputs for the JORN surface layer"))
    add(("p", ("The whole 0–10 cm layer is the A horizon. Texture (sand, silt, clay) and bulk density are the reported inputs. Organic matter is not a measurement: 0.3448 % is a provisional derived assumption, converted as 1.724 × estimated organic carbon (g/kg) / 10. No hydraulic property here is measured. OM enters the paper's equations as percent of dry mass, so the number 0.3448 is used (not 0.003448). "
              "Soil-moisture quantities (θ) are volume fractions in m³/m³. The code takes density in kg/m³ and divides by 1000 to the paper's g/cm³.")))
    add(("tab", "inputs", "Table 2. Inputs, JORN A horizon: texture and density as reported; OM a provisional derived assumption, not a measurement."))

    add(("h1", "4. Step A: retention anchors and density adjustment"))
    add(("h2", "4.1 Regression intermediates (Eq 1, 2, 3)"))
    add(("p", ("Each equation forms a first estimate t from the same seven terms (S, C, OM, their products S·OM, C·OM, S·C, and a constant); only the coefficients differ. The first estimate is then corrected to the final retention value. "
              "Table 3 shows every term for the JORN A horizon.")))
    add(("eq", EQA(X(SUB("t", "1500"), " =", AL, "−0.024 S + 0.487 C + 0.006 OM"), X(AL, "+ 0.005 S·OM − 0.013 C·OM + 0.068 S·C + 0.031")), "Eq 1a"))
    add(("eq", X(th("1500"), " = 1.14 ", SUB("t", "1500"), " − 0.02"), "Eq 1b"))
    add(("eq", EQA(X(SUB("t", "33"), " =", AL, "−0.251 S + 0.195 C + 0.011 OM"), X(AL, "+ 0.006 S·OM − 0.027 C·OM + 0.452 S·C + 0.299")), "Eq 2a"))
    add(("eq", X(SUB(th("33"), "N"), " = ", SUB("t", "33"), " + 1.283 ", SUP(SUB("t", "33"), "2"), " − 0.374 ", SUB("t", "33"), " − 0.015"), "Eq 2b"))
    add(("eq", EQA(X(SUB("t", "g"), " =", AL, "0.278 S + 0.034 C + 0.022 OM"), X(AL, "− 0.018 S·OM − 0.027 C·OM − 0.584 S·C + 0.078")), "Eq 3a"))
    add(("eq", X("g = 1.636 ", SUB("t", "g"), " − 0.107"), "Eq 3b"))
    add(("tab", "terms", "Table 3. Term-by-term evaluation (S = 0.867, C = 0.071, OM = 0.3448, S·OM = 0.298942, C·OM = 0.024481, S·C = 0.061557)."))
    add(("p", f"Eq 1b gives the permanent-wilting-point fraction θ₁₅₀₀ (water held at 1500 kPa). With t₁₅₀₀ = {fx(t1, 10)}:"))
    add(("eq", X(th("1500"), " = 1.14 × ", N(fx(t1, 10)), " − 0.02 = ", N(fx(A["theta1500"], 12))), "substituted"))
    add(("p", f"Eq 2b gives the normal-density 33 kPa (field-capacity-type) content θ₃₃ₙ. With t₃₃ = {fx(t2, 10)}:"))
    add(("eq", EQA(X(SUB(th("33"), "N"), " =", AL, N(fx(t2, 10)), " + 1.283 × ", SUP(N(fx(t2, 10)), "2")), X(AL, " − 0.374 × ", N(fx(t2, 10)), " − 0.015 = ", N(fx(A["theta33N"], 12)))), "substituted"))
    add(("p", (f"Eq 3b gives g, the paper's intermediate for the saturation-minus-33 kPa gap at normal density. Its value is g = {fx(A['g'], 13)}. "
              f"It is only an ingredient of Eq 5. It is not the final θS − θ₃₃: after the density adjustment below, the final difference is a different number: {fx(A['gap'], 12)} here, against g = {fx(A['g'], 12)}.")))
    add(("eq", X(SUB(th("S"), "N"), " = ", SUB(th("33"), "N"), " + g − 0.097 S + 0.043"), "Eq 5"))
    add(("eq", EQA(X(SUB(th("S"), "N"), " =", AL, N(fx(A["theta33N"], 12)), " + ", N(fx(A["g"], 13))), X(AL, " − 0.097 × 0.867 + 0.043 = ", N(fx(A["thetaSN"], 12)))), "substituted"))
    add(("p", "In words: Eq 5 gives the saturated water content a soil of this texture would have at the paper's “normal” density. Eq 6 converts it to that normal bulk density:"))
    add(("eq", X(SUB("ρ", "N"), " = ", PAR("1 − ", SUB(th("S"), "N")), " × 2.65"), "Eq 6"))
    add(("eq", X(SUB("ρ", "N"), " = (1 − ", N(fx(A["thetaSN"], 12)), ") × 2.65 = ", N(fx(A["rhoN"], 13)), " g/cm³"), "substituted"))
    add(("h2", "4.2 Density adjustment (Eq 7, 8, 9) and the effective density"))
    add(("p", ("The regression assumes normal density. The actual soil is looser or tighter, so saturation and field capacity are shifted. SYRUP uses the effective density, the largest of the raw density, 1000 kg/m³, and 0.901 times the normal density. "
              "This floor is applied after dry-mass mixing of the raw layer density, and the reported top-layer density is left unchanged when it is already above the floor. "
              f"For the A horizon, 0.901 × {fx(A['rhoN'], 4)} = {fx(D('0.901') * A['rhoN'], 4)} g/cm³ is below the reported 1.4196 g/cm³, so the reported value stands.")))
    add(("eq", X(FR(SUB("ρ", "DF"), SUB("ρ", "N")), " = DF"), "Eq 7 (reported only)"))
    add(("eq", X(SUB(th("S"), "DF"), " = 1 − ", FR(SUB("ρ", "DF"), "2.65")), "Eq 8"))
    add(("eq", X(SUB(th("33"), "DF"), " = ", SUB(th("33"), "N"), " − 0.2 ", PAR(SUB(th("S"), "N"), " − ", SUB(th("S"), "DF"))), "Eq 9"))
    add(("p", "In words: DF compares actual to normal density; Eq 8 turns the actual density into a porosity-based saturation (2.65 g/cm³ is particle density); Eq 9 lowers or raises field capacity by 20 % of the saturation shift."))
    add(("eq", X(SUB(th("S"), "DF"), " = 1 − ", FR(N(fx(A["rho_eff"], 4)), "2.65"), " = ", N(fx(A["thetaSDF"], 13))), "substituted"))
    add(("eq", EQA(X(SUB(th("33"), "DF"), " =", AL, N(fx(A["theta33N"], 12)), " − 0.2 × ("), X(AL, N(fx(A["thetaSN"], 12)), " − ", N(fx(A["thetaSDF"], 12)), ") = ", N(fx(A["theta33DF"], 14)))), "substituted"))
    add(("p", (f"The density factor is DF = {fx(A['DF'], 13)} (within the accepted 0.9–1.3 range). The retention values that feed conductivity are therefore θ₁₅₀₀ = {fx(A['theta1500'], 12)}, "
              f"θ₃₃,DF = {fx(A['theta33DF'], 14)}, θS,DF = {fx(A['thetaSDF'], 13)}.")))

    add(("h1", "5. Step B: the saturated conductivity Ks"))
    add(("p", "The paper fits a power-law retention curve through two anchors: 1500 kPa and 33 kPa. Eq 15 gives its slope B from them, Eq 18 inverts it to λ = 1/B, and Eq 16 turns the wet-end gap into Ks in mm/h."))
    add(("eq", X("B = ", FR(X(LN("1500"), " − ", LN("33")), X(LN(SUB(th("33"), "DF")), " − ", LN(th("1500"))))), "Eq 15"))
    add(("eq", X("λ = ", FR("1", "B")), "Eq 18"))
    add(("eq", X(SUB("K", "s"), " = 1930 ", SUP(PAR(SUB(th("S"), "DF"), " − ", SUB(th("33"), "DF")), X("3 − λ")), " (mm/h)"), "Eq 16"))
    add(("p", ("In words: a wider gap between saturation and 33 kPa content means more water-filled pore space that drains freely, so Ks rises; because the gap is below 1, a larger exponent 3 − λ gives a smaller Ks. "
              "The 33 kPa anchor is the one the paper uses here; SYRUP's separate 6 kPa field capacity is not in this formula.")))
    add(("eq", EQA(X("B =", AL, FR(X(N(sg(D(1500).ln(), 10)), " − ", N(sg(D(33).ln(), 10))), X(N(sg(A["theta33DF"].ln(), 10)), " − ", PAR(N(sg(A["theta1500"].ln(), 10)))))), X(AL, " = ", FR(N(fx(A["lnratio"], 8)), N(fx(A["lnth"], 8))), " = ", N(fx(A["B"], 8)))), "substituted"))
    add(("eq", X("λ = ", FR("1", N(fx(A["B"], 8))), " = ", N(fx(A["lam"], 8)), ";  3 − λ = ", N(fx(A["expo"], 8))), "substituted"))
    add(("eq", EQA(X(SUB("K", "s"), " =", AL, "1930 × ", SUP(PAR(N(fx(A["thetaSDF"], 9)), " − ", N(fx(A["theta33DF"], 9))), N(fx(A["expo"], 6)))), X(AL, "= 1930 × ", SUP(N(fx(A["gap"], 9)), N(fx(A["expo"], 6))), " = ", N(fx(A["Ks_mm_h"], 8)), " mm/h")), "substituted"))
    add(("p", "SYRUP stores conductivity in m/s. Numerically, divide the value in mm/h by 3 600 000 to get m/s (one mm/h is 10⁻³ m per 3600 s); the subscripts name the unit of each number."))
    add(("eq", X(SUB("K", "s,m/s"), " = ", FR(SUB("K", "s,mm/h"), "3 600 000"), " = ", FR(N(fx(A["Ks_mm_h"], 8)), "3 600 000"), " = ", sci_node(A["Ks_m_s"], 5)), "unit conversion"))

    add(("h1", "6. What does not enter Ks, and how bounds are treated"))
    add(("bul", [
        f"**Field capacity at 6 kPa** (θ₆ = {fx(R['theta6'], 10)}) is a separate SYRUP quantity built from the air-entry estimate. It is not an input to Ks, which uses 33 kPa.",
        f"**Air entry.** The Eq 4 prediction for this soil is {fx(R['air_raw'], 9)} kPa, which is negative and physically meaningless. For the approved prescribed-surrogate case it is floored at 0 as an independent what-if. The floored value does not enter Ks.",
        "**Native strict provider is unchanged.** It still refuses a negative air-entry prediction. These JORN parameters are prescribed independently; they are not an automatic acceptance by the native provider, not measured hydraulics, and not a hidden override.",
        "**Eq 17**, the Saxton–Rawls moisture-dependent conductivity K(θ), is not used in this diagnostic, and nothing here asserts a full unsaturated conductivity curve or a dynamic K."]))

    add(("h1", "7. Step C: the Tietjen/EcoHyD within-layer adjustment"))
    add(("p", ("The next step is distinct from Saxton–Rawls. In the later EcoHyD model (WaterCell.cpp, which follows the conductivity discussion in Section 2.1 of Tietjen et al. 2009, DOI 10.1029/2007WR006589; code downloaded at commit 80b10e6, lines 549–552), "
              "the conductivity of a layer is the geometric mean of the saturated Ks and a representative unsaturated value Kᵤ, using Gardner's (1958) form with reference ratio 30/26 and exponent 3. "
              "This is a geometric adjustment between saturated and unsaturated conductivity within one layer. It is not the averaging of adjacent layers (Section 8).")))
    add(("eq", X(SUB("K", "u"), " = ", FR(SUB("K", "s"), SUP(PAR("1 + ", FR("30", "26")), "3"))), "EcoHyD"))
    add(("eq", X("K = ", SQRT(SUB("K", "s"), SUB("K", "u")), " = f ", SUB("K", "s"), ",   f = ", SUP(PAR("1 + ", FR("30", "26")), X("−3/2"))), "EcoHyD"))
    add(("p", f"Here f = (56/26)^(−1.5) = {fx(R['f'], 17)} (exact arithmetic; double precision prints 0.31635738822500753). The model's separate dimensionless multiplier d is not in this reported K. Applying f to the surface Ks:"))
    add(("eq", X(SUB("K", "1"), " = ", N(fx(R["f"], 10)), " × ", N(fx(A["Ks_mm_h"], 8)), " = ", N(fx(R["K1"], 8)), " mm/h"), "substituted"))

    add(("h1", "8. Step D: conductivity between adjacent layers"))
    add(("p", ("Redistribution between two layers needs one K at their interface. This rule is a SYRUP experiment extension. It is not a Saxton–Rawls formula and is not the original Tietjen heterogeneous-layer rule. "
              "The weights come from half-layer distances: with thicknesses hᵢ and hⱼ, wᵢ = hᵢ/(hᵢ + hⱼ). The physical motivation of the harmonic form is flow in series through layers, where resistances add (e.g. Hoffmann, Ripich and Ellett 2002, USGS WRI 01-4257, Eq 5, p. 19; DOI 10.3133/wri014257). "
              "The geometric form is the log-average alternative. Both apply to the within-layer K values of Step C.")))
    add(("eq", X(SUB("w", "i"), " = ", FR(SUB("h", "i"), X(SUB("h", "i"), " + ", SUB("h", "j"))), ",   ", SUB("w", "j"), " = ", FR(SUB("h", "j"), X(SUB("h", "i"), " + ", SUB("h", "j")))), "weights"))
    add(("eq", X(SUB("K", "H"), " = ", FR("1", X(FR(SUB("w", "i"), SUB("K", "i")), " + ", FR(SUB("w", "j"), SUB("K", "j"))))), "harmonic"))
    add(("eq", X(SUB("K", "G"), " = exp", PAR(SUB("w", "i"), " ln ", SUB("K", "i"), " + ", SUB("w", "j"), " ln ", SUB("K", "j"))), "geometric"))
    add(("p", "In words: the harmonic mean is dominated by the less conductive layer; the geometric mean is the log-midpoint. For equal weights they reduce to 2KᵢKⱼ/(Kᵢ + Kⱼ) and √(KᵢKⱼ), and always satisfy harmonic ≤ geometric."))
    add(("h2", "8.1 Building layer 2 (10–20 cm) from the A and Bw horizons"))
    add(("p", (f"Layer 2 holds 8 cm of A horizon and 2 cm of Bw (S 0.837, Si 0.079, C 0.084, OM 0.1724 %, raw density 0.9035 g/cm³). Texture and OM are averaged with dry-mass weights, not 80/20 by depth: "
              f"the A share is wA = 0.8 × 1.4196 / (0.8 × 1.4196 + 0.2 × 0.9035) = {fx(R['wA'], 10)}. The raw layer density is the depth-weighted 0.8 × 1.4196 + 0.2 × 0.9035 = {fx(R['rho2_raw'], 5)} g/cm³ ({fx(R['rho2_raw'] * 1000, 2)} kg/m³).")))
    add(("tab", "layer2", "Table 4. Layer 2 mixed properties and Ks (same Eq 1–16 chain as Section 4–5)."))
    add(("p", (f"The effective density rule acts here: 0.901 × {fx(L2['rhoN'], 13)} g/cm³ = {fx(L2['rho_eff'], 13)} g/cm³ = {fx(L2['rho_eff'], 13)} × 1000 = {fx(L2['rho_eff'] * 1000, 10)} kg/m³. This exceeds the mixed raw density {fx(R['rho2_raw'], 5)} g/cm³ ({fx(R['rho2_raw'] * 1000, 2)} kg/m³) and the 1000 kg/m³ floor, so the effective density is "
              f"{fx(L2['rho_eff'] * 1000, 10)} kg/m³. The resulting Ks₂ is {fx(L2['Ks_mm_h'], 8)} mm/h, and K₂ = f·Ks₂ = {fx(R['K2'], 8)} mm/h.")))
    add(("h2", "8.2 The interface"))
    add(("p", "Both layers are 10 cm thick, so wᵢ = wⱼ = 0.5."))
    add(("eq", EQA(X(SUB("K", "H"), " =", AL, FR("1", X(FR("0.5", N(fx(R["K1"], 6))), " + ", FR("0.5", N(fx(R["K2"], 6)))))), X(AL, "= ", N(fx(R["H"], 10)), " mm/h")), "substituted"))
    add(("eq", X(SUB("K", "G"), " = exp", PAR("0.5 ln ", N(fx(R["K1"], 6)), " + 0.5 ln ", N(fx(R["K2"], 6))), " = ", N(fx(R["G"], 10)), " mm/h"), "substituted"))
    hy = R["hyp"]
    add(("p", (f"The two results differ by only {fx(abs(R['G'] - R['H']), 6)} mm/h ({fx((R['G'] - R['H']) / R['H'] * 100, 5)} %) because the neighbours are alike. "
              f"The choice of mean matters only for contrast. As pure arithmetic, not a site property: 38.2 mm/h above 3.82 mm/h gives harmonic {fx(hy['H'], 3)} and geometric {fx(hy['G'], 3)} mm/h. "
              "An interface value cannot be derived from the surface texture alone: it needs the lower layer's properties and the layer geometry.")))

    add(("h1", "9. Note on the 0.18 wilting-point parameter at Tabernas"))
    add(("p", ("Tietjen et al. (2009, Section 3.1) tied that model's site thresholds to −1500 kPa. The later Tabernas wilting value 0.18 is explicitly labelled a calibration in Baldauf's dissertation (Table A4.3, printed p. 124; bounds 0.08–0.18 in Table A4.1, p. 123, described as literature and standard values for silty loam), "
              "so it should not be read as an established measured θ₁₅₀₀. It does not enter Ks. The parameter file's “loamy sand” header also conflicts with field literature describing predominantly silty soils; the header is not verified texture. "
              "Baldauf et al. (2023, Front. Microbiol., Sec. 2.3.3) calibrate the wilting point against soil moisture at 3 and 10 cm. This note does not fit soil parameters and makes no claim about the exact pressure behind 0.18.")))

    add(("h1", "10. Limits"))
    add(("bul", [
        "Arithmetic and documentation only. No model, GPU or soil-water simulation was run, and no production source, test, reference or raw data were changed.",
        "The Ks values are pedotransfer estimates, not measured conductivity. Saxton–Rawls is an empirical regression; its Ks has large uncertainty for individual soils.",
        "Admission issue: the bulk density and density factor DF are inside the accepted bounds after the declared effective-density floor, but the Eq 4 regression predicts a negative air-entry tension for this soil, so the full native provider refuses the curve. The explicit zero floor is a labelled case bound for this prescribed surrogate; it does not enter Ks and does not make the native provider accept the soil.",
        "The interface rule is an experiment choice; its adequacy for JORN is not established."]))

    add(("h1", "Appendix. Source identities"))
    add(("tab", "sources", "Table 5. Sources, lines and SHA-256. Paths are relative to the candidate root unless absolute."))
    add(("p", ("Saxton KE, Rawls WJ (2006). SSSAJ 70:1569–1578, DOI 10.2136/sssaj2005.0117; https://rashms.com/wp-content/uploads/2021/08/soil-characteristics-by-texture-and-organic-rawls.pdf. "
              "Tietjen B et al. (2009) DOI 10.1029/2007WR006589. Baldauf S (dissertation), https://refubium.fu-berlin.de/bitstream/handle/fub188/37307/Dissertation_SelinaBaldauf.pdf. "
              "Baldauf et al. (2023) DOI 10.3389/fmicb.2023.1179291. Hoffmann JP, Ripich MA, Ellett KM (2002) USGS WRI 01-4257, DOI 10.3133/wri014257.")))
    return B


def find_line(path, needle):
    for i, ln in enumerate(Path(path).read_text().splitlines(), 1):
        if needle in ln:
            return i
    raise AssertionError(needle)


def sources():
    stp = ROOT / "src/maple_syrup/soil_texture_properties.py"
    itf = ROOT / "benchmarks/subsurface_site_comparison/interfaces.py"
    rows = []
    L = lambda p, n: find_line(p, n)
    rows.append(("Saxton–Rawls 2006 PDF (Table 1, printed p.1571 = PDF p.3)", str(PDF), "hash verified; location per assignment", PDF_SHA))
    rows.append(("Coefficients Eq 1–4, constants", "src/maple_syrup/soil_texture_properties.py", f"L{L(stp, 'EQ1_REGRESSION =')}–{L(stp, 'KS_COEFF_MM_H, MM_H_PER_M_S')}", sha(stp)))
    rows.append(("Eq 1–6 normal regressions", "same", f"L{L(stp, '# normal-density estimates')}–{L(stp, 'rho_n = (1.0 - sat_n)')}", ""))
    rows.append(("Eq 7–9 density adjustment", "same", f"L{L(stp, '# density adjustment (Eq 7-10)')}–{L(stp, 'fc_d = fc_n - DF_AIR_COEFF')}", ""))
    rows.append(("Eq 15, 18, 16 Ks", "same", f"L{L(stp, 'B = LN_RATIO')}–{L(stp, 'ks_est = ')}", ""))
    rows.append(("Interface rule, EcoHyD factor", "benchmarks/subsurface_site_comparison/interfaces.py", f"L{L(itf, 'ECOHYD_FACTOR =')}, L{L(itf, 'def interface_weights')}–{L(itf, 'def local_conductivities') - 1}", sha(itf)))
    rows.append(("EcoHyD Ku, K_geom", str(ECOHYD_SRC), f"L{L(ECOHYD_SRC, 'Ku = (double)Ks')}–{L(ECOHYD_SRC, 'K_geom = sqrt')}", sha(ECOHYD_SRC)))
    pr = SM / "data/root_relative_density_probe.json"
    rows.append(("Prior probe values (cross-check)", "docs/subsurface_hydrology/site_properties_mixing/data/root_relative_density_probe.json", "records JORN_A, JORN_10_20_dry_mass", sha(pr)))
    rp = SM / "data/property_audit/root_probe.json"
    rows.append(("Property audit probe (cross-check)", "…/data/property_audit/root_probe.json", "", sha(rp)))
    return rows


def tables(R):
    A, L2, M = R["A"], R["L2"], R["mixed"]
    T = {}
    T["map"] = (["Eq", "Quantity", "Role in Ks chain", "Code"], [
        ["1", "θ₁₅₀₀ wilting (1500 kPa)", "Lower anchor", "EQ1_*"],
        ["2", "θ₃₃,N normal 33 kPa", "Upper anchor, via Eq 9", "EQ2_*"],
        ["3", "g (intermediate)", "Ingredient of Eq 5 only", "EQ3_*"],
        ["5", "θS,N normal saturation", "Gives Eq 6, Eq 9", "EQ3_SAT_TERMS"],
        ["6", "ρN normal density", "Reference for DF", "rho_n"],
        ["7", "DF = ρDF/ρN", "Reported, range check", "df"],
        ["8", "θS,DF density-adjusted", "Saturation in Eq 16", "sat_d"],
        ["9", "θ33,DF density-adjusted", "33 kPa anchor", "fc_d"],
        ["15", "B retention slope", "From 33 and 1500 kPa", "B"],
        ["18", "λ = 1/B", "Exponent 3 − λ", "lam"],
        ["16", "Ks (mm/h)", "Result", "ks_est"],
        ["4, 17", "Air entry; K(θ)", "Not in Ks chain", "—"]],)
    T["inputs"] = (["Quantity", "Value", "Unit / note"], [
        ["Sand S", "0.867", "mass fraction"], ["Silt Si", "0.062", "not used by Eq 1–16"], ["Clay C", "0.071", "mass fraction"],
        ["Organic matter OM", "0.3448", "derived, provisional: 1.724 × est. OC (g/kg) / 10; percent of dry mass"], ["Bulk density", "1419.6", "reported; kg/m³ = 1.4196 g/cm³"]],)
    T["terms"] = terms_table_global(R)
    T["layer2"] = (["Quantity", "A horizon (layer 1)", "Layer 2 (mixed)"], [
        ["Sand S", fx(A["S"], 6), fx(M["S"], 8)], ["Clay C", fx(A["C"], 6), fx(M["C"], 8)], ["OM (%)", fx(A["OM"], 6), fx(M["OM"], 8)],
        ["Raw density (kg/m³)", "1419.6", fx(R["rho2_raw"] * 1000, 2)], ["Normal density ρN (g/cm³)", fx(A["rhoN"], 8), fx(L2["rhoN"], 8)],
        ["Effective density (kg/m³)", fx(A["rho_eff"] * 1000, 4), fx(L2["rho_eff"] * 1000, 7)], ["θ₁₅₀₀", fx(A["theta1500"], 9), fx(L2["theta1500"], 9)],
        ["θ₃₃,DF", fx(A["theta33DF"], 9), fx(L2["theta33DF"], 9)], ["θS,DF", fx(A["thetaSDF"], 9), fx(L2["thetaSDF"], 9)],
        ["Ks (mm/h)", fx(A["Ks_mm_h"], 8), fx(L2["Ks_mm_h"], 8)], ["K = f·Ks (mm/h)", fx(R["K1"], 8), fx(R["K2"], 8)]],)
    T["sources"] = (["Item", "Path", "Lines / note", "SHA-256"], [[a, b, c, d[:16] + "…" + d[-6:] if d else "(same file)"] for a, b, c, d in sources()])
    return T


def terms_table_global(R):
    A = R["A"]
    names = ["S", "C", "OM", "S·OM", "C·OM", "S·C", "const."]
    rows = []
    for i, nme in enumerate(names):
        row = [nme, fx(A["terms"]["eq1"][i][1], 6)]
        for key in ("eq1", "eq2", "eq3"):
            c, _, p = A["terms"][key][i]
            row.append(f"{c:+.3f} → {fx(p, 6)}")
        rows.append(row)
    rows.append(["sum", "", "t₁₅₀₀ = " + fx(A["t1500"], 8), "t₃₃ = " + fx(A["t33"], 8), "t_g = " + fx(A["tg"], 8)])
    return ["Term", "Value", "Eq 1 coef → term", "Eq 2 coef → term", "Eq 3 coef → term"], rows


# ---------------------------------------------------------------- docx
def shade(cell, color):
    sh = OxmlElement("w:shd")
    sh.set(qn("w:val"), "clear"); sh.set(qn("w:color"), "auto"); sh.set(qn("w:fill"), color)
    cell._tc.get_or_add_tcPr().append(sh)


def add_field(par, instr):
    r = par.add_run()
    for t, txt in (("begin", None), (None, instr), ("end", None)):
        if t:
            e = OxmlElement("w:fldChar"); e.set(qn("w:fldCharType"), t)
        else:
            e = OxmlElement("w:instrText"); e.set(qn("xml:space"), "preserve"); e.text = txt
        r._r.append(e)


def runs(par, text, size=None):
    for i, part in enumerate(re.split(r"\*\*(.+?)\*\*", text)):
        if part:
            r = par.add_run(part); r.bold = bool(i % 2)
            if size:
                r.font.size = Pt(size)


def setup_styles(doc):
    st = doc.styles
    n = st["Normal"]
    n.font.name, n.font.size = "Calibri", Pt(10.5)
    n.paragraph_format.space_after = Pt(5); n.paragraph_format.line_spacing = 1.08
    for name, sz, col in (("Heading 1", 14, "1F3A5F"), ("Heading 2", 11.5, "2B5C8A")):
        h = st[name]
        h.font.name, h.font.size, h.font.bold = "Calibri", Pt(sz), True
        h.font.color.rgb = RGBColor.from_string(col)
        h.paragraph_format.space_before, h.paragraph_format.space_after = Pt(10), Pt(3)
        h.paragraph_format.keep_with_next = True
    for nm, base in (("JTitle", "Title"), ("JSub", "Subtitle"), ("JCaption", "Normal"), ("JTable", "Normal"), ("JEq", "Normal")):
        st.add_style(nm, WD_STYLE_TYPE.PARAGRAPH).base_style = st[base]
    t = st["JTitle"]; t.font.name, t.font.size, t.font.bold = "Calibri", Pt(21), True
    t.font.color.rgb = RGBColor.from_string("1F3A5F")
    st["JSub"].font.size = Pt(10); st["JSub"].font.italic = True
    c = st["JCaption"]; c.font.size, c.font.italic = Pt(9), True; c.paragraph_format.keep_together = True
    tt = st["JTable"]; tt.font.size = Pt(8.5); tt.paragraph_format.space_after = Pt(0); tt.paragraph_format.line_spacing = 1.0
    st["JEq"].paragraph_format.space_after = Pt(2); st["JEq"].paragraph_format.keep_together = True


def add_table(doc, hd, rows, caption, width=6.7):
    cp = doc.add_paragraph(style="JCaption")
    runs(cp, caption); cp.paragraph_format.keep_with_next = True
    t = doc.add_table(rows=1, cols=len(hd))
    t.style = "Table Grid"; t.alignment = WD_TABLE_ALIGNMENT.CENTER; t.autofit = False
    wts = [max(len(str(h)) * 0.6, min(max(len(str(r[i])) for r in rows), 40), 5) for i, h in enumerate(hd)]
    ws = [Inches(width * w / sum(wts)) for w in wts]
    for i, h in enumerate(hd):
        c = t.rows[0].cells[i]
        c.text = ""; c.paragraphs[0].style = "JTable"; c.paragraphs[0].add_run(str(h)).bold = True
        shade(c, "D9E2F3"); c.width = ws[i]
    th_ = OxmlElement("w:tblHeader"); th_.set(qn("w:val"), "true")
    t.rows[0]._tr.get_or_add_trPr().append(th_)
    for ri, row in enumerate(rows):
        cells = t.add_row().cells
        t.rows[-1]._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
        for i, v in enumerate(row):
            cells[i].text = ""; cells[i].paragraphs[0].style = "JTable"
            cells[i].paragraphs[0].add_run(str(v)); cells[i].width = ws[i]
            if ri % 2:
                shade(cells[i], "F3F6FA")
    doc.add_paragraph().paragraph_format.space_after = Pt(1)


def add_equation(doc, node, label):
    p = doc.add_paragraph(style="JEq")
    p._p.append(etree.fromstring(omml(node)))
    cp = doc.add_paragraph(style="JCaption")
    cp.add_run(f"[{label}]"); cp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    cp.paragraph_format.space_after = Pt(4)


def to_docx(B, T):
    doc = Document()
    setup_styles(doc)
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Inches(8.5), Inches(11)
    sec.left_margin = sec.right_margin = Inches(0.9)
    sec.top_margin = sec.bottom_margin = Inches(0.8)
    fp = sec.footer.paragraphs[0]; fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    fp.add_run("SYRUP JORN conductivity worked example · documentation of arithmetic, not field validation · page ").font.size = Pt(8)
    add_field(fp, "PAGE")
    for b in B:
        k = b[0]
        if k == "title": doc.add_paragraph(b[1], style="JTitle")
        elif k == "sub": doc.add_paragraph(b[1], style="JSub")
        elif k == "h1": doc.add_heading(b[1], level=1)
        elif k == "h2": doc.add_heading(b[1], level=2)
        elif k == "p": runs(doc.add_paragraph(), b[1])
        elif k == "bul":
            for x in b[1]:
                runs(doc.add_paragraph(style="List Bullet"), x)
        elif k == "tab": add_table(doc, *T[b[1]], b[2])
        elif k == "eq": add_equation(doc, b[1], b[2])
    doc.core_properties.title = "How SYRUP gets JORN's conductivities: Saxton-Rawls Ks, EcoHyD adjustment, layer-to-layer averaging"
    doc.core_properties.author = "SYRUP project"
    doc.save(DOCX)


def to_markdown(B, T):
    out = []
    for b in B:
        k = b[0]
        if k == "title": out.append(f"# {b[1]}\n")
        elif k == "sub": out.append(f"*{b[1]}*\n")
        elif k == "h1": out.append(f"## {b[1]}\n")
        elif k == "h2": out.append(f"### {b[1]}\n")
        elif k == "p": out.append(b[1] + "\n")
        elif k == "bul": out.append("\n".join(f"- {x}" for x in b[1]) + "\n")
        elif k == "eq": out.append(f"    {b[1][1]}    [{b[2]}]\n")
        elif k == "tab":
            hd, rows = T[b[1]]
            out.append(f"*{b[2]}*\n\n| " + " | ".join(hd) + " |\n|" + "---|" * len(hd) + "\n" + "\n".join("| " + " | ".join(str(c) for c in r) + " |" for r in rows) + "\n")
    (HERE / "explanation.md").write_text("\n".join(out))


def verify_docx():
    z = zipfile.ZipFile(DOCX)
    assert z.testzip() is None
    xml = z.read("word/document.xml")
    root = etree.fromstring(xml)                                     # well-formed
    for name in z.namelist():
        if name.endswith((".xml", ".rels")):
            etree.fromstring(z.read(name))
    ns = {"m": MNS, "w": WNS}
    txt = xml.decode()
    paras = root.xpath("//m:oMathPara", namespaces=ns)
    res = {"oMathPara": len(paras), "oMath": len(root.xpath("//m:oMath", namespaces=ns)), "fractions": len(root.xpath("//m:f", namespaces=ns)),
               "subscripts": len(root.xpath("//m:sSub", namespaces=ns)), "superscripts": len(root.xpath("//m:sSup", namespaces=ns)),
               "radicals": len(root.xpath("//m:rad", namespaces=ns)), "equation_arrays": len(root.xpath("//m:eqArr", namespaces=ns)), "tables": len(root.xpath("//w:tbl", namespaces=ns)),
               "raw_latex_or_dollar": bool(re.search(r"\\frac|\$\$|\\theta|\\\(", txt)), "words": len(re.sub(r"<[^>]+>", " ", txt).split()),
               "image_parts": [n for n in z.namelist() if n.startswith("word/media")]}
    assert res["oMathPara"] >= 25 and res["fractions"] >= 10 and not res["raw_latex_or_dollar"] and not res["image_parts"]
    return res


def main():
    R = derive()
    R["theta6"] = D(str(json.loads((SM / "data/root_relative_density_probe.json").read_text())["records"][0]["FC6_with_zero_air_floor"]))
    R["air_raw"] = D(str(json.loads((SM / "data/root_relative_density_probe.json").read_text())["records"][0]["raw_air_entry_kpa"]))
    checks(R)
    B, T = content(R), None
    T = tables(R)
    to_markdown(B, T)
    to_docx(B, T)
    ver = verify_docx()
    A, L2 = R["A"], R["L2"]
    nums = {k: str(v) for k, v in {
        "theta1500": A["theta1500"], "theta33DF": A["theta33DF"], "thetaSDF": A["thetaSDF"], "rhoN": A["rhoN"], "DF": A["DF"], "g": A["g"], "B": A["B"], "lam": A["lam"],
        "Ks_mm_h": A["Ks_mm_h"], "Ks_m_s": A["Ks_m_s"], "f": R["f"], "K1_mm_h": R["K1"], "A_mass_fraction_layer2": R["wA"], "layer2_raw_density_g_cm3": R["rho2_raw"],
        "layer2_effective_density_g_cm3": L2["rho_eff"], "layer2_theta1500": L2["theta1500"], "layer2_theta33DF": L2["theta33DF"], "layer2_thetaSDF": L2["thetaSDF"],
        "layer2_Ks_mm_h": L2["Ks_mm_h"], "K2_mm_h": R["K2"], "interface_harmonic_mm_h": R["H"], "interface_geometric_mm_h": R["G"]}.items()}
    nums = {"precision_digits": 50, "scope": "pure arithmetic; no model run", "values": nums, "assignment_values_asserted": True}
    (HERE / "numbers.json").write_text(json.dumps(nums, indent=2) + "\n")
    prov = {"baseline_head": "2fcc0b80624f857e17cb9b92d951be932e163857",
                "sources": [{"item": a, "path": b, "lines": c, "sha256": d} for a, b, c, d in sources()],
                "builder_sha256": sha(__file__), "explanation_md_sha256": sha(HERE / "explanation.md"), "numbers_json_sha256": sha(HERE / "numbers.json"),
                "docx_sha256": sha(DOCX), "docx_verification": ver,
                "pdf_check": {"sha256_verified": sha(PDF) == PDF_SHA, "text_reextracted": False, "note": "no PDF text tool available; Table 1 page location is per assignment"}}
    (HERE / "provenance.json").write_text(json.dumps(prov, indent=2) + "\n")
    print(json.dumps(ver, indent=1))


if __name__ == "__main__":
    main()
