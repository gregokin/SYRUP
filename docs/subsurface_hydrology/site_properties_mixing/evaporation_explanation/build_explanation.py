"""Build the plain-language evaporation / d / E explanation (DOCX, markdown, figure, tables, provenance, verification).

Reads only saved results (../data, ../../observed_drydown/data) and source files; runs no model. Every illustrative number is
recomputed here and asserted against the values given in the task. Writes ONLY into this directory and the DOCX one level up.
"""

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
from docx import Document
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

HERE = Path(__file__).resolve().parent
SM = HERE.parent
DATA = SM / "data" / "actual_cpu"
OLD = SM.parent / "observed_drydown" / "data"
CAND = SM.parents[2]  # docs_candidate root (src/, benchmarks/ live here)
LIVE = Path("/home/okin/SYRUP")
TASK = CAND.parent  # task directory holding the Root check files (read only)
DOCX = SM / "SYRUP_evaporation_and_d_E_explained_20261117.docx"
FIGDIR, TABDIR = HERE / "figures", HERE / "tables"
FIGDIR.mkdir(exist_ok=True)
TABDIR.mkdir(exist_ok=True)

WINDOWS = {"JORN4_A": ("training", "JORN A"), "JORN4_B": ("heldout", "JORN B"), "MOAB2_C": ("heldout", "MOAB C")}
STRESS_OLD, STRESS_JORN, STRESS_MOAB = 0.18, 0.039508152304, 0.05900343186546672
K_OLD_MM_H = 2.182866


def J(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def near(a, b, tol=1e-9):
    assert abs(a - b) <= tol * max(1.0, abs(b)), (a, b)


def evap_fraction(theta, stress, residual):
    """Fraction of the (already E-scaled) potential demand allowed by moisture: 0 at/below residual, squared ratio below the scale, 1 at/above."""
    if theta <= residual:
        return 0.0
    if theta >= stress:
        return 1.0
    return (theta / stress) ** 2


def pair_transfer_mm(theta_dry, theta_wet, k_mm_h, d, h_mm, dt_h):
    """One day of SYRUP-style exact pair exchange with equal thicknesses (mm of water moved from the wetter to the drier layer)."""
    g = d * k_mm_h * h_mm / h_mm  # conductance (mm/h): donor thickness times K times d, divided by the centre distance
    rate = g * (1 / h_mm + 1 / h_mm)  # per hour
    equalising = (theta_wet - theta_dry) / (1 / h_mm + 1 / h_mm)
    return equalising * (1 - math.exp(-rate * dt_h)), rate * dt_h, equalising


# ------------------------------------------------------------------ numbers
def derive():
    R = {}
    # worked example (illustrative, not a field run)
    th, res, thick_mm, pet, E = 0.05, 0.01, 100.0, 6.0, 0.5
    avail = (th - res) * thick_mm
    ex = {"theta": th, "residual": res, "thickness_mm": thick_mm, "pet": pet, "E": E, "available_mm": avail}
    ex["pre_moisture_mm"] = pet * E
    ex["generic_ratio"] = th / STRESS_OLD
    ex["generic_fraction"] = evap_fraction(th, STRESS_OLD, res)
    ex["generic_actual_mm"] = min(ex["pre_moisture_mm"] * ex["generic_fraction"], avail)
    ex["jorn_fraction"] = evap_fraction(th, STRESS_JORN, res)
    ex["jorn_actual_mm"] = min(ex["pre_moisture_mm"] * ex["jorn_fraction"], avail)
    ex["e12_demand_mm"] = pet * 1.2
    ex["e12_jorn_actual_mm"] = min(ex["e12_demand_mm"] * 1.0, avail)
    ex["e12_generic_actual_mm"] = min(ex["e12_demand_mm"] * ex["generic_fraction"], avail)
    ex["half_scale"], ex["quarter_scale"] = 0.25, 1 / 16
    near(avail, 4.0)
    near(ex["generic_fraction"], 0.0771604938, 1e-8)
    near(ex["generic_actual_mm"], 0.231481481, 1e-8)
    assert ex["jorn_fraction"] == 1.0 and abs(ex["jorn_actual_mm"] - 3.0) < 1e-12
    near(ex["e12_demand_mm"], 7.2)
    near(ex["e12_jorn_actual_mm"], 4.0)
    R["example"] = ex

    # actual JORN-A first day from saved JSON
    new = J(DATA / "JORN4_A_training_harmonic_result.json")
    old = J(OLD / "JORN4_A_training_result.json")
    theta0 = new["plan"]["initial_layer_theta"][0]
    pet1 = new["plan"]["pet_mm_day"][0]
    near(theta0, 0.04759332)
    near(pet1, 6.139189215844968)
    a = {"theta0": theta0, "pet1": pet1, "pre_mm": pet1 * 0.5}
    a["old_fraction"] = evap_fraction(theta0, STRESS_OLD, 0.01)
    a["new_fraction"] = evap_fraction(theta0, STRESS_JORN, 0.01)
    near(a["old_fraction"], 0.06991123792044444)
    assert a["new_fraction"] == 1.0
    a["old_mm"] = a["pre_mm"] * a["old_fraction"]
    a["new_mm"] = a["pre_mm"]
    a["available_mm"] = (theta0 - 0.01) * 100
    near(new["runs"]["d0_E0.5_open"]["trajectory"]["evaporation_m"][0] * 1000, a["new_mm"], 1e-9)
    near(old["runs"]["d0_E0.5_open"]["trajectory"]["evaporation_m"][0] * 1000, a["old_mm"], 1e-9)
    a["new_total_mm"] = sum(new["runs"]["d0_E0.5_open"]["trajectory"]["evaporation_m"]) * 1000
    near(a["new_total_mm"], a["available_mm"], 1e-9)  # the whole stock above residual is removed
    a["new_final_top"] = new["runs"]["d0_E0.5_open"]["layer_theta"][-1][0]
    R["actual_first_day"] = a

    # controls and outcome table from saved trajectories
    rows = {}
    for w, (kind, short) in WINDOWS.items():
        n = J(DATA / f"{w}_{kind}_harmonic_result.json")
        o = J(OLD / f"{w}_{'training' if w == 'JORN4_A' else 'heldout'}_result.json")
        eff = J(DATA / f"{w}_property_effect.json")["harmonic"]["matched"]
        r = {}
        for key in ("d0_E0.5_open", "d0.05_E1.2_open"):
            r[key] = {
                "old_evap_mm": sum(o["runs"][key]["trajectory"]["evaporation_m"]) * 1000,
                "new_evap_mm": sum(n["runs"][key]["trajectory"]["evaporation_m"]) * 1000,
                "old_rmse": eff[key]["historical_overall_rmse"],
                "new_rmse": eff[key]["new_overall_rmse"],
                "drain": n["runs"][key]["cumulative"]["drainage_m3"],
                "perc": sum(sum(x) for x in n["runs"][key]["trajectory"]["percolation_m"]),
                "ret": n["runs"][key]["cumulative"]["surface_return_m3"],
            }
        r["days"] = len(n["plan"]["pet_mm_day"])
        r["pet_mean"] = sum(n["plan"]["pet_mm_day"]) / r["days"]
        rows[short] = r
    expect = {"JORN A": (1.5824183857851986, 3.759332), "JORN B": (0.4131385228566277, 2.359582), "MOAB C": (1.8411914254220925, 5.34090914970658)}
    for s, (eo, en) in expect.items():
        near(rows[s]["d0_E0.5_open"]["old_evap_mm"], eo, 1e-9)
        near(rows[s]["d0_E0.5_open"]["new_evap_mm"], en, 1e-9)
        for key in rows[s]:
            if isinstance(rows[s][key], dict):
                assert rows[s][key]["drain"] == 0.0 and rows[s][key]["perc"] == 0.0 and rows[s][key]["ret"] == 0.0
    R["rows"] = rows

    # interface conductivities (harmonic, from the sealed config) in mm/h
    ic = J(SM / "config_templates" / "interface_coefficients.json")
    R["interface_raw"] = ic

    # worked redistribution example, JORN interface 1-2 harmonic K
    kj = 38.0403
    ex2 = {"k_new": kj, "k_old": K_OLD_MM_H, "d": 0.05, "h": 100.0, "dt_h": 24.0, "dry": 0.04, "wet": 0.06}
    for tag, k in (("new", kj), ("old", K_OLD_MM_H)):
        t, kd, eq = pair_transfer_mm(ex2["dry"], ex2["wet"], k, ex2["d"], ex2["h"], ex2["dt_h"])
        ex2[f"{tag}_transfer"], ex2[f"{tag}_kdt"], ex2["equalising"] = t, kd, eq
    ex2["ratio"] = ex2["new_transfer"] / ex2["old_transfer"]
    near(ex2["equalising"], 1.0)
    R["pair"] = ex2

    # independent Root checks (read-only files): worked pair example and the six first-day cases
    rd = J(TASK / "root_d_example_check.json")
    near(rd["values"]["new"]["transfer_mm"], ex2["new_transfer"], 1e-12)
    near(rd["values"]["old"]["transfer_mm"], ex2["old_transfer"], 1e-12)
    near(rd["transfer_ratio"], ex2["ratio"], 1e-12)
    rn = J(TASK / "root_numeric_checks.json")["actual_first_day_and_window_controls"]
    first = {}
    for rec in rn:
        w = rec["window"]
        kind = WINDOWS[w][0]
        src = J(DATA / f"{w}_{kind}_harmonic_result.json") if rec["soil"] == "texture_derived" else J(OLD / f"{w}_{'training' if w == 'JORN4_A' else 'heldout'}_result.json")
        th0 = src["plan"]["initial_layer_theta"][0]
        pet = src["plan"]["pet_mm_day"][0]
        frac = evap_fraction(th0, rec["stress_scale"], 0.01)
        near(frac, rec["moisture_fraction"], 1e-12)
        got = src["runs"]["d0_E0.5_open"]["trajectory"]["evaporation_m"][0] * 1000
        near(got, rec["actual_first_day_mm"], 1e-9)
        near(pet * 0.5 * frac, rec["pre_supply_cap_mm"], 1e-9)
        first[f"{w}/{rec['soil']}"] = {"fraction": frac, "first_day_mm": got}
    R["first_day_six_cases"] = first
    return R


# ------------------------------------------------------------------ figure
def figure(R):
    th = [i / 1000 for i in range(261)]
    fig, ax = plt.subplots(figsize=(6.6, 3.7), dpi=150)
    for stress, lab, col in ((STRESS_OLD, "Earlier generic soil: scale 0.18", "#B0413E"), (STRESS_JORN, "JORN-derived: scale 0.0395", "#2B5C8A"), (STRESS_MOAB, "MOAB-derived: scale 0.0590", "#3A8F5C")):
        ax.plot(th, [100 * evap_fraction(t, stress, 0.01) for t in th], label=lab, color=col, lw=2)
    ax.axvline(0.01, color="grey", ls=":", lw=1)
    ax.text(0.012, 60, "residual 0.01:\nmultiplier is zero here\nand jumps just above it", fontsize=7.5, color="grey")
    ax.scatter([0.05], [100 * R["example"]["generic_fraction"]], color="#B0413E", zorder=5)
    ax.annotate("worked example: moisture 0.05\ngives 7.7 % (generic scale)", (0.05, 7.7), (0.085, 26), fontsize=8, arrowprops={"arrowstyle": "->", "color": "#B0413E"})
    ax.set_xlabel("Top-layer moisture (fraction of soil volume, m³ per m³; 0.18 = 18 %)")
    ax.set_ylabel("Moisture multiplier (%)")
    ax.set_xlim(0, 0.26)
    ax.set_ylim(0, 108)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_moisture_response.png")
    plt.close(fig)


# ------------------------------------------------------------------ original-2009 text (paraphrase, word-budgeted)
T09_PARA = (
    "From the publisher's full text of Tietjen et al. (2009) [1] (Sections 2.3 and 3.1, Table 2): the exchange constant d is 0.01 at all "
    "four sites, adjusted so the exchange matches the diffusion scale of Bai et al. (2007) and observed moisture series. A constant, "
    "a_ET, reduces evaporation for site effects such as soil crusts; it is normally 1 and was calibrated to 0.4 at Sde Boqer only. The "
    "authors propose estimating it by comparing measured evaporation from crusted and bare patches. The moisture where stress begins "
    "corresponds to about -1500 kPa, taken from similar nearby field sites."
)
# every other place where 2009-specific facts are attributed to that paper (counted in the same budget)
T09_MATRIX = ["0.01", "a_ET 1; 0.4 at one site (calibrated)", "Not given", "Moisture at about -1500 kPa", "See section 7", "See section 7"]
T09_OTHER = [
    "Nor is it the original Tietjen constant a_ET, an additional site or crust factor (section 7).",
    "Tietjen, B., Zehe, E. and Jeltsch, F. (2009). Simulating plant water availability in dry lands under climate change: a generic model of two soil layers. Water Resources Research 45, W01418. DOI 10.1029/2007WR006589.",
]
T09_WORDS = len(" ".join([T09_PARA, *T09_MATRIX, *T09_OTHER]).split())
assert T09_WORDS <= 200, T09_WORDS


# ------------------------------------------------------------------ content
def content(R):
    ex, a, pr = R["example"], R["actual_first_day"], R["pair"]
    B = []
    A = B.append
    A(("title", "Why the soil-property results differ: evaporation, moisture stress, and d and E"))
    A(("sub", "A plain-language companion to the JORN/MOAB interface-comparison report · 2026-11-17 · built from saved results; no new model run, no parameter or default changed"))

    A(("h1", "1. The short version"))
    A(("bul", [
        "**Potential evapotranspiration (PET) is the weather's demand for water, not the water actually lost.** It is estimated from the calendar date, the day's air temperatures (lowest, highest, mean) and the site's latitude. The soil can only give up what it holds.",
        "**Capital E is a plain multiplier with no units.** It scales the PET we hand to the soil. E = 1 keeps the PET, E = 0.5 halves it, E = 0 turns soil evaporation off. E is not a number of millimetres.",
        "**Lowercase d is a dimensionless tuning knob for how fast neighbouring soil layers trade water; it moves water between layers and does not remove it.** It is not a diffusivity in square metres per second, not a drainage rate, not an amount of water and not a sediment property.",
        "**The evaporation-stress scale is a different thing again: a moisture level.** It is a volume fraction, so 0.18 means 18 % of the soil volume is water, not 0.18 %. Residual moisture is the floor below which the model never removes water; it is not the stress scale.",
        "**The big difference between the earlier generic soil and the new site soils, for evaporation, is mostly the stress scale (0.18 versus about 0.04–0.06), not E or d.** A lower scale lets the same dry soil lose more water.",
    ]))

    A(("h1", "2. Terms that are easy to confuse"))
    A(("tab", "terms", "Table 1. What each setting is, in everyday terms."))
    A(("p", "A note on units. Moisture is always a fraction of soil volume (a decimal such as 0.05, which is 5 %). Water amounts are millimetres of water depth over the soil surface; a 10 cm layer holding a moisture of 0.05 contains 5 mm of water. PET is in millimetres per day. The test column is one square metre, so millimetres of depth and litres per square metre are the same number."))

    A(("h1", "3. How one day of soil evaporation is calculated"))
    A(("p", "In the bare, flat dry-down test the top soil layer is treated like this, in order. No plants, no rain, no slope, no wind:"))
    A(("bul", [
        "**Step 1 – start from the day's PET.** For example 6 mm. (If the PET is zero, nothing is demanded whatever E is.)",
        "**Step 2 – multiply by E.** This is the demand before the soil has any say. With E = 0.5 and PET 6 mm the demand is 3 mm.",
        "**Step 3 – apply the moisture multiplier.** Look at the top layer's moisture. At or above the stress scale the multiplier is 1 and the full demand stands. Below the scale, divide the moisture by the scale and multiply that answer by itself. At or below the residual moisture the multiplier is zero and nothing is removed.",
        "**Step 4 – cap by what the layer can give.** Whatever the demand is, SYRUP cannot take more than the water in the layer above its residual moisture.",
        "**Step 5 – take the actual loss out of the layer.** SYRUP debits only the actual amount and records any shortfall as unmet demand. No water appears from nowhere to fill it.",
    ]))
    A(("p", "Two consequences. At half the scale the multiplier is one quarter (half times half); at a quarter of the scale it is one sixteenth. Both are before the water-stock cap, and only if the moisture is still above the residual: for the new JORN soil a quarter of the scale (0.0099) is already below the residual (0.01), so the actual loss there is zero, not one sixteenth. And **lowering the scale lets MORE evaporate at the same moisture**. The scale is not a moisture cut-off below which evaporation stops. The multiplier itself jumps from zero at the residual to a modest value just above it; the actual loss still changes continuously, because the water available above the residual shrinks to nothing there."))
    A(("fig", "fig_moisture_response.png", "Figure 1. The moisture multiplier (the share of the E-scaled demand that moisture allows) against top-layer moisture, for three stress scales. This is the multiplier BEFORE the water-stock cap in step 4. The actual loss can be lower, and is zero at or below the residual (0.01). Moisture is a volume fraction (0.05 = 5 %)."))
    A(("p", "**What E means physically.** E is an empirical scaling of the prescribed reference demand, applied BEFORE the moisture multiplier and the water cap. E = 1.2 scales the reference to 120 %, then the soil still decides how much it can give. It is not an energy-balance result, it does not validate the weather-based PET, and it creates no water. Because the soil's own stock limits the answer, **doubling E need not double the actual evaporation**: once the stock is the limit, extra demand goes unmet."))
    A(("p", "**Why 1.2, and what E is not.** The reference E = 1.2 used in our tests is the combined bare-soil factor of the downloaded EcoHyD example case (its evaporation factor 1 times its vegetation intercept 1.2). It is not a literal factor of 1.2 in that input file, and our bare dry-down driver does not apply a second 1.2 on top. Nor is it the original Tietjen constant a_ET, an additional site or crust factor (section 7). In SYRUP's general vegetation-aware interface the site factor and the cover term are kept separately and the column's own E must be exactly 1, so nothing is applied twice. Do not set 1.2 in several places."))

    A(("h1", "4. A worked example with round numbers"))
    A(("p", f"This is an illustration, not a new field run. A top layer 10 cm thick holds a moisture of {ex['theta']} (5 %). The residual is {ex['residual']} (1 %). PET is {ex['pet']:.0f} mm and E is {ex['E']}."))
    A(("bul", [
        f"**Water available.** Moisture above residual is 0.05 minus 0.01, which is 0.04. Over 100 mm of layer thickness that is **{ex['available_mm']:.0f} mm** that can be taken.",
        f"**Demand before moisture.** 6 mm times 0.5 is **{ex['pre_moisture_mm']:.0f} mm**.",
        f"**Generic soil (scale 0.18).** The moisture 0.05 is below 0.18. Divide 0.05 by 0.18 to get {ex['generic_ratio']:.4f}; multiply that by itself to get the multiplier **{ex['generic_fraction']:.5f}** (7.716 %). Three millimetres times that is **{ex['generic_actual_mm']:.4f} mm**, well below the 4 mm available, so that is the actual loss.",
        "**JORN-derived soil (scale 0.0395).** The moisture 0.05 is above the scale, so the multiplier is **1** and the full 3 mm is demanded. 3 mm is below the 4 mm available, so the actual loss is **3 mm**, about 13 times the generic result.",
        f"**With E = 1.2 instead.** The demand becomes 6 times 1.2, which is 7.2 mm before moisture. The JORN-derived soil lets all of it through the moisture step, but only 4 mm are available, so the actual loss is **4 mm** and 3.2 mm of demand goes unmet. The generic soil would lose {ex['e12_generic_actual_mm']:.4f} mm. So raising E from 0.5 to 1.2 multiplies the JORN-type result by only 4 ÷ 3 = 1.33, because the stock became the limit.",
    ]))
    A(("p", f"**The same calculation with the real first JORN A day**, using values from the saved results. The top layer starts at a moisture of {a['theta0']:.5f} and the first day's PET is {a['pet1']:.3f} mm. With E = 0.5 the demand before moisture is {a['pre_mm']:.3f} mm. Under the earlier generic scale the moisture ratio is {a['theta0'] / STRESS_OLD:.4f}; times itself the multiplier is **{a['old_fraction']:.5f}** (about 7 %), so the first day's loss is {a['old_mm']:.4f} mm. Under the new JORN scale the multiplier is **1**, so the loss is {a['new_mm']:.3f} mm. The layer holds {a['available_mm']:.3f} mm above residual, so over the ten-day window the new soil gives all of it ({a['new_total_mm']:.3f} mm) and ends at the residual, {a['new_final_top']:.2f}. The same first-day arithmetic matches stored records for all six old and new window cases (an independent check by Root)."))

    A(("h1", "5. What changed between the earlier generic soil and the new site soils"))
    A(("tab", "oldnew", "Table 2. Earlier generic reference soil against the texture-derived site soils (moisture values are volume fractions)."))
    A(("p", "The new soils have layer-to-layer conductivities roughly 8 to 18 times larger. In these runs that affects only the exchange between layers; gravity drainage is not limited by it here and is exactly zero. Far more important for evaporation, the new stress scale is much lower. It is the top layer's predicted water content at 1,500 kPa suction from the Saxton–Rawls regression. Choosing that as the evaporation scale is a SYRUP experiment choice, in the spirit of Tietjen's stress moisture (section 7); Saxton–Rawls does not calibrate how evaporation responds to moisture, and the choice does not validate the evaporation."))
    A(("p", "A word on two kinds of averaging. Inside one soil, Tietjen's model and SYRUP take the redistribution conductivity as a fixed fraction of the saturated conductivity (about 31.6 %). Our harmonic-versus-geometric comparison is a different step: it combines the values of two adjacent layers into one conductivity for the boundary between them."))
    A(("h2", "The controlled comparison: d = 0, E = 0.5"))
    A(("p", "To separate the stress-scale effect from everything else, we compare the two soils with d = 0 (no exchange between layers) and E = 0.5. The saved trajectories of the single one-square-metre column give the total evaporation below. In these controls no exchange, no gravity and no saturation limit acts; the residual cap does act. All states stay below field capacity, so drainage, percolation and surface return are exactly zero in every run (the bottom does not leak)."))
    A(("tab", "controls", "Table 3. Total evaporation over each window with d = 0 and E = 0.5, from the saved trajectories (mm of water from a 1 m² column)."))
    A(("p", "Evaporation is roughly 2.4 to 5.7 times larger with the lower scale. JORN A's new total of 3.759 mm is exactly the water its top layer held above residual, so the new soil simply ran out; the residual cap, not the weather, ended the loss."))
    A(("h2", "What happens when d is above zero"))
    A(("p", "With d above zero the interface conductivity matters. It rose from 2.182866 mm per hour (earlier soil) to about 35–38 at JORN and 18–19 at MOAB. A drying top layer is then refilled from the layer below, so water can evaporate again on later days. The actual exchange is not 8 to 18 times larger, because the exchange has a finite capacity and the donor and receiver limits cap it (worked example in section 6). Harmonic versus geometric averaging changes the result only trivially."))
    A(("tab", "outcome", "Table 4. Old and new outcomes from the saved runs at two settings. Score = equal-layer RMSE against observed moisture (volume fraction, ×10⁻³; lower is better). Evaporation = total over the window."))
    A(("p", "In this small comparison the new soils score worse than the earlier generic soil at both settings, mainly because they evaporate too much when the same E is used. That says something about these three short windows and a coarse grid, not about which soil is right."))

    A(("h1", "6. What d does"))
    A(("p", "Neighbouring layers trade water towards the drier one. In SYRUP (and in the later EcoHyD code) the law uses four ingredients: the conductivity between the layers, the thickness of the layer giving up water, the moisture difference, and the distance between the layer centres. **d scales the strength of that trade.** The transfer moves water from the wetter layer to the drier one, so the profile's total water is unchanged; E, by contrast, removes water from the profile."))
    A(("p", f"**A plain-number illustration (analytical, not a model run).** Take two 10 cm layers at the JORN upper interface, one at moisture 0.04 and one at 0.06, with d = 0.05 and one day. Levelling the two would move **{pr['equalising']:.0f} mm** of water, which is the most the exchange can ever move here. SYRUP's unchanged solver moves only a share of that in a day. With the new site conductivity (38.04 mm per hour) the share is about **60 %**, which is **{pr['new_transfer']:.3f} mm**. With the earlier soil's conductivity (2.18 mm per hour) it is only about **5.1 %**, which is **{pr['old_transfer']:.3f} mm**. The conductivity is about 17 times larger, yet the transfer is only {pr['ratio']:.1f} times larger, because it cannot exceed the 1 mm available for levelling; the share levels off as it nears 100 %."))
    A(("p", "Three cautions. First, **d = 0 does not switch off gravity**: water above field capacity still drains downward. **E = 0 does not switch off redistribution or drainage.** Second, this is a simple moisture-gradient approximation, not a Richards-equation solver. Where layers have different retention properties, real water can keep a moisture difference that this law would erase. Third, a given d need not transfer the same amount for other conductivities, layer depths, soils or ways of reading the sensors; it is a calibration constant for a configuration. The exact pair levelling and the top-down sequential update of several layers are SYRUP's own numerical adaptations."))

    A(("h1", "7. Where d and E come from"))
    A(("p", "This section separates what the publications say, what later code and downloaded model inputs contain, and what SYRUP chose."))
    A(("h2", "Tietjen's published model (2009, 2010)"))
    A(("p", T09_PARA))
    A(("p", "Tietjen et al. (2010, Table I) uses d = 0.05, which it says was estimated according to Bai et al. (2007). The same table lists a saturated conductivity of 17.6 mm per hour, a moisture at the beginning of stomatal closure of about 0.079, a field capacity of about 0.226, a residual of 0.06 and layers 0.2 m and 0.6 m thick, credited to unpublished measurements [2]. The table does not describe an optimiser or calibration procedure, and we do not invent one. It lists no evaporation multiplier. The change from 0.01 to 0.05 shows d is chosen for context; it is not a universal soil constant. The \"stomatal closure\" moisture is a plant response level, not a general wilting point."))
    A(("p", "**Our E and the site factor.** In the later EcoHyD code (below) the factor applied to PET is the product of an additional site factor (EP) and a cover term; in our bare dry-down E stands for that whole product. So a site factor of 0.4 would not by itself mean E = 0.4, and we use no such value."))
    A(("h2", "The downloaded later EcoHyD code and parameter files"))
    A(("p", "The reference code we examined is a later application (Baldauf, Cantón and Tietjen 2023 [4]), not the original 2009 code. Its water-cell routine reduces the upper-layer potential loss by a squared moisture ratio below the wilting-point input, scales it by a vegetation function and an extra evaporation factor (EP, described there as an additional site or crust factor), then caps the result by the water above the residual. Its layer-exchange routine uses the diffusion constant, a geometric conductivity and the donor layer's thickness; SYRUP does the same and we make no claim about the original solver. Its Tabernas soil file has EP = 1, a vegetation intercept of 1.2, a wilting point of 0.18, a residual of 0.01 and a diffusion constant of 0.05. These are inputs for that case; their calibration history is not documented here. With bare soil the combined factor is 1 × 1.2 = 1.2, the origin of our reference E."))
    A(("p", "Two later applications (Irob and co-authors, data archives) use much lower values: EP 0.16 (bare effective 0.16 × 1.2 = 0.192, if the cover term is the same 1.2), wilting point 0.088, residual 0.075 and a diffusion constant of 0 [5]. The dissertation labels the evaporation factor as calibrated and the wilting point as calculated from the residual; the objective and optimiser are not established. These belong to a different site and application and are not the parameters of our tests."))
    A(("h2", "Saxton and Rawls (2006)"))
    A(("p", "Saxton and Rawls give regression equations that predict water retention at 1,500 kPa and 33 kPa, saturation and saturated conductivity from sand, clay and organic matter, with density adjustments [6]. The paper has no evaporation multiplier, no squared-ratio evaporation law and no d, and no old evaporation model was replaced by it. Predicting the 1,500 kPa moisture to use as the stress scale is our explicit choice, in the spirit of Tietjen's field-based scale but not calibrated for evaporation. Once the layer's texture and organic matter are fixed, the implemented prediction of that moisture does not depend on our JORN density floor; the floor changes field capacity, saturation and conductivity, not the stress scale. The air-entry and density floors are project bounds, not recommendations of the paper."))
    A(("h2", "What SYRUP chose and tested"))
    A(("p", "The bare diagnostic sets E and d directly for each scenario and the stress scale from the top layer: 0.18 for the earlier generic soil, 0.0395 for JORN and 0.0590 for MOAB; there is no vegetation or transpiration and the terrain is flat. A coarse grid was tried: d of 0, 0.001, 0.003, 0.01, 0.03, 0.05, 0.1, 0.3, 1 and E of 0, 0.1, 0.25, 0.5, 1, 1.2. JORN A was the training window; JORN B and MOAB C were scored without refitting. For the generic soil the best coarse-grid pair was d = 0 and E = 0.5; for the new soils both averaging methods chose d = 0 and E = 0. These are the best of a coarse grid under stated assumptions. They are not measured values of d or E and not evidence that evaporation is zero; untested values between 0 and 0.1 might matter."))
    A(("p", "In SYRUP's general vegetation-aware interface the site factor and the cover function are kept separately, and the column's own E must be exactly 1. The soil evaporation share is applied after the combined moisture and supply cap, SYRUP debits only actual soil evaporation, and the plant model asks for transpiration separately against the water that remains; SYRUP stays the single water authority. Evaporation, transpiration and growth run only between wind and rain events."))
    A(("tab", "matrix", "Table 5. Where each value comes from. The site factor (a_ET in 2009, EP in later code) is NOT the same thing as our combined bare E. Moisture values are volume fractions."))

    A(("h1", "8. How one could estimate d and E, and what this report does not do"))
    A(("p", "Section 7 shows how the cited sources estimated d and the site factor. A general approach for our E would be to compare modelled drying with measured moisture or evaporation over rain-free periods; that is a suggestion of ours. Because d, E and the stress scale all shape the drying curve, one moisture series cannot easily separate them, and a fit for one soil, depth set and sensor arrangement need not carry to another. This report did no new fitting."))

    A(("h1", "9. Limits"))
    A(("bul", [
        "Everything quantitative here comes from three short dry-down windows (JORN A, JORN B, MOAB C) of one-square-metre columns. It is a diagnostic, not field validation.",
        "The worked examples in sections 4 and 6 use round numbers and are illustrations; the 'real first day' numbers are read from saved results.",
        "The site-derived hydraulic values are pedotransfer surrogates with project bounds, not measured hydraulic properties.",
        "No soil here exceeds field capacity, so gravity drainage and the saturation limit are untested by these runs.",
    ]))

    A(("h1", "Sources"))
    A(("tab", "appendix", "Table 6. Main sources. Exact file hashes are in evidence/provenance.json."))
    A(("refs", [
        ("1", T09_OTHER[1], "Publisher full text: https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2007WR006589 (recovered 2026-11-17)."),
        ("2", "Tietjen, B., Jeltsch, F., Zehe, E., Classen, N., Groengroeft, A., Schiffers, K. and Oldeland, J. (2010). Effects of climate change on the coupled dynamics of water and vegetation in drylands. Ecohydrology 3, 226–237 (online 2009). DOI 10.1002/eco.70, Table I.", "Author-hosted copy: https://www.biodiversity-plants.de/downloads/Tietjen_etal_2009_Ecohydro.pdf"),
        ("3", "Bai, J., Deng, W., Cui, B. and Ouyang, H. (2007). Water diffusion coefficients of horizontal soil columns from natural saline-alkaline wetlands in a semiarid area. Eurasian Soil Science 40, 660–664. DOI 10.1134/S1064229307060075.", ""),
        ("4", "Baldauf, S., Cantón, Y. and Tietjen, B. (2023). Biocrusts intensify water redistribution and improve water availability to dryland vegetation: insights from a spatially-explicit ecohydrological model. Frontiers in Microbiology 14, 1179291. DOI 10.3389/fmicb.2023.1179291. Code commit 80b10e6b6b82ad16e9fd41a31901cc3157d880e9.", "Local copy under reference_models/ecohyd (GPL source, not reproduced)."),
        ("5", "Irob (2023). Doctoral dissertation, Freie Universität Berlin, Appendix Table A.1.3, p. 138.", "https://refubium.fu-berlin.de/bitstream/handle/fub188/40230/Dissertation_Irob.pdf"),
        ("6", "Saxton, K. E. and Rawls, W. J. (2006). Soil water characteristic estimates by texture and organic matter for hydrologic solutions. Soil Science Society of America Journal 70, 1569–1578. DOI 10.2136/sssaj2005.0117.", ""),
    ]))
    return B


def tables(R):
    rows = R["rows"]
    T = {}
    T["terms"] = (
        ["Name", "What it is", "Units", "What it does NOT mean"],
        [
            ["PET (potential evapotranspiration)", "Weather-driven demand for water, from date, air temperatures and latitude", "mm per day", "Actual water lost"],
            ["E (evaporation multiplier)", "Scales the PET given to the soil before any soil limit", "none (dimensionless)", "Millimetres of evaporation; a measured quantity"],
            ["d (exchange coefficient)", "Scales how fast neighbouring layers trade water", "none (dimensionless)", "Physical diffusivity (m² per s), drainage rate, water amount, sediment parameter"],
            ["Evaporation-stress scale", "Top-layer moisture at and above which the moisture multiplier is 1", "volume fraction (0.18 = 18 %)", "A cut-off where evaporation stops; the residual"],
            ["Residual moisture", "Lowest moisture the model lets the layer reach", "volume fraction", "The stress scale"],
        ],
    )
    T["oldnew"] = (
        ["Soil", "Stress scale", "Residual", "Saturated conductivity (mm/h)", "Layer-to-layer conductivity (mm/h)"],
        [
            ["Earlier generic reference", "0.18", "0.01", "6.9", "2.1829 (all pairs)"],
            ["New JORN (3 layers)", "0.0395 (top-layer water content at 1,500 kPa)", "0.01", "120.8 / 119.7 / 104.3", "38.0 and 35.3 (harmonic)"],
            ["New MOAB (3 layers)", "0.0590 (same rule)", "0.01", "64.0 / 57.3 / 57.3", "19.1 and 18.1 (harmonic)"],
        ],
    )
    T["controls"] = (
        ["Window", "Days", "Earlier generic (mm)", "New site soil (mm)", "New ÷ old"],
        [[s, str(r["days"]), f"{r['d0_E0.5_open']['old_evap_mm']:.4f}", f"{r['d0_E0.5_open']['new_evap_mm']:.4f}", f"{r['d0_E0.5_open']['new_evap_mm'] / r['d0_E0.5_open']['old_evap_mm']:.2f}"] for s, r in rows.items()],
    )
    out = []
    for s, r in rows.items():
        for key, lab in (("d0_E0.5_open", "d 0, E 0.5"), ("d0.05_E1.2_open", "d 0.05, E 1.2")):
            v = r[key]
            out.append([s, lab, f"{v['old_evap_mm']:.3f}", f"{v['new_evap_mm']:.3f}", f"{v['old_rmse'] * 1000:.2f}", f"{v['new_rmse'] * 1000:.2f}"])
    T["outcome"] = (["Window", "Setting", "Evaporation old (mm)", "Evaporation new (mm)", "Score old (×10⁻³)", "Score new (×10⁻³)"], out)
    T["matrix"] = (
        ["Source / case", "d", "Site factor (a_ET or EP)", "Combined bare E equivalent", "Stress / wilting scale", "Residual", "How obtained"],
        [
            ["Tietjen 2009, Table 2", *T09_MATRIX],
            ["Tietjen 2010, Table I", "0.05", "No multiplier listed", "Not given", "About 0.079 (start of stomatal closure)", "0.06", "d estimated according to Bai 2007; soil values unpublished measurements"],
            ["EcoHyD Tabernas input (later code)", "0.05", "EP 1", "1 × cover intercept 1.2 = 1.2 (flat bare)", "0.18 (wilting point)", "0.01", "Case input; calibration history not established"],
            ["Irob data archives (later, other sites)", "0", "EP 0.16", "0.16 × 1.2 = 0.192 (flat bare)", "0.088", "0.075", "Dissertation labels EP calibrated, wilting point calculated; objective unverified"],
            ["Saxton–Rawls 2006", "none", "none", "none", "Predicts water at 1,500 kPa", "none", "Texture regression; no evaporation or d law"],
            ["SYRUP earlier generic soil", "grid; best 0", "not separate", "grid; best 0.5", "0.18", "0.01", "Coarse grid, JORN A training; not measured"],
            ["SYRUP new JORN / MOAB", "grid; best 0", "not separate", "grid; best 0", "0.0395 / 0.0590 (predicted 1,500 kPa water)", "0.01", "Experiment choice, consistent with Tietjen's concept; not calibrated for E"],
        ],
    )
    T["appendix"] = (
        ["Source", "Used for"],
        [
            ["Tietjen et al. 2009 (publisher text); Tietjen et al. 2010 PDF Table I", "Origins of d, the site factor and the stress moisture"],
            ["EcoHyD WaterCell.cpp lines 28–112, 475–495; WaterCell.h line 76; soilparameters_tabernas.txt; two Irob soilparameters_EH.txt", "Later-code evaporation and exchange laws; case inputs"],
            ["Saxton and Rawls 2006 PDF", "Retention prediction at 1,500 kPa"],
            ["SYRUP subsurface.py (step stages 4–5), soil_evaporation.py, drydown_driver.py (candidate worktree)", "Law as implemented; bare diagnostic settings"],
            ["Saved results: data/actual_cpu/*_result.json and property_effect.json; ../observed_drydown/data/*_result.json", "All numbers in tables and examples"],
            ["Root checks: root_numeric_checks.json, root_d_example_check.json, root_source_evidence.json, primary_2009_update.md", "Independent checks and 2009 evidence record"],
        ],
    )
    return T


# ------------------------------------------------------------------ writers
def md_table(hd, rows):
    return "\n".join(["| " + " | ".join(hd) + " |", "|" + "---|" * len(hd)] + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def to_markdown(B, T):
    out = []
    for b in B:
        k = b[0]
        if k == "title":
            out.append(f"# {b[1]}\n")
        elif k == "sub":
            out.append(f"*{b[1]}*\n")
        elif k == "h1":
            out.append(f"\n## {b[1]}\n")
        elif k == "h2":
            out.append(f"\n### {b[1]}\n")
        elif k == "p":
            out.append(b[1] + "\n")
        elif k == "bul":
            out.append("\n".join(f"- {x}" for x in b[1]) + "\n")
        elif k == "tab":
            out.append(f"**{b[2]}**\n\n{md_table(*T[b[1]])}\n")
        elif k == "fig":
            out.append(f"![{b[2]}](figures/{b[1]})\n\n*{b[2]}*\n")
        elif k == "refs":
            out.append("\n".join(f"[{n}] {c} {u}".rstrip() for n, c, u in b[1]) + "\n")
    return "\n".join(out)


def shade(cell, color):
    sh = OxmlElement("w:shd")
    sh.set(qn("w:val"), "clear")
    sh.set(qn("w:color"), "auto")
    sh.set(qn("w:fill"), color)
    cell._tc.get_or_add_tcPr().append(sh)


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
    for nm, base in (("EvapTitle", "Title"), ("EvapSubtitle", "Subtitle"), ("EvapCaption", "Normal"), ("EvapTable", "Normal")):
        st.add_style(nm, WD_STYLE_TYPE.PARAGRAPH).base_style = st[base]
    t = st["EvapTitle"]
    t.font.name, t.font.size, t.font.bold = "Calibri", Pt(21), True
    t.font.color.rgb = RGBColor.from_string("1F3A5F")
    s = st["EvapSubtitle"]
    s.font.size, s.font.italic = Pt(10.5), True
    c = st["EvapCaption"]
    c.font.size, c.font.italic = Pt(9), True
    c.paragraph_format.keep_together = True
    tt = st["EvapTable"]
    tt.font.size = Pt(8.5)
    tt.paragraph_format.space_after = Pt(0)
    tt.paragraph_format.line_spacing = 1.0


def add_table(doc, hd, rows, caption, width=6.7):
    cp = doc.add_paragraph(style="EvapCaption")
    runs(cp, caption)
    cp.paragraph_format.keep_with_next = True
    t = doc.add_table(rows=1, cols=len(hd))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    wts = [max(len(str(h)) * 0.5, min(max(len(str(r[i])) for r in rows), 45), 6) for i, h in enumerate(hd)]
    ws = [Inches(width * w / sum(wts)) for w in wts]
    for i, h in enumerate(hd):
        c = t.rows[0].cells[i]
        c.text = ""
        c.paragraphs[0].style = "EvapTable"
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
            cells[i].paragraphs[0].style = "EvapTable"
            cells[i].paragraphs[0].add_run(str(v))
            cells[i].width = ws[i]
            if ri % 2:
                shade(cells[i], "F3F6FA")
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def to_docx(B, T):
    doc = Document()
    setup_styles(doc)
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Inches(8.5), Inches(11)
    sec.left_margin = sec.right_margin = Inches(0.9)
    sec.top_margin = sec.bottom_margin = Inches(0.85)
    fp = sec.footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    fp.add_run("SYRUP soil-property results: evaporation, moisture stress, d and E · diagnostic explanation, not field validation · page ").font.size = Pt(8)
    add_field(fp, "PAGE")
    for b in B:
        k = b[0]
        if k == "title":
            doc.add_paragraph(b[1], style="EvapTitle")
        elif k == "sub":
            doc.add_paragraph(b[1], style="EvapSubtitle")
        elif k == "h1":
            doc.add_heading(b[1], level=1)
        elif k == "h2":
            doc.add_heading(b[1], level=2)
        elif k == "p":
            runs(doc.add_paragraph(), b[1])
        elif k == "bul":
            for x in b[1]:
                runs(doc.add_paragraph(style="List Bullet"), x)
        elif k == "tab":
            add_table(doc, *T[b[1]], b[2])
        elif k == "fig":
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.keep_with_next = True
            p.add_run().add_picture(str(FIGDIR / b[1]), width=Inches(6.2))
            runs(doc.add_paragraph(style="EvapCaption"), b[2])
        elif k == "refs":
            for n, c, u in b[1]:
                p = doc.add_paragraph()
                p.paragraph_format.left_indent = Inches(0.25)
                p.paragraph_format.first_line_indent = Inches(-0.25)
                p.add_run(f"[{n}] ").bold = True
                p.add_run(f"{c} {u}").font.size = Pt(9)
    doc.core_properties.title = "Why the soil-property results differ: evaporation, moisture stress, and d and E"
    doc.core_properties.author = "SYRUP project"
    doc.save(DOCX)


def provenance():
    files = {
        "src/maple_syrup/subsurface.py": (CAND / "src/maple_syrup/subsurface.py", "SYRUP column step stages (evaporation stage 4, exchange stage 5), docstring lines 20-75"),
        "src/maple_syrup/soil_evaporation.py": (CAND / "src/maple_syrup/soil_evaporation.py", "vegetation-aware E-only helper; evaporation_factor must be 1"),
        "benchmarks/subsurface_validation/drydown_driver.py": (CAND / "benchmarks/subsurface_validation/drydown_driver.py", "bare driver: E, d, stress scale set directly"),
        "diffusivity_literature_and_fit.md": (LIVE / "docs/subsurface_hydrology/diffusivity_literature_and_fit.md", "Tietjen 2009 eq. 9, d provenance"),
        "Tietjen_2010.pdf": (LIVE / "agent_handoffs/tasks/diffusivity_literature_sweep_20261101/sources/Tietjen_2010.pdf", "Table I (d 0.05, soil values)"),
        "saxton_rawls_2006.pdf": (LIVE / "agent_handoffs/tasks/neon_texture_failure_audit_20261117/saxton_rawls_2006.pdf", "primary SR2006 paper"),
        "ecohyd WaterCell.cpp": (LIVE / "reference_models/ecohyd/github_baldauf_2023/source/model/src/WaterCell.cpp", "lines 28-112 ET, 34-35 vegetation factor, 84-92 moisture ratio and cap, 475-495 layerDiffusion"),
        "ecohyd WaterCell.h": (LIVE / "reference_models/ecohyd/github_baldauf_2023/source/model/src/WaterCell.h", "line 76 EPfactor description"),
        "soilparameters_tabernas.txt": (LIVE / "reference_models/ecohyd/github_baldauf_2023/source/model/Parameters/soilparameters_tabernas.txt", "EP 1, WP 0.18, rw 0.01, diffConst 0.05, VegregETL1 1.2"),
        "Irob soilparameters_EH.txt (4j0zpc8gn)": (LIVE / "reference_models/ecohyd/dryad_4j0zpc8gn/extracted/Irob_et_al_drought_resilience/Model/Parameters/soilparameters_EH.txt", "EP 0.16, WP 0.088, rw 0.075, diffConst 0"),
        "Irob soilparameters_EH.txt (ngf1vhj1t)": (LIVE / "reference_models/ecohyd/dryad_ngf1vhj1t/extracted/Navigating_Uncertainty_Irob_et_al_23/Source/Parameters/soilparameters_EH.txt", "same values as above"),
        "primary_2009_update.md": (TASK / "primary_2009_update.md", "Root record of the recovered original-2009 publisher text (Sections 2.3, 3.1, Table 2)"),
        "root_numeric_checks.json": (TASK / "root_numeric_checks.json", "independent first-day and control arithmetic"),
        "root_d_example_check.json": (TASK / "root_d_example_check.json", "independent redistribution example arithmetic"),
        "root_source_evidence.json": (TASK / "root_source_evidence.json", "hashes of the read-only sources"),
        "JORN4_A_training_harmonic_result.json": (DATA / "JORN4_A_training_harmonic_result.json", "new-soil trajectories, first-day moisture and PET"),
        "JORN4_B_heldout_harmonic_result.json": (DATA / "JORN4_B_heldout_harmonic_result.json", "new-soil trajectories"),
        "MOAB2_C_heldout_harmonic_result.json": (DATA / "MOAB2_C_heldout_harmonic_result.json", "new-soil trajectories"),
        "old JORN4_A_training_result.json": (OLD / "JORN4_A_training_result.json", "earlier generic trajectories"),
        "old JORN4_B_heldout_result.json": (OLD / "JORN4_B_heldout_result.json", "earlier generic trajectories"),
        "old MOAB2_C_heldout_result.json": (OLD / "MOAB2_C_heldout_result.json", "earlier generic trajectories"),
        "JORN4_A_property_effect.json": (DATA / "JORN4_A_property_effect.json", "old/new scores"),
        "JORN4_B_property_effect.json": (DATA / "JORN4_B_property_effect.json", "old/new scores"),
        "MOAB2_C_property_effect.json": (DATA / "MOAB2_C_property_effect.json", "old/new scores"),
    }
    out = {}
    for k, (p, role) in files.items():
        assert p.is_file(), p
        out[k] = {"path": str(p), "role": role, "sha256": sha(p)}
    return out


def verify_docx(B, T):
    z = zipfile.ZipFile(DOCX)
    assert z.testzip() is None
    names = z.namelist()
    xml = z.read("word/document.xml").decode()
    media = [n for n in names if n.startswith("word/media/")]
    footers = [n for n in names if re.match(r"word/footer\d*\.xml", n)]
    foot = "".join(z.read(n).decode() for n in footers)
    text = re.sub(r"<[^>]+>", " ", xml)
    info = {
        "zip_ok": True,
        "tables": xml.count("<w:tbl>"),
        "tables_expected": sum(1 for b in B if b[0] == "tab"),
        "media": media,
        "page_field_in_footer": "PAGE" in foot and "fldChar" in foot,
        "styles_part": "word/styles.xml" in names,
        "latex_or_equation_markup": bool(re.search(r"\\frac|\$\$|<m:oMath", xml)),
        "words": len(text.split()),
        "bytes": DOCX.stat().st_size,
        "header_rows_repeat": xml.count("<w:tblHeader"),
        "caret_or_exp_formula_text": bool(re.search(r"e\^|\^\(|1 − e", text)),
    }
    empty = [c for c in re.findall(r"<w:tc>(.*?)</w:tc>", xml, re.DOTALL) if not re.sub(r"<[^>]+>", "", c).strip()]
    info["empty_table_cells"] = len(empty)
    assert info["header_rows_repeat"] == info["tables"] and info["empty_table_cells"] == 0 and not info["caret_or_exp_formula_text"]
    assert info["tables"] == info["tables_expected"] and len(media) == 1 and info["page_field_in_footer"] and not info["latex_or_equation_markup"]
    return info


def main():
    R = derive()
    figure(R)
    T = tables(R)
    P = provenance()
    B = content(R)
    md = to_markdown(B, T)
    for chunk in (T09_PARA, *T09_OTHER, *T09_MATRIX):
        assert chunk in md, chunk[:50]  # every counted 2009-attributed string really is in the document
    (HERE / "explanation.md").write_text(md, encoding="utf-8")
    for k, (hd, rows) in T.items():
        with open(TABDIR / f"{k}.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh, lineterminator="\n")
            w.writerow(hd)
            w.writerows(rows)
    to_docx(B, T)
    info = verify_docx(B, T)
    (HERE / "evidence").mkdir(exist_ok=True)
    (HERE / "evidence" / "provenance.json").write_text(
        json.dumps(
            {
                "task": "soil_evaporation_explanation_20261117",
                "inputs": P,
                "docx": {"path": str(DOCX), "sha256": sha(DOCX), **info},
                "figure_sha256": sha(FIGDIR / "fig_moisture_response.png"),
                "worked_numbers": {k: R[k] for k in ("example", "actual_first_day", "pair")},
                "controls": R["rows"],
                "build_command": "cd <candidate>/docs/subsurface_hydrology/site_properties_mixing/evaporation_explanation && PYTHONPATH=/home/okin/SYRUP/agent_handoffs/tasks/site_hydraulics_interface_comparison_20261117/report_tools MPLCONFIGDIR=/tmp/syrup-evap-mpl PYTHONDONTWRITEBYTECODE=1 /home/okin/MAPLE/.venv/bin/python -B build_explanation.py",
                "original_2009_evidence": {
                    "source": "https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2007WR006589 (publisher full text indexed via Root web search, recovered 2026-11-17 UTC; direct open still refused)",
                    "record": "primary_2009_update.md",
                    "facts_used": "Sections 2.3 and 3.1, Table 2: a_ET 1 at Ein Yaaqov/Matta/Lahav, 0.4 calibrated at Sde Boqer; d 0.01 at all four sites adjusted to Bai 2007 scale and moisture series; wsc from similar nearby sites, listed in percent",
                    "2009_attributed_words_total_all_blocks": T09_WORDS,
                    "2009_attributed_blocks": ["T09_PARA", "T09_MATRIX", "T09_OTHER (E-link sentence, bibliography)"],
                    "not_claimed": "original PDF, original 2009 code, 2009 optimiser, 2009 Equation 9 donor-layer choice, any 2023/Irob value for 2009",
                },
                "not_verified": ["Office pagination/rendering", "original 2009 PDF/code and optimiser details"],
            },
            indent=1,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (HERE / "verification.md").write_text(
        "# Verification (static)\n\n"
        f"- DOCX ZIP integrity OK; {info['tables']} tables (expected {info['tables_expected']}); media {info['media']}; PAGE field in footer: {info['page_field_in_footer']}; styles part present; no LaTeX/equation markup.\n"
        f"- About {info['words']} words; {info['bytes']} bytes. Page count NOT measured (no Office/renderer available), so the 5-8 page target is unconfirmed.\n"
        "- Every worked number is recomputed from saved JSON in the builder and asserted against the task values (old/new totals, fractions, first-day PET and moisture, first-day trajectory evaporation, zero drainage/percolation/return).\n"
        "- No model, GPU or Git command was run. Original report, manifests, data and figures untouched.\n"
        f"- Original-2009 facts (a_ET, d, wsc) come from the publisher full text recovered by Root (see evidence/provenance.json); {T09_WORDS} words attributed to the 2009 paper in total across body, matrix, estimation note and bibliography (asserted at most 200); no long quotations. The publisher text was used, not the original PDF or code; the 2009 optimiser and Equation 9 donor choice are not claimed.\n"
        "- First-day arithmetic for all six old/new window cases and the redistribution illustration are asserted against Root's root_numeric_checks.json and root_d_example_check.json.\n"
        f"- Header rows repeat on all {info['tables']} tables; every table cell is non-empty; no caret or exponential formula text in the DOCX; figure is 1 PNG.\n"
        "- NOT verified: Office layout/pagination; original 2009 PDF/code and optimiser details. No publication claim.\n",
        encoding="utf-8",
    )
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
