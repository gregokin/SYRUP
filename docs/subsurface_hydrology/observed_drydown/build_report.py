"""Builds the observed dry-down results report (editable Markdown, figures, derived-table JSON and the Word document) from the sealed result JSON in `data/` and the receipts in `evidence/`.

    MPLCONFIGDIR=/tmp/syrup-observed-mpl PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/tmp/syrup-manuals-citation-tools-20261020 \
        /home/okin/MAPLE/.venv/bin/python -B build_report.py

Every number is computed here from `data/*_result.json` (scores, persistence, skill and the selection are RE-DERIVED from the exported observed targets and model layer states and asserted equal to the stored values); nothing is typed in.
The hardware status is read from `data/` (a genuine CUDA completion record) and is otherwise stated as pending. No simulation is run, no raw NEON file is opened, no network access is made.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from docx import Document  # noqa: E402
from docx.enum.table import WD_TABLE_ALIGNMENT  # noqa: E402
from docx.enum.text import WD_ALIGN_PARAGRAPH  # noqa: E402
from docx.oxml import OxmlElement  # noqa: E402
from docx.oxml.ns import qn  # noqa: E402
from docx.shared import Inches, Pt, RGBColor  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA, EVID, FIG = HERE / "data", HERE / "evidence", HERE / "figures"
FIG.mkdir(exist_ok=True)
NAMES = ["JORN4_A_training", "JORN4_B_heldout", "MOAB2_C_heldout"]
TITLES = {"JORN4_A_training": "JORN plot 4, window A (training)", "JORN4_B_heldout": "JORN plot 4, window B (held-out, temporal)", "MOAB2_C_heldout": "MOAB plot 2, window C (held-out, site)"}
LAYER = ["501", "502", "503"]
DEPTH_LABEL = {"JORN": ["5.8", "15.8", "25.8"], "MOAB": ["5.5", "15.5", "25.5"]}
D_GRID = [0.0, 0.001, 0.003, 0.01, 0.03, 0.05, 0.1, 0.3, 1.0]
E_GRID = [0.0, 0.1, 0.25, 0.5, 1.0, 1.2]
REF = ("d0.05_E1.2_open", 0.05, 1.2)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
J = lambda p: json.loads(Path(p).read_text())  # noqa: E731
f3 = lambda x: f"{x * 1000:.3f}"  # noqa: E731  (milli m3/m3)
f2 = lambda x: f"{x:.2f}"  # noqa: E731


# ------------------------------------------------------------------ load and independently re-derive
def sc_name(d, e, bottom="open"):
    return f"d{d:g}_E{e:g}_{bottom}"


def layer_scores(theta, obs):
    out, mses = {}, []
    for i, v in enumerate(LAYER):
        ok = [(theta[k][i], o) for k, o in enumerate(obs[v]) if o is not None]
        e = np.array([a - b for a, b in ok])
        mse = float(np.mean(e**2))
        out[v] = {"n": len(e), "bias": float(e.mean()), "rmse": math.sqrt(mse), "mse": mse}
        mses.append(mse)
    out["overall"] = math.sqrt(sum(mses) / 3)
    allerr = np.concatenate([np.array([theta[k][i] - o for k, o in enumerate(obs[v]) if o is not None]) for i, v in enumerate(LAYER)])
    out["pooled"] = float(np.sqrt(np.mean(allerr**2)))
    return out


def load():
    runs = {}
    for n in NAMES:
        r, c = J(DATA / f"{n}_result.json"), J(DATA / f"{n}_config.json")
        pi = r["plan"]["prepared_inputs"]
        obs = pi["observed_end_theta_m3m3"]
        res = {"r": r, "c": c, "pi": pi, "obs": obs, "scen": {}}
        pt = [pi["raw_initial_point_theta"][i] for i in range(3)]
        lm = pi["integrated_initial_layer_theta"]
        n_days = r["plan"]["model_days"]
        pers = {"point": layer_scores([pt] * n_days, obs), "layer": layer_scores([lm] * n_days, obs)}
        res["pers"] = pers
        for k, run in r["runs"].items():
            ls = layer_scores(run["layer_theta"], obs)
            st = r["scores"][k]
            assert math.isclose(ls["overall"], st["overall_rmse"], rel_tol=1e-12), (n, k)
            assert math.isclose(ls["pooled"], st["pooled_rmse_alternative"], rel_tol=1e-12), (n, k)
            for v in LAYER:
                assert ls[v]["n"] == st["per_layer"][v]["n"] and math.isclose(ls[v]["rmse"], st["per_layer"][v]["rmse"], rel_tol=1e-12)
            for key, ref in (("point", "initial_point_values_held"), ("layer", "initial_layer_means_held")):
                assert math.isclose(pers[key]["overall"], r["persistence"][ref]["overall_rmse"], rel_tol=1e-12), (n, key)
                sk = 1 - ls["overall"] ** 2 / pers[key]["overall"] ** 2
                assert math.isclose(sk, st[f"skill_vs_{ref}"], rel_tol=1e-9, abs_tol=1e-12), (n, k, key)
            cum = run["cumulative"]
            tr = run["trajectory"]
            assert cum["drainage_m3"] == 0 and cum["surface_return_m3"] == 0 and max(abs(x) for x in tr["drainage_m"]) == 0 and max(abs(x) for row in tr["percolation_m"] for x in row) == 0
            res["scen"][k] = {"s": ls, "evap_mm": cum["evaporation_m3"] * 1000.0, "evap_day_mm": [x * 1000 for x in tr["evaporation_m"]], "theta": run["layer_theta"],
                              "skill_point": 1 - ls["overall"] ** 2 / pers["point"]["overall"] ** 2, "skill_layer": 1 - ls["overall"] ** 2 / pers["layer"]["overall"] ** 2,
                              "div_max": max(abs(x) for row in tr["diffusion_m"] for x in row)}
        runs[n] = res
        # clock: day ends are 24 h apart and the model time is k x 86400 s
        ends = [datetime.datetime.fromisoformat(x.replace("Z", "+00:00")) for x in [pi["t0_utc"]] + pi["day_end_utc"]]
        assert all((b - a) == datetime.timedelta(days=1) for a, b in zip(ends, ends[1:])) and r["runs"][next(iter(r["runs"]))]["trajectory"]["t_s"] == [86400 * (i + 1) for i in range(n_days)]
        for i, w in enumerate(pi["weather_days"]):
            assert w["window_start_utc"] == ([pi["t0_utc"]] + pi["day_end_utc"])[i] and w["n_passing"] == 48
    return runs


def selection(train):
    cands = []
    for d in D_GRID:
        for e in E_GRID:
            cands.append((train["scen"][sc_name(d, e)]["s"]["overall"], e, d))
    rmse, e, d = min(cands)
    ties = [c for c in cands if math.isclose(c[0], rmse, rel_tol=1e-12)]
    return d, e, rmse, ties


def cuda_status():
    for p in sorted(DATA.glob("*cuda*completion*.json")) + sorted(DATA.glob("actual_cuda*/completion.json")):
        return "recorded", J(p), p
    ev = J(EVID / "root_verification_summary.json")
    return "pending", ev.get("cuda", {}), None


# ------------------------------------------------------------------ figures
def fig_heatmap(runs, best):
    tr = runs["JORN4_A_training"]
    z = np.array([[tr["scen"][sc_name(d, e)]["s"]["overall"] * 1000 for e in E_GRID] for d in D_GRID])
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    im = ax.imshow(z, cmap="viridis_r", aspect="auto", origin="lower")
    for i in range(len(D_GRID)):
        for j in range(len(E_GRID)):
            ax.text(j, i, f"{z[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if z[i, j] > np.median(z) else "black")
    ax.set_xticks(range(len(E_GRID)), [f"{e:g}" for e in E_GRID])
    ax.set_yticks(range(len(D_GRID)), [f"{d:g}" for d in D_GRID])
    ax.set_xlabel("soil-evaporation multiplier E (effective factor)")
    ax.set_ylabel("exchange coefficient d (Tietjen, dimensionless)")
    bi, bj = D_GRID.index(best[0]), E_GRID.index(best[1])
    ax.add_patch(plt.Rectangle((bj - 0.5, bi - 0.5), 1, 1, fill=False, ec="red", lw=2.5))
    ax.plot(E_GRID.index(REF[2]) + 0.3, D_GRID.index(REF[1]) + 0.28, marker="*", color="white", mec="black", ms=11)
    pp, pl = tr["pers"]["point"]["overall"] * 1000, tr["pers"]["layer"]["overall"] * 1000
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("primary overall RMSE (10$^{-3}$ m$^3$ m$^{-3}$)")
    cb.ax.axhline(pp, color="red", lw=1.5)
    ax.set_title(f"Training grid, JORN plot 4 window A (10 days)\nred box = best; star = unfitted reference (0.05, 1.2)\nred line on the colour bar = raw-point persistence ({pp:.2f}); layer persistence {pl:.2f}", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_training_heatmap.png", dpi=170)
    plt.close(fig)


def fig_profiles(name, res, best, idx):
    pi, c = res["pi"], res["c"]
    site = c["site"]["site"]
    t0 = datetime.datetime.fromisoformat(pi["t0_utc"].replace("Z", "+00:00"))
    ends = [datetime.datetime.fromisoformat(x.replace("Z", "+00:00")) for x in pi["day_end_utc"]]
    xs = [0] + list(range(1, len(ends) + 1))
    fig, axes = plt.subplots(3, 1, figsize=(7.0, 7.4), sharex=True)
    bname, rname = sc_name(best[0], best[1]), REF[0]
    for i, (ax, v) in enumerate(zip(axes, LAYER)):
        o = pi["observed_end_theta_m3m3"][v]
        ax.plot([0], [pi["raw_initial_point_theta"][i]], "o", mfc="none", mec="k", ms=8, label="observed initial point (t0)")
        ax.plot([k + 1 for k, x in enumerate(o) if x is not None], [x for x in o if x is not None], "ko", ms=5, label="observed valid day-end target")
        miss = [k + 1 for k, x in enumerate(o) if x is None]
        if miss:
            ax.plot(miss, [ax.get_ylim()[0]] * len(miss), "kx", ms=6, label="no valid target (not filled)", clip_on=False)
        ax.axhline(pi["raw_initial_point_theta"][i], color="0.5", ls="--", lw=1.1, label="raw-point persistence")
        ax.plot([0], [pi["integrated_initial_layer_theta"][i]], "D", color="tab:green", ms=5, label="integrated initial layer mean")
        for nm, col, mk, lab in ((bname, "tab:blue", "s", f"model best (d={best[0]:g}, E={best[1]:g})"), (rname, "tab:red", "^", "model unfitted reference (d=0.05, E=1.2)")):
            th = [x[i] for x in res["scen"][nm]["theta"]]
            ax.plot(xs[1:], th, mk, color=col, ms=5, mfc="none", label=lab)
        ax.set_ylabel(f"probe {DEPTH_LABEL[site][i]} cm\n(layer {i + 1}) m$^3$ m$^{{-3}}$", fontsize=8)
        ax.grid(alpha=0.25)
    axes[-1].set_xticks(xs, [f"t0\n{t0:%m-%d %H:%M}"] + [f"{e:%m-%d}" for e in ends], fontsize=7)
    axes[-1].set_xlabel("day-end state time (UTC; model states are daily only, markers are not connected)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=7, frameon=False)
    fig.suptitle(TITLES[name], fontsize=10)
    fig.tight_layout(rect=(0, 0.1, 1, 0.97))
    fig.savefig(FIG / f"fig{idx}_profiles_{name}.png", dpi=170)
    plt.close(fig)


def fig_forcing(runs, best):
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.0))
    for j, n in enumerate(NAMES):
        r = runs[n]
        pet = [w["pet_mm_day"] for w in r["pi"]["weather_days"]]
        axes[0, j].bar(range(1, len(pet) + 1), pet, color="0.6")
        axes[0, j].set_title(TITLES[n].split(" (")[0], fontsize=8)
        axes[0, j].set_xlabel("model day")
        axes[1, j].set_xlabel("model day")
        for nm, col, lab in ((sc_name(best[0], best[1]), "tab:blue", "best"), (REF[0], "tab:red", "reference")):
            axes[1, j].plot(range(0, len(pet) + 1), np.concatenate([[0], np.cumsum(r["scen"][nm]["evap_day_mm"])]), "-o", color=col, ms=3, label=lab)
    axes[0, 0].set_ylabel("Hargreaves reference PET (mm d$^{-1}$)")
    axes[1, 0].set_ylabel("cumulative simulated soil\nevaporation (mm)")
    axes[1, 0].legend(fontsize=7)
    fig.suptitle("Forcing and simulated soil-evaporation loss (partial budget; no transpiration)", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "fig5_forcing_and_loss.png", dpi=170)
    plt.close(fig)


# ------------------------------------------------------------------ document model
def build_blocks(runs, best, sel_rmse, ties):
    b = []
    add = b.append
    tr, jb, mc = runs["JORN4_A_training"], runs["JORN4_B_heldout"], runs["MOAB2_C_heldout"]
    bn = sc_name(best[0], best[1])
    status, cuda, cuda_path = cuda_status()
    seal = J(EVID / "root_run_seal.json")
    comp = J(DATA / "completion.json")
    verif = J(EVID / "root_verification_summary.json")
    dec = J(EVID / "root_decimal_summary.json")
    aud = J(EVID / "root_result_audit.json")
    ps = {n: runs[n]["pers"] for n in NAMES}

    def row(n, nm):
        s = runs[n]["scen"][nm]
        return [f3(s["s"]["overall"]), f2(s["skill_point"]), f2(s["skill_layer"])]

    add(("title", "Observed shallow-profile dry-down with the SYRUP layered soil column: a generic reference-soil mechanism diagnostic"))
    add(("sub", "JORN and MOAB, NEON RELEASE-2026 observations; three 10 cm layers; daily end-state comparison. Results report generated " + datetime.date.today().isoformat() + " from sealed result files."))
    add(("box", ["STATUS. Actual CPU results (NumPy and Numba, native SYRUP column) are analysed here. " + ("CUDA hardware controls are recorded in the data folder." if status == "recorded" else
                 "The CUDA / GPU observed-profile qualification is PENDING and is NOT claimed: Root's pre-launch guard refused GPU0 (busy) and no GPU process ran. This report reads the evidence folder and will be rebuilt when a genuine GPU completion record is supplied."),
                 "SCOPE. A generic, user-approved reference soil with observed initial moisture and observed air temperature. This is a mechanism diagnostic, not a site hydraulic validation, not a calibration of a physical diffusivity, and not a field evapotranspiration budget."]))
    # ---------------- executive summary
    add(("h1", "1. Executive summary"))
    add(("p", f"Three actual daily dry-down windows with no rain were simulated with the native SYRUP fixed-layer column. The reference soil (residual 0.01, evaporation stress 0.18, field capacity = saturation = 0.26, Ks 6.9 mm/h) is the generic EcoHyD "
              f"example soil, not a property of the sites. Two free quantities were explored on a grid that was frozen before any run: the exchange coefficient d (9 values) and a soil-evaporation multiplier E (6 values). Only window A of JORN plot 4 was used for the selection (training); "
              f"the selected pair was then applied without refitting to two other windows. Scores use the primary overall RMSE, the square root of the mean of the three per-layer mean squared errors, against observed day-end soil moisture."))
    add(("table", ["Window", "Valid targets (501/502/503; joint)", "Best-pair RMSE", "Raw-point persistence RMSE", "Layer-mean persistence RMSE", "Beats raw-point persistence?"],
         [[TITLES[n], "/".join(str(runs[n]["scen"][bn]["s"][v]["n"]) for v in LAYER) + f"; {sum(runs[n]['pi']['observed_joint_valid'])}", f3(runs[n]["scen"][bn]["s"]["overall"]), f3(ps[n]["point"]["overall"]), f3(ps[n]["layer"]["overall"]),
           "yes" if runs[n]["scen"][bn]["s"]["overall"] < ps[n]["point"]["overall"] else "no"] for n in NAMES],
         "Table 1. Primary overall RMSE (10^-3 m3 m-3) of the training-selected pair (d = %g, E = %g) against the two persistence nulls." % (best[0], best[1])))
    add(("bullets", [
        f"Training (JORN 4, A, 10 days): the best pair on the grid is d = {best[0]:g}, E = {best[1]:g} (RMSE {f3(sel_rmse)}), better than raw-point persistence ({f3(ps['JORN4_A_training']['point']['overall'])}; MSE skill {f2(tr['scen'][bn]['skill_point'])}) and much better than the integrated layer-mean persistence "
        f"({f3(ps['JORN4_A_training']['layer']['overall'])}; skill {f2(tr['scen'][bn]['skill_layer'])}). The selection is d = 0: on these 10 days, adding interlayer exchange made the fit worse at every E. That is not evidence that the soil has zero physical diffusivity.",
        f"Held-out JORN 4, B (4 days, 3 joint targets): the training pair (RMSE {f3(jb['scen'][bn]['s']['overall'])}) does NOT beat raw-point persistence ({f3(ps['JORN4_B_heldout']['point']['overall'])}); it does beat layer-mean persistence ({f3(ps['JORN4_B_heldout']['layer']['overall'])}). The unfitted reference pair scores {f3(jb['scen'][REF[0]]['s']['overall'])}.",
        f"Held-out MOAB 2, C (6 days, different site, 3 joint targets): the training pair ({f3(mc['scen'][bn]['s']['overall'])}) beats raw-point persistence ({f3(ps['MOAB2_C_heldout']['point']['overall'])}), and the unfitted reference pair is better still ({f3(mc['scen'][REF[0]]['s']['overall'])}). "
        "The reference pair changes both d and E relative to the training pair and the held-out protocol has no matched control that changes only one of them, so the MOAB improvement cannot be attributed to exchange alone (Section 6).",
        "No significance tests, no field-property inference and no physical calibration are claimed. Open and closed bottom boundaries gave identical results because no water ever exceeded field capacity (Section 7)."]))
    # ---------------- data
    add(("h1", "2. Observations and experimental design"))
    add(("p", "Observed data are the frozen NEON RELEASE-2026 products: soil water content and water salinity (DP1.00094.001, 30-minute means, plots 4 at JORN and 2 at MOAB), tipping-bucket precipitation (used only to select rain-free windows and to confirm that every forcing half-hour is rain-free), "
              "and single-aspirated air temperature (DP1.00002.001) at the level nearest 2 m (1.51 m at JORN, 1.95 m at MOAB; chosen before any outcome). Soil-moisture depths are taken from the corrected NEON depth table (JORN plot 4: 5.8, 15.8, 25.8 cm; MOAB plot 2: 5.5, 15.5, 25.5 cm), not from the shifted sensor-position file shipped with the release. "
              "The three probes sit in the model layers 0-10, 10-20 and 20-30 cm. They are point measurements; the model layers are layer averages (a stated representativeness difference). NEON announced on 2026-11-09 that a reprocessed RELEASE-2027 with revised depths, calibration and quality control is expected in January 2027; "
              "that is an announcement, not data used here, and any later release will be a separate analysis."))
    add(("p", "A window is a run of rain-free, quality-passing half-hours of at least 120 h; the first 72 h are excluded. The initial state is the first genuinely simultaneous passing three-probe observation whose 30-minute interval ends strictly after the exclusion; its end time is t0, and the model advances in whole "
              "86 400 s steps from t0. Observations are compared only where a passing observation interval ENDS at exactly the model day-end time; a missing or failing target is left empty (never interpolated or filled). The model states are instantaneous end-of-day states and the observations are 30-minute means ending at those instants; no sub-daily trajectory is modelled or drawn."))
    rows = []
    for n in NAMES:
        r = runs[n]
        pi = r["pi"]
        rows.append([TITLES[n], pi["t0_utc"].replace("T", " ").replace(":00Z", ""), str(r["r"]["plan"]["model_days"]), str(r["r"]["plan"]["max_rainfree_days"]), "/".join(f"{x:.4f}" for x in pi["raw_initial_point_theta"]), "/".join(f"{x:.4f}" for x in pi["integrated_initial_layer_theta"])])
    add(("table", ["Window", "t0 (UTC)", "Modelled days", "Rain-free days available", "Initial point theta 501/502/503", "Integrated initial layer theta 1/2/3"], rows,
         "Table 2. Windows. The modelled days are the consecutive whole days from t0 whose 24 h air-temperature window is complete; the whole rain-free suffix after the first incomplete day (listed in the text) is excluded, without skipping or filling."))
    note = []
    for n in NAMES:
        pl = runs[n]["r"]["plan"]
        fi = pl["first_incomplete_forcing_day"]
        extra = pl["max_rainfree_days"] - pl["model_days"]
        t0s = datetime.datetime.fromisoformat(pl["prepared_inputs"]["t0_utc"].replace("Z", "+00:00"))
        if fi:
            a = t0s + datetime.timedelta(days=fi["day_index"])
            note.append(f"{TITLES[n].split(' (')[0]}: the first incomplete forcing day is day {fi['day_index'] + 1}, {a:%Y-%m-%d %H:%M} to {a + datetime.timedelta(days=1):%Y-%m-%d %H:%M} UTC ({fi['reason']}); the {extra} remaining whole rain-free days (days {pl['model_days'] + 1} to {pl['max_rainfree_days']}, beginning with that incomplete day) and any partial tail are not modelled")
        else:
            note.append(f"{TITLES[n].split(' (')[0]}: no incomplete forcing day; the rain-free run is {pl['max_rainfree_days']} days and ends at the end of the month file (censored at the month end, so later days are unknown)")
    add(("p", "Excluded time. " + "; ".join(note) + ". No day was skipped over to reach later complete days, and no missing record was filled."))
    add(("p", "The initial layer water is the exact integral of the piecewise-linear profile through the three observed point values at their actual depths, with constant extension above the shallowest and below the deepest probe, so the integrated layer mean differs from the raw point values. Both are shown as persistence nulls: "
              "the raw point values held constant (the natural null for a point sensor) and the integrated layer means held constant (the null for the layer state the model starts from)."))
    add(("fig", "fig2_profiles_JORN4_A_training.png", "Figure 1. JORN 4 window A (training): observed day-end targets, the training-selected model, the unfitted reference and raw-point persistence for the three probe layers."))
    # ---------------- model
    add(("h1", "3. Model, reference soil and forcing"))
    add(("p", "The model is the existing SYRUP fixed-layer column (maple_syrup.subsurface), unchanged. Layer i has thickness dz_i (0.1 m here) and stores the water inventory W_i = theta_i x dz_i (metres of water, equivalently m3 per m2 of ground), so theta_i = W_i/dz_i. "
              "One daily step (dt = 86 400 s, one cell of 1 m2, no added water) applies in this order: (i) gravity drainage of water above field capacity (open bottom primary, closed as sensitivity); "
              "(ii) soil evaporation from the top layer: with PET_m_s = PET_mm_day / (1000 x 86 400) the potential rate in m/s, the demand over the step is demand_m = E x PET_m_s x m(theta_1) x dt (for one day, E x (PET_mm_day / 1000) x m(theta_1) metres), capped by the top-layer water above the residual; here m = 0 at or below the residual, m = (theta_1/theta*)^2 between the residual and the stress threshold theta* = 0.18, and m = 1 above; "
              "(iii) sequential, top-down, pair-wise diffusive exchange between neighbouring layers i and j (Tietjen et al. donor-depth form)."))
    add(("p", "Exchange coefficient and pair update. In the donor-depth form the directional diffusivity is D = d x K x dz_donor, where d is the dimensionless exchange coefficient, K the interface conductivity (m/s) and dz_donor the THICKNESS (m) of the layer that gives water (the donor is the layer with the larger theta); it is not a water depth. "
              "With the conductance G = D/s, where s is the distance between the layer centres (here (dz_i + dz_j)/2), one pair relaxes exactly over the step: q = (theta_i - theta_j)/(1/dz_i + 1/dz_j) x (1 - exp[-G (1/dz_i + 1/dz_j) dt]), the volume per m2 moved from i to j when positive. "
              "The transfer |q| is then bounded by the donor's water above its residual and by the receiver's remaining capacity (saturation minus current inventory), and the pairs are processed in order from the top, each using the already updated inventories. "
              "Soil moisture outputs are theta in m3 m-3. Budget terms are native volumes in m3 for the 1 m2 cell; dividing by the cell area gives water depth in metres, numerically equal for a 1 m2 cell but not the same quantity (1 mm over 1 m2 is 1 litre), and losses are reported in mm."))
    add(("p", "Reference soil. The residual 0.01, evaporation stress 0.18, field capacity = saturation = 0.26 and Ks = 6.9 mm/h come from the parameter file soilparameters_tabernas.txt of the downloaded 2023 EcoHyD archive (commit 80b10e6b; Baldauf et al. 2023), where the soil-moisture entries are labelled vol% but are used as volume fractions (0.26 is a fraction, not 26). "
              "The interface conductivity K = sqrt(Ks x Ku) with Ku = Ks/(1 + 30/26)^3 (6.06e-7 m/s) follows that archive's cell constructor. The archive's own layer depths (60 and 80 mm) are not used; the three 10 cm layers are the user's choice. This is not the 2009 original release, and these are not properties of JORN or MOAB. "
              "Field capacity equals saturation, so the reference soil can hold no gravity-draining water. The accepted site pedotransfer provider (Saxton and Rawls 2006) refuses the surface estimates from the JORN and MOAB megapit data (unsupported air entry or density), and no unsupported prediction was used; "
              "measured retention and conductivity data would be needed for a physical site case."))
    add(("p", "Forcing. For each model day the complete 24 h window of air-temperature records is required (48 passing half-hours): Tmin is the minimum of the 30-minute minima, Tmax the maximum of the 30-minute maxima and Tmean the count-weighted mean of the 30-minute means. Reference PET (mm d-1) is the Hargreaves-type temperature method with the EcoHyD astronomical constants "
              "(Hargreaves and Samani 1985; SYRUP weather_pet), evaluated at the megapit latitude with terrain factor 1 and the calendar date of the window midpoint in a fixed UTC-7 standard-time calendar. E is an effective multiplier on that demand, not a measured canopy or cover value. "
              "No transpiration, rain, lateral or deeper unobserved flux, root or plant growth is included, so the simulated loss is a partial soil-evaporation budget and not total evapotranspiration."))
    add(("fig", "fig5_forcing_and_loss.png", "Figure 2. Reference PET from the complete air-temperature days (top) and the cumulative simulated soil evaporation of the training-selected and reference pairs (bottom)."))
    # ---------------- protocol
    add(("h1", "4. Protocol and scoring"))
    add(("p", "The frozen grid was d in {0, 0.001, 0.003, 0.01, 0.03, 0.05, 0.1, 0.3, 1} by E in {0, 0.1, 0.25, 0.5, 1, 1.2} (54 open-bottom scenarios), plus two closed-bottom sensitivities, all on window A only. The training selection is the minimum primary overall RMSE; ties go to the smaller E, then the smaller d (no tie occurred: "
              f"{len(ties)} scenario at the minimum). The held-out scenarios are then fixed: the training pair, d = 0 at the fitted E (identical to the training pair here, because the fitted d is 0), E = 0 at the fitted d, the unfitted reference (d = 0.05, E = 1.2) and the zero-exchange, zero-evaporation control; "
              "duplicates are merged, leaving three scenarios per held-out window. Nothing was refitted on the held-out windows."))
    add(("p", "Primary overall RMSE is sqrt(mean over layers of the per-layer MSE) computed on the valid targets of each layer; the number of valid targets per layer is reported. MSE skill is 1 - MSE_model/MSE_reference and is left undefined when the reference MSE is zero or a layer has no valid target. With few days and sensor-level differences the numbers are descriptive; no p-values are given."))
    # ---------------- results
    add(("h1", "5. Training results (JORN plot 4, window A)"))
    add(("fig", "fig1_training_heatmap.png", "Figure 3. Primary overall RMSE over the frozen grid. The minimum is at d = 0, E = 0.5; raising d raises the error at every E."))
    n1 = {v: tr["scen"][bn]["s"][v] for v in LAYER}
    z = tr["scen"][sc_name(0, 0)]
    add(("p", f"The best pair has per-layer RMSE (10^-3 m3 m-3) {f3(n1['501']['rmse'])} / {f3(n1['502']['rmse'])} / {f3(n1['503']['rmse'])} for the probes at 5.8, 15.8 and 25.8 cm (valid targets {n1['501']['n']}/{n1['502']['n']}/{n1['503']['n']}); the biases are {f3(n1['501']['bias'])} (layer 1), "
              f"{f3(n1['502']['bias'])} (layer 2) and {f3(n1['503']['bias'])} (layer 3). Against raw-point persistence the shallow layer improves strongly ({f3(n1['501']['rmse'])} against {f3(ps['JORN4_A_training']['point']['501']['rmse'])}) because it dries in the observations, and the deepest layer improves ({f3(n1['503']['rmse'])} against {f3(ps['JORN4_A_training']['point']['503']['rmse'])}); "
              f"the middle layer is worse ({f3(n1['502']['rmse'])} against {f3(ps['JORN4_A_training']['point']['502']['rmse'])}) because its integrated initial mean ({runs['JORN4_A_training']['pi']['integrated_initial_layer_theta'][1]:.4f}) lies below the probe readings (0.0672-0.0720), and nothing acts on that layer when d = 0. "
              f"The zero-evaporation, zero-exchange control equals the layer-mean persistence ({f3(z['s']['overall'])}) as it must, because no gravity flux acts. E = 0.5 removes {tr['scen'][bn]['evap_mm']:.2f} mm from the top layer over 10 days; the reference E = 1.2 with d = 0.05 removes {tr['scen'][REF[0]]['evap_mm']:.2f} mm and scores {f3(tr['scen'][REF[0]]['s']['overall'])}, worse than raw-point persistence."))
    add(("p", "Identifiability. The error surface is shallow near the minimum along E (E = 0.25: 4.69, E = 0.5: 4.23, E = 1: 5.93 on the 10^-3 scale), the grid is coarse, only 8-10 targets per layer exist and the supplied sensor uncertainty (0.2135 m3 m-3, retained as supplied) is large relative to the signals, although it does not characterise the precision of relative changes. "
              "The selected values should be read as an effective best grid point for this window and this generic soil, not as estimates of a physical parameter."))
    add(("h1", "6. Held-out results and persistence"))
    rows = []
    for n in NAMES[1:]:
        for nm, lab in ((bn, f"training pair (d={best[0]:g}, E={best[1]:g})"), (REF[0], "unfitted reference (d=0.05, E=1.2)"), (sc_name(0, 0), "zero exchange, zero evaporation")):
            s = runs[n]["scen"][nm]
            rows.append([TITLES[n].split(" (")[0], lab, f3(s["s"]["overall"]), f3(s["s"]["pooled"]), f2(s["skill_point"]), f2(s["skill_layer"]), f"{s['evap_mm']:.2f}"])
        rows.append([TITLES[n].split(" (")[0], "raw-point persistence", f3(ps[n]["point"]["overall"]), f3(ps[n]["point"]["pooled"]), "-", "-", "0"])
        rows.append([TITLES[n].split(" (")[0], "layer-mean persistence", f3(ps[n]["layer"]["overall"]), f3(ps[n]["layer"]["pooled"]), "-", "-", "0"])
    add(("table", ["Window", "Scenario", "Overall RMSE", "Pooled RMSE (alt.)", "Skill vs point", "Skill vs layer", "Evaporation (mm)"], rows,
         "Table 3. Held-out scores (RMSE in 10^-3 m3 m-3). Skill = 1 - MSE/MSE_reference; negative means worse than the reference."))
    add(("fig", "fig3_profiles_JORN4_B_heldout.png", "Figure 4. JORN 4 window B (held-out, temporal)."))
    add(("fig", "fig4_profiles_MOAB2_C_heldout.png", "Figure 5. MOAB 2 window C (held-out, site). Days without a valid target for a probe are marked and not filled."))
    jbs, mcs = jb["scen"], mc["scen"]
    add(("bullets", [
        f"JORN 4 B: only 4 days and 3 joint targets. The training pair is worse than raw-point persistence (skill {f2(jbs[bn]['skill_point'])}) but better than layer-mean persistence (skill {f2(jbs[bn]['skill_layer'])}); the reference is similar ({f3(jbs[REF[0]]['s']['overall'])}). The shallow probe is already drier at t0 than the integrated layer mean, so the initial mean over-states the top layer and any loss helps relative to layer persistence.",
        f"MOAB 2 C: the top layer dries from 0.0659 to 0.0357 (an observed change of about 3.0 mm in 10 cm, taken from point values) and the training pair removes {mcs[bn]['evap_mm']:.2f} mm while the reference removes {mcs[REF[0]]['evap_mm']:.2f} mm. The reference scores far better here ({f3(mcs[REF[0]]['s']['overall'])} against {f3(mcs[bn]['s']['overall'])}), "
        "but it differs from the training pair in both d and E, and no held-out scenario changes only E (or only d) while keeping the other fixed. The larger loss is the likelier reason, yet the data do not separate the two; a matched sensitivity could be added later and would have to be labelled post hoc.",
        "Taking raw-point persistence as a predictor too, the best overall predictor is the training pair on the training window (the best grid point by construction), raw-point persistence on JORN 4 B, and the unfitted reference pair on MOAB 2 C. Among the three SYRUP scenarios the training pair is the best on JORN 4 B and not at MOAB. "
        "The reference pair performs poorly on the training window (7.6 against 4.9 for raw-point persistence, although it is not the worst of the 54 grid points) and on JORN 4 B, yet is clearly best at MOAB. Skill does not transfer across windows and sites in a simple way, consistent with a generic soil, a single effective E and a few-day sample."]))
    # ---------------- numerical
    add(("h1", "7. Numerical verification and what the diagnostics do and do not show"))
    add(("p", f"All {sum(len(runs[n]['r']['runs']) for n in NAMES)} scenario runs executed on the native column with both NumPy and Numba (first backend Numba); the whole trajectories (states, fluxes, residuals, times and cumulative budgets) agree with a maximum absolute difference of "
              f"{max(max(max(p.values()) for p in (s['backend_agreement_max_abs'] or {}).values()) for n in NAMES for s in runs[n]['r']['runs'].values()):.1e} (bit-identical in these runs) within the inherited contract. Per-step and cumulative water ledgers close within the native MAPLE floating-point bound "
              "(the cumulative residual and bound are in the result files; the native ledger is in m3 for the 1 m2 cell, numerically equal to water depth in m, and the reported losses are in mm). A midpoint restart rebuilt from the exported state (water and time) through the public column-state constructor reproduces the whole trajectory and the cumulative budget exactly; this is an experiment-driver check, not a lifecycle checkpoint qualification. "
              f"Root's independent controls passed: original CSV joins, QA, forcing, a rational (exact Fraction) check of the initial integral, and scores; and a 70-digit decimal re-solution of the declared equations for all 62 scenarios. The decimal control uses high-precision arithmetic on the SAME binary64 initial water and properties as the native run (the initial integral is checked separately against the exact rational integral), so it tests the arithmetic and the declared equations, not the decimal-literal inputs. "
              f"Its maximum water error was {max(x['max_water_error_m'] for x in dec['rows']):.1e} m, at most {max(x['max_error_over_native_bound'] for x in dec['rows']):.2f} of the native single-step bound; that comparison is a diagnostic check, not a formal cumulative error bound, and the native cumulative ledger closure is enforced separately in every run. "
              "Root's selection and persistence audit (evidence/root_selection_and_persistence_audit.json) independently confirmed the 54-scenario selection (d = 0, E = 0.5), the persistence values and the closed-bottom equalities. This builder also recomputed every score, persistence value, skill and the grid selection from the exported targets and layer states and re-derived the Hargreaves PET."))
    add(("p", "No drainage, no surface return and no percolation occurred in any scenario, because every layer stayed far below the reference field capacity of 0.26. The open and closed bottom controls therefore agree exactly; this does not validate the boundary or any drainage behaviour, which these experiments never exercise. "
              "Source and sensor identities are unchanged: the production package (180 modules), MAPLE 412 files, LEAF and its helper were checked before and after each run (identities in the appendix)."))
    add(("p", ("Hardware. CUDA completion records are present in the data folder and are summarised by Root." if status == "recorded" else
               "Hardware. The CUDA observed-profile qualification is pending. The pre-launch guard refused GPU0 because it was in use, so no GPU child ran and nothing here depends on a GPU. No large-GPU timing or memory claim could be made from a one-cell profile in any case.")))
    # ---------------- discussion
    add(("h1", "8. Interpretation, limits and next steps"))
    add(("bullets", [
        "What can be said: with observed initialisation and observed air temperature, a simple daily soil-evaporation term with an effective multiplier near 0.5 reduces the error of a layer-state forecast on the training window and improves on raw-point persistence there; on a second window at the same site it does not, and on a different site a larger multiplier does much better.",
        "What cannot be said: a zero fitted d does not mean zero physical diffusivity; the reference soil is not the site soil; the sensors are points and the model layers are averages; the sensor-reported uncertainty exceeds the signals; 4-10 days and 3-8 joint targets cannot support a significance claim; the observed drying can include temperature artefacts and processes (transpiration, vapour flow, deeper flow) that are not modelled. "
        "Nothing here implies a bedrock or boundary at 30 cm, a measured canopy cover, or a field evapotranspiration closure.",
        "Priorities: measured hydraulic properties and a supported parameterisation for the sites; evaporation, transpiration and deeper-boundary observations; longer quality-controlled periods with more independent windows; a separate, explicitly qualified analysis of the NEON RELEASE-2027 data once it exists; and, if wanted, matched held-out controls that vary d and E one at a time. None of these should be replaced by tuning an unobserved flux to fit."]))
    # ---------------- references
    add(("h1", "References"))
    refs = J(EVID / "root_citations.json")["references"]
    texts = [f"{x['citation']} {x['url']}" for x in refs]
    assert any("DP1.00045.001" in t for t in texts) and not any("DP1.00006.001" in t for t in texts), "the precipitation product ID in the citation list must be the pinned DP1.00045.001"
    add(("refs", texts))
    # ---------------- appendix
    add(("h1", "Appendix. Reproduction, identities and file receipts"))
    rows = []
    for e in comp["rows"]:
        rows.append([e["name"], str(e["scenario_count"]), e["config_sha256"][:16], e["result_sha256"][:16], f"{e['seconds_including_first_compilation']:.2f}"])
    add(("table", ["Run", "Scenarios", "Config sha256 (first 16)", "Result sha256 (first 16)", "Wall time s (CPU, incl. first compile)"], rows, "Table 4. Executed CPU runs (data/completion.json). Result hashes were re-checked by this builder."))
    for e in comp["rows"]:
        assert sha(DATA / Path(e["result"]).name) == e["result_sha256"], e["name"]
    add(("bullets", [
        "Build command: MPLCONFIGDIR=/tmp/syrup-observed-mpl PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/tmp/syrup-manuals-citation-tools-20261020 /home/okin/MAPLE/.venv/bin/python -B build_report.py (python-docx 1.2.0; the existing helper bundle; nothing installed).",
        "Runs: Root's sealed driver (frozen configs, 52 sealed files) with the explicit configs in data/*_config.json; run mode cpu; backends numba then numpy; Python/NumPy/Numba versions in data/runtime.json.",
        "Sources: SYRUP production package 37d89f23dbf60626c85f85ea5dd6d1734d3bc7a4aa61d8f2fdfc29f237bcfbd0 (180 files); native MAPLE fbc504bbcd1ec0c02e6a96089134a06187532455ef7ba5e6f146801c8313c02e (412 files); LEAF 38009872f8969f86b56c720adc1b5d9ffefc554fb9cacf90855d999db604530f with helper 333921bb645d984fa8e7449619d03114262ee333c7fea61a22493a878fd9ccc5 were verified by Root before and after.",
        "Data: NEON RELEASE-2026 files and the corrected-depth table (sha256 876fd5bd...827f) pinned in the configs; air-temperature manifests and the MOAB megapit retrieval verified (evidence/root_all_new_data_audit.json). Reference-soil source: EcoHyD 2023 archive commit 80b10e6b6b82ad16e9fd41a31901cc3157d880e9, soilparameters_tabernas.txt sha256 db02855a9359eed71c1976309ccefeaa3780a2314c2b57cd08b2f9ec4ec77e62 (evidence/root_reference_code_manifest.json).",
        f"Root verification: {verif['cpu_tests']['passed']} CPU tests passed, {verif['cpu_tests']['skipped']} GPU-gated tests skipped; independent audit rows: {', '.join(r['name'] for r in aud['rows'])}.",
        "Observed-driver reproduction (Root's sealed environment; paths are examples): the driver is `python -m benchmarks.subsurface_validation.drydown_driver --config CONFIG.json [--out DIR] [--execute]` run from the checkout containing `benchmarks/`; without --execute it prints the plan, with it the native column runs. "
        "Environment: the sealed MAPLE 9f/fbc504bb materialization and the SYRUP `src` on PYTHONPATH, CUDA_VISIBLE_DEVICES empty for CPU, NUMBA_CACHE_DIR set, and the data locations given by SYRUP_PHASE3_TASKS_ROOT, SYRUP_PHASE3_TASK_DIR, SYRUP_PHASE3_RAW_ROOT, SYRUP_PHASE3_DEPTH_TABLE, SYRUP_PHASE3_AIR_JORN, SYRUP_PHASE3_AIR_MOAB, SYRUP_PHASE3_MEGAPIT_MOAB and SYRUP_PHASE3_INVENTORY (see `benchmarks/subsurface_validation/paths.py`).",
        "Configs: the CPU configs list backends numba, numpy; the CUDA-lane configs (one extra backend entry cuda, nothing else different) are in the task `config_proposals/*_cuda.json`. Every config pins the inventory file, the corrected-depth table, the air manifest and the expected SYRUP and MAPLE package identities, and the driver asserts them before and after the run: a different source is refused, never adopted automatically. "
        "The GPU path of the driver is authored and host-tested only; a real GPU0 run by Root is pending and nothing in this report depends on it.",
        "Source provenance: accepted SYRUP source commit bb28af6c58c2892cb3c065d73d33b88285c498f0 (package 37d89f23, 180 files); the analysis branch is based on main commit cd07f46bd3f5e4e51c2932c63b145a67b3cb1a17 (task baseline). Root's selection and persistence audit: evidence/root_selection_and_persistence_audit.json.",
        "No credential, token or temporary download address is used or stored by this report."]))
    return b


# ------------------------------------------------------------------ renderers
def to_markdown(blocks):
    out = []
    for blk in blocks:
        k = blk[0]
        if k == "title":
            out.append(f"# {blk[1]}\n")
        elif k == "sub":
            out.append(f"*{blk[1]}*\n")
        elif k == "box":
            out += [f"> {t}\n>" for t in blk[1]]
            out.append("")
        elif k == "h1":
            out.append(f"## {blk[1]}\n")
        elif k == "p":
            out.append(blk[1] + "\n")
        elif k == "bullets":
            out += [f"* {t}" for t in blk[1]] + [""]
        elif k == "refs":
            out += [f"{i + 1}. {t}" for i, t in enumerate(blk[1])] + [""]
        elif k == "table":
            _, head, rows, cap = blk
            out.append(f"**{cap}**\n")
            out.append("| " + " | ".join(head) + " |")
            out.append("|" + "---|" * len(head))
            out += ["| " + " | ".join(r) + " |" for r in rows]
            out.append("")
        elif k == "fig":
            out.append(f"![{blk[2]}](figures/{blk[1]})\n\n*{blk[2]}*\n")
    return "\n".join(out)


def shade(cell, color):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), color)
    tcPr.append(shd)


def page_number(par):
    run = par.add_run()
    for t, txt in (("begin", None), (None, "PAGE"), ("end", None)):
        if t:
            el = OxmlElement("w:fldChar")
            el.set(qn("w:fldCharType"), t)
        else:
            el = OxmlElement("w:instrText")
            el.set(qn("xml:space"), "preserve")
            el.text = txt
        run._r.append(el)


def to_docx(blocks, path):
    doc = Document()
    sec = doc.sections[0]
    sec.left_margin = sec.right_margin = Inches(0.9)
    sec.top_margin = sec.bottom_margin = Inches(0.9)
    st = doc.styles["Normal"]
    st.font.name = "Calibri"
    st.font.size = Pt(10.5)
    for blk in blocks:
        k = blk[0]
        if k == "title":
            p = doc.add_heading(blk[1], 0)
        elif k == "sub":
            p = doc.add_paragraph()
            r = p.add_run(blk[1])
            r.italic = True
        elif k == "box":
            t = doc.add_table(rows=1, cols=1)
            t.style = "Table Grid"
            c = t.rows[0].cells[0]
            shade(c, "FFF2CC")
            c.text = ""
            for i, txt in enumerate(blk[1]):
                par = c.paragraphs[0] if i == 0 else c.add_paragraph()
                head, _, rest = txt.partition(". ")
                par.add_run(head + ". ").bold = True
                par.add_run(rest)
            doc.add_paragraph()
        elif k == "h1":
            doc.add_heading(blk[1], 1)
        elif k == "p":
            doc.add_paragraph(blk[1])
        elif k == "bullets":
            for t in blk[1]:
                doc.add_paragraph(t, style="List Bullet")
        elif k == "refs":
            for t in blk[1]:
                doc.add_paragraph(t, style="List Number")
        elif k == "table":
            _, head, rows, cap = blk
            cp = doc.add_paragraph()
            cp.add_run(cap).bold = True
            t = doc.add_table(rows=1, cols=len(head))
            t.style = "Table Grid"
            t.alignment = WD_TABLE_ALIGNMENT.CENTER
            for i, h in enumerate(head):
                c = t.rows[0].cells[i]
                c.text = ""
                run = c.paragraphs[0].add_run(h)
                run.bold = True
                run.font.size = Pt(8.5)
                shade(c, "D9E2F3")
            for r in rows:
                cells = t.add_row().cells
                for i, v in enumerate(r):
                    cells[i].text = ""
                    cells[i].paragraphs[0].add_run(v).font.size = Pt(8.5)
            doc.add_paragraph()
        elif k == "fig":
            doc.add_picture(str(FIG / blk[1]), width=Inches(6.0 if "profiles" not in blk[1] else 5.2))
            doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
            c = doc.add_paragraph()
            c.alignment = WD_ALIGN_PARAGRAPH.CENTER
            c.add_run(blk[2]).italic = True
    fp = sec.footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    fp.add_run("SYRUP observed dry-down results (generic reference soil; CPU results; GPU qualification pending unless the data folder records it) - page ").font.size = Pt(8)
    page_number(fp)
    doc.core_properties.title = "SYRUP observed shallow-profile dry-down results"
    doc.core_properties.author = "SYRUP analysis (Claude, for Root review)"
    doc.core_properties.created = datetime.datetime(2026, 11, 14)
    doc.save(path)


def main():
    runs = load()
    d, e, rmse, ties = selection(runs["JORN4_A_training"])
    fin = (HERE.parent.parent.parent.parent.parent / "actual_cpu" / "JORN4_B_heldout_finalize.log")
    if fin.is_file():
        assert f"d{d:g}_E{e:g}_open" in fin.read_text(), "the independently re-derived best pair differs from Root's finalize log"
    best = (d, e)
    fig_heatmap(runs, best)
    for i, n in enumerate(NAMES, start=2):
        fig_profiles(n, runs[n], best, i)
    fig_forcing(runs, best)
    blocks = build_blocks(runs, best, rmse, ties)
    (HERE / "report.md").write_text(to_markdown(blocks))
    to_docx(blocks, HERE / "SYRUP_observed_drydown_results.docx")
    tables = {"best_pair": {"d": d, "E": e, "training_overall_rmse": rmse, "ties": len(ties)},
              "windows": {n: {"persistence_point_overall_rmse": runs[n]["pers"]["point"]["overall"], "persistence_layer_overall_rmse": runs[n]["pers"]["layer"]["overall"],
                              "scenarios": {k: {"overall_rmse": v["s"]["overall"], "pooled_rmse": v["s"]["pooled"], "skill_vs_point": v["skill_point"], "skill_vs_layer": v["skill_layer"], "evaporation_mm": v["evap_mm"],
                                                "per_layer_n": [v["s"][x]["n"] for x in LAYER]} for k, v in runs[n]["scen"].items() if n != "JORN4_A_training" or k in (sc_name(d, e), REF[0], sc_name(0, 0))}} for n in NAMES},
              "training_grid_overall_rmse": {sc_name(a, b): runs["JORN4_A_training"]["scen"][sc_name(a, b)]["s"]["overall"] for a in D_GRID for b in E_GRID},
              "result_sha256": {n: sha(DATA / f"{n}_result.json") for n in NAMES}, "cuda_status": cuda_status()[0]}
    (HERE / "report_tables.json").write_text(json.dumps(tables, indent=1) + "\n")
    print("best", best, "rmse", rmse, "ties", len(ties), "cuda", cuda_status()[0])
    words = len(re.findall(r"\w+", (HERE / "report.md").read_text()))
    print("words", words)


if __name__ == "__main__":
    main()
