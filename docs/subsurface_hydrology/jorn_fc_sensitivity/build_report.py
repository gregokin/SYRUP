"""Builds the JORN 160-day uniform field-capacity x stress sensitivity report: figures, CSV tables, editable Markdown, verification note, manifest and the Word document.

    PYTHONPATH=<tasks>/site_hydraulics_interface_comparison_20261117/report_tools MPLCONFIGDIR=/tmp/syrup-fc-report PYTHONDONTWRITEBYTECODE=1 /home/okin/MAPLE/.venv/bin/python -B build_report.py

DOCUMENTS ONLY. It refuses to start unless Root's `root_report_ready.json` says PASS and `root_run_seal.json` exists. It re-hashes every sealed file before and after the build, recomputes scores, period scores, budgets,
endpoints and the January deep-layer response independently from the daily rows and compares them with Root's audit files, runs no model, fits nothing and uses no network or GPU.
Declared input: FC_TASK (this task folder; default = the folder four levels above this file). Writes only inside its own folder.
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
import sys
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
from docx.shared import Inches, Pt, RGBColor, Twips

HERE = Path(__file__).resolve().parent
TASK = Path(os.environ.get("FC_TASK", str(HERE.parents[3])))
TASKS = TASK.parent
PRIOR = TASKS / "jorn_sr33_160day_20261120"
FIG, TAB, DATA = HERE / "figures", HERE / "tables", HERE / "data"
DOCX = HERE / "SYRUP_JORN_160day_uniform_FC_sensitivity_20261120.docx"
FCS = [9, 10, 11, 12, 13, 14, 15]
SLAB = {"1": 0.2017, "1p5": 0.2017 * 1.5}
SK = tuple(SLAB)
KEYS = [f"fc{p:02d}_s{s}" for s in SLAB for p in FCS]
FAMS = {s: [f"fc{p:02d}_s{s}" for p in FCS] for s in SLAB}
LV = ["501", "502", "503"]
DEPTH = {"501": "5.8 cm sensor (layer 0-10 cm)", "502": "15.8 cm sensor (layer 10-20 cm)", "503": "25.8 cm sensor (layer 20-30 cm)"}
SYR, MAP = "4e38a472e268a56491b788d5b1271cc3c3da23bb1cc3221a8c59d90f3cc864d2", "fbc504bbcd1ec0c02e6a96089134a06187532455ef7ba5e6f146801c8313c02e"
BASE_SHA = "f024054f5198ec8358c95f7a33683e6228084e6af246f9790cde586d437f3d2a"
_pal = plt.cm.viridis(np.linspace(0.02, 0.88, 7))
FCCOL = {p: _pal[i] for i, p in enumerate(FCS)}
FCLS = {9: "-", 10: "--", 11: "-.", 12: ":", 13: "-", 14: "--", 15: "-."}
SC = {"1": "#1F5FA8", "1p5": "#C2185B"}
SM = {"1": "-", "1p5": "--"}


def fcp(k):
    return int(k[2:4])


def sk(k):
    return k.split("_s")[1]


def LAB(k):
    return f"FC {fcp(k)}%, stress {SLAB[sk(k)]:.5g}"


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
    S = lambda k, r: TASK / "study" / k / r
    R = {"ready": ready, "seal": seal, "audit": J(TASK / "root_result_audit.json"), "per": J(TASK / "root_period_scores.json"), "deep": J(TASK / "root_deep_wetting.json"), "inp": J(TASK / "root_independent_inputs.json"),
         "param": J(TASK / "root_parameter_verification.json")}
    for name in ("root_result_audit.json", "root_period_scores.json", "root_deep_wetting.json", "root_independent_inputs.json", "root_parameter_verification.json"):
        assert str(TASK / name) in seal, name
    for k in KEYS:
        R[k] = {"res": J(S(k, "actual_numba/result.json")), "cfg": J(S(k, "case_numba.json")), "tm": J(S(k, "actual_numba/timing.json")), "np": J(S(k, "actual_numpy/result.json")), "d900": J(S(k, "actual_numba_dt900/result.json")),
                "wall": J(S(k, "actual_numba_runs/result.json")), "dec": J(S(k, "root_decimal_verification.json")), "src": {v: (J(S(k, f"actual_{v}_runs/sources_before.json")), J(S(k, f"actual_{v}_runs/sources_after.json"))) for v in ("numba", "numpy", "numba_dt900")}}
        for rel in ("actual_numba/result.json", "actual_numba/timing.json", "case_numba.json", "actual_numpy/result.json", "actual_numba_dt900/result.json", "root_decimal_verification.json", "actual_numba_runs/result.json"):
            assert str(S(k, rel)) in seal, (k, rel)
    R["base"] = J(PRIOR / "study/k1/case_numba.json")
    assert sha(PRIOR / "study/k1/case_numba.json") == BASE_SHA
    rv, rs = TASK / "root_restart_verification.json", TASK / "root_restart_seal.json"
    R["restart"], R["rseal"] = (J(rv), None) if rv.exists() else (None, None)
    if R["restart"] is not None and not rs.exists():
        assert str(rv) in seal, "restart receipt must be sealed"
    if R["restart"] is not None and rs.exists():
        R["rseal"] = J(rs)["files"]
        restart_seal_check(R["rseal"])
    return R


def restart_seal_check(files):
    for p, rec in files.items():
        b = Path(p).read_bytes()
        assert hashlib.sha256(b).hexdigest() == rec["sha256"] and len(b) == rec["bytes"], p


def prep(R):
    k0 = KEYS[0]
    r0 = R[k0]["res"]["rows"]
    R["obs"] = np.array([[np.nan if v is None else v for v in r["obs_theta"]] for r in r0])
    for k in KEYS:
        rows = R[k]["res"]["rows"]
        R[k]["theta"] = np.array([r["layer_theta"] for r in rows])
        other = np.array([[np.nan if v is None else v for v in r["obs_theta"]] for r in rows])
        assert np.array_equal(np.nan_to_num(other, nan=-1), np.nan_to_num(R["obs"], nan=-1))
        assert R[k]["res"]["plan"]["initial"]["layer_theta"] == R[k0]["res"]["plan"]["initial"]["layer_theta"]
    R["theta0"] = np.array(R[k0]["res"]["plan"]["initial"]["layer_theta"])
    R["obs0"] = np.array(R[k0]["res"]["plan"]["initial"]["point_theta"])
    R["close"] = [dt.date.fromisoformat(r["date"]) + dt.timedelta(days=1) for r in r0]
    R["start"] = dt.date.fromisoformat(r0[0]["date"])
    R["sat"] = np.array(R[k0]["cfg"]["layers"]["theta_saturation"])
    R["early"] = np.array([r["date"] < "2024-03-01" for r in r0])
    R["base_fc"] = np.array(R["base"]["layers"]["theta_field_capacity"])
    return R


def stats(e):
    return {"n": int(e.size), "rmse": 100 * math.sqrt(float(np.mean(e**2))), "bias": 100 * float(e.mean())}


def err(R, k, j, kind, mask=None):
    th, ob = R[k]["theta"][:, j], R["obs"][:, j]
    ok = np.isfinite(ob) if mask is None else (np.isfinite(ob) & mask)
    if kind == "raw":
        return th[ok] - ob[ok]
    if kind == "aligned":
        return (th[ok] - R["theta0"][j]) - (ob[ok] - R["obs0"][j])
    return -(ob[ok] - R["obs0"][j])


def scores(R, k):
    out = {}
    for kind in ("raw", "aligned", "persistence"):
        allv = [err(R, k, j, kind) for j in range(3)]
        out[kind] = {v: stats(allv[j]) for j, v in enumerate(LV)} | {"pooled": stats(np.concatenate(allv))}
    return out


def periods(R, k):
    out = {}
    for name, mask in (("before_March1", R["early"]), ("March1_onward", ~R["early"])):
        d = {}
        for kind in ("raw", "aligned"):
            per = [err(R, k, j, kind, mask) for j in range(3)]
            d[kind] = {"501": stats(per[0]), "502": stats(per[1]), "503": stats(per[2]), "upper_two": stats(np.concatenate(per[:2])), "all_three": stats(np.concatenate(per))}
        out[name] = d
    return out


def budget(R, k):
    rows = R[k]["res"]["rows"]
    s = lambda key: 1000 * sum(r[key] for r in rows)
    perc = np.array([r["percolation_m3"] for r in rows]) * 1000
    dif = np.array([r["diffusion_m3"] for r in rows]) * 1000
    th = R[k]["theta"]
    return {"rain": s("rain_m3"), "intake": s("intake_m3"), "runoff": s("runoff_pool_m3") + s("surface_return_m3"), "drain": s("drainage_m3"), "evap": s("evaporation_m3"), "final": 1000 * rows[-1]["storage_m3"],
            "grav": perc.sum(0).tolist(), "diff_down": dif.sum(0).tolist(), "peak": (th.max(0) * 100).tolist(), "final_theta": (th[-1] * 100).tolist(), "th_series": th}


def deep_events(R, k):
    rows = R[k]["res"]["rows"]
    get = lambda c: next(x for x in rows if x["nominal_close_utc"] == c)
    before, after, obsday = get("2024-01-23T07:00:00Z"), get("2024-01-24T07:00:00Z"), get("2024-01-25T07:00:00Z")
    assert after["date"] == "2024-01-23" and obsday["obs_theta"][2] is not None and after["obs_theta"][2] is None
    peak = max(rows, key=lambda x: x["layer_theta"][2])
    return {"pre": 100 * before["layer_theta"][2], "post": 100 * after["layer_theta"][2], "post_all": [100 * x for x in after["layer_theta"]], "jan25_model": 100 * obsday["layer_theta"][2], "jan25_obs": 100 * obsday["obs_theta"][2],
            "peak": 100 * peak["layer_theta"][2], "peak_close": peak["nominal_close_utc"], "storm_grav": [1000 * x for x in after["percolation_m3"]], "storm_drain": 1000 * after["drainage_m3"]}


def endpoints(R, k):
    rows = R[KEYS[0]]["res"]["rows"]
    out = []
    for j, v in enumerate(LV):
        i = int(np.flatnonzero(np.isfinite(R["obs"][:, j]))[-1])
        out.append({"level": v, "index": i, "date": rows[i]["date"], "close": rows[i]["nominal_close_utc"], "obs": float(R["obs"][i, j] * 100), "model": float(R[k]["theta"][i, j] * 100)})
    return out


def review(R):
    """Independent recomputation from the daily rows against Root's audit, period scores, deep-wetting audit and controls; every assertion aborts the build."""
    A = {r["key"]: r for r in R["audit"]["records"]}
    P = {r["key"]: r for r in R["per"]["records"]}
    Dp = {r["key"]: r for r in R["deep"]["records"]}
    assert [r["key"] for r in R["audit"]["records"]] == KEYS and R["audit"]["status"] == "PASS" and R["per"]["status"] == "PASS" and R["deep"]["status"] == "PASS" and R["param"]["status"] == "PASS"
    obs_inp = R["inp"]["observations"]
    for j, v in enumerate(LV):
        want = [np.nan if x is None else x for x in obs_inp[v]["day_end_values"]]
        assert np.array_equal(np.nan_to_num(np.array(want, dtype=float), nan=-1), np.nan_to_num(R["obs"][:, j], nan=-1)), v
    assert [int(np.isfinite(R["obs"][:, j]).sum()) for j in range(3)] == [56, 147, 79]
    S, B, PS, EP, DE = {}, {}, {}, {}, {}
    base_cfg = R["base"]
    for k in KEYS:
        res, rows, cfg = R[k]["res"], R[k]["res"]["rows"], R[k]["cfg"]
        S[k], B[k], PS[k], EP[k], DE[k] = scores(R, k), budget(R, k), periods(R, k), endpoints(R, k), deep_events(R, k)
        rec = A[k]
        fc, stress = fcp(k) / 100.0, SLAB[sk(k)]
        # the case really is the requested uniform override on the accepted base
        assert cfg["layers"]["theta_field_capacity"] == [fc, fc, fc] and abs(cfg["layers"]["theta_evaporation_stress"] - stress) < 1e-15 and rec["field_capacity"] == [fc, fc, fc]
        assert cfg["physics"] == base_cfg["physics"] and cfg["matrix"] == base_cfg["matrix"] and cfg["layers"]["interface_conductivity_m_s"] == base_cfg["layers"]["interface_conductivity_m_s"] and cfg["layers"]["theta_saturation"] == base_cfg["layers"]["theta_saturation"]
        for v in LV:
            for kind, rk in (("raw", "raw"), ("aligned", "offset_aligned"), ("persistence", "persistence_raw")):
                assert math.isclose(S[k][kind][v]["rmse"], rec["scores"][v][rk]["rmse_pp"], rel_tol=1e-9, abs_tol=1e-12) and S[k][kind][v]["n"] == rec["scores"][v][rk]["n"]
        assert math.isclose(S[k]["raw"]["pooled"]["rmse"], rec["pooled"]["raw"]["rmse_pp"], rel_tol=1e-9) and math.isclose(S[k]["aligned"]["pooled"]["rmse"], rec["pooled"]["offset_aligned"]["rmse_pp"], rel_tol=1e-9)
        b, bud = B[k], rec["budgets_mm"]
        assert math.isclose(b["evap"], bud["evaporation"], rel_tol=1e-9) and math.isclose(b["drain"], bud["bottom_drainage"], rel_tol=1e-9, abs_tol=1e-12) and math.isclose(b["final"], bud["final_storage"], rel_tol=1e-9)
        assert all(math.isclose(x, y, rel_tol=1e-9, abs_tol=1e-12) for x, y in zip(b["grav"], bud["gravity_transfers"])) and all(math.isclose(x, y, rel_tol=1e-9, abs_tol=1e-12) for x, y in zip(b["diff_down"], bud["net_diffusive_transfers_down_positive"]))
        assert abs(22.9424 + b["intake"] - b["evap"] - b["drain"] - b["final"]) < 1e-9 and math.isclose(b["intake"], 25.088, rel_tol=1e-12) and math.isclose(b["rain"], 25.088, rel_tol=1e-12) and b["runoff"] == 0.0
        assert math.isclose(bud["initial_storage"], 22.9424, rel_tol=1e-12)                      # the initial water is NOT rescaled with the field capacity
        assert res["complete"] and res["days_done"] == 160 and res["physical_end_t_s"] == 13923000.0 and res["added_rain_hours"] == 27.5 and rows[-1]["nominal_close_utc"] == "2024-06-26T07:00:00Z"
        assert all(r["n_ponded_steps"] == 0 and r["n_onsets"] == 0 and r["room_rejected_m3"] == 0 and r["surface_return_m3"] == 0 for r in rows) and res["ledger"]["n_matrix_steps"] == 55
        for pname in ("before_March1", "March1_onward"):
            for sub in ("upper_two", "all_three", "501", "502", "503"):
                for kind in ("raw", "aligned"):
                    got, want = PS[k][pname][kind][sub], P[k]["periods"][pname][sub][kind]
                    assert got["n"] == want["n"] and math.isclose(got["rmse"], want["rmse_pp"], rel_tol=1e-9, abs_tol=1e-12) and math.isclose(got["bias"], want["bias_pp"], rel_tol=1e-9, abs_tol=1e-12), (k, pname, sub, kind)
        assert [PS[k]["before_March1"]["raw"]["upper_two"]["n"], PS[k]["March1_onward"]["raw"]["upper_two"]["n"]] == [84, 119]
        for e, w in zip(EP[k], P[k]["endpoints"]):
            assert e["level"] == w["level"] and e["close"] == w["close_utc"] and math.isclose(e["obs"], 100 * w["observed_theta"], abs_tol=1e-9) and math.isclose(e["model"], 100 * w["model_theta"], rel_tol=1e-12)
        assert [e["close"] for e in EP[k]] == ["2024-03-15T07:00:00Z", "2024-06-25T07:00:00Z", "2024-06-26T07:00:00Z"]
        dw, de = Dp[k], DE[k]
        assert dw["fc_percent_all_layers"] == fcp(k) and math.isclose(dw["Jan25_observed_deep_percent"], de["jan25_obs"], rel_tol=1e-9) and abs(de["jan25_obs"] - 6.34) < 1e-9
        for a_, b_ in ((dw["prestorm_layer_theta_percent"][2], de["pre"]), (dw["poststorm_deep_change_pp"], de["post"] - de["pre"]), (dw["Jan25_model_deep_percent"], de["jan25_model"]), (dw["whole_record_model_deep_peak_percent"], de["peak"]), (dw["storm_bottom_drainage_mm"], de["storm_drain"])):
            assert math.isclose(a_, b_, rel_tol=1e-9, abs_tol=1e-9), (k, a_, b_)
        assert dw["whole_record_peak_close"] == de["peak_close"]
        assert R[k]["np"]["rows"] == rows and R[k]["np"]["ledger"] == res["ledger"] and all(v == 0.0 for v in rec["backend_max_absolute_difference"].values())
        d9 = np.array([r["layer_theta"] for r in R[k]["d900"]["rows"]])
        dd = (np.abs(d9 - R[k]["theta"]).max(0) * 100).tolist()
        assert all(math.isclose(x, y, rel_tol=1e-6, abs_tol=1e-12) for x, y in zip(dd, rec["dt900_max_theta_difference_pp_by_layer"]))
        B[k]["d900_pp"] = dd
        for v, (a_, c_) in R[k]["src"].items():
            for blk in (a_, c_):
                got = {x["module"]: (x["actual_sha256"], x["actual_files"]) for x in blk}
                assert got == {"maple_syrup": (SYR, 180), "maple": (MAP, 412)}, (k, v, got)
        assert R[k]["dec"]["status"] == "PASS" and len(R[k]["dec"]["cases"]) == 3 and R[k]["wall"]["returncode"] == 0 and res["config_sha256"] == sha(TASK / "study" / k / "case_numba.json")
        B[k]["dec_max_err_m"] = max(c["max_water_error_m"] for c in R[k]["dec"]["cases"])
        assert max(max(r["layer_theta"]) for r in rows) <= 0.1500 + 1e-9                  # no layer ever exceeds the largest field capacity at a daily close
    assert max(B[k]["dec_max_err_m"] for k in KEYS) < 1.5e-17
    ref = R["audit"]["reference_SR33_K1_E1"]
    R["ref"] = {"aligned": ref["pooled_aligned_rmse_pp"], "raw": ref["pooled_raw_rmse_pp"], "E": ref["E_mm"], "drain": ref["bottom_drainage_mm"]}
    R["A"], R["P"], R["Dp"] = A, P, Dp
    return S, B, PS, EP, DE


# ------------------------------------------------------------------------------------------------ figures
def style(ax):
    ax.grid(alpha=0.25)
    ax.tick_params(labelsize=8)


def rain_panel(ax, R, idx):
    rows = R[KEYS[0]]["res"]["rows"]
    ax.bar([R["close"][i] - dt.timedelta(days=0.5) for i in idx], [rows[i]["observed_rain_mm"] for i in idx], width=0.85, color="#1F5FA8")
    ax.set_ylabel("rain\n(mm/day)", fontsize=8)
    ax.set_title("Observed rain per nominal day", fontsize=9, loc="left")
    style(ax)


def fig_traj(R, s, aligned, name, title):
    idx = list(range(len(R["close"])))
    x = [R["close"][i] for i in idx]
    fig, axs = plt.subplots(4, 1, figsize=(7.2, 8.4), sharex=True, gridspec_kw={"height_ratios": [0.75, 1.4, 1.4, 1.4]})
    rain_panel(axs[0], R, idx)
    for j in range(3):
        ax = axs[j + 1]
        for p in FCS:
            k = f"fc{p:02d}_s{s}"
            off = R["theta0"][j] * 100 if aligned else 0
            ax.plot(x, R[k]["theta"][:, j] * 100 - off, color=FCCOL[p], ls=FCLS[p], lw=1.3, label=f"FC {p}%")
            ax.axhline(p - off, color=FCCOL[p], lw=0.6, ls=":", alpha=0.55)
        o = R["obs"][:, j] * 100 - (R["obs0"][j] * 100 if aligned else 0)
        ax.scatter(x, o, facecolors="none", edgecolors="black", s=13, lw=0.7, label="sensor point value" + (" (change since day 0)" if aligned else ""), zorder=4)
        if aligned:
            ax.axhline(0, color="#777777", lw=0.6)
        ax.set_ylabel("change since day 0\n(percentage points)" if aligned else "water content\n(vol. %)", fontsize=8)
        ax.set_title(DEPTH[LV[j]], fontsize=9, loc="left")
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        style(ax)
    h, l = axs[1].get_legend_handles_labels()
    h.append(plt.Line2D([], [], color="#555555", ls=":", lw=0.8))
    l.append("each field capacity (thin dotted, same colour)")
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=7, frameon=False)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0.09, 1, 0.97))
    fig.savefig(FIG / name, dpi=150)
    plt.close(fig)


def _line(ax, xs, key_fn, s, **kw):
    ax.plot(xs, [key_fn(f"fc{p:02d}_s{s}") for p in FCS], color=SC[s], ls=SM[s], marker="o", ms=3.5, lw=1.3, **kw)


def fig_scores(S, PS):
    fig, axs = plt.subplots(1, 3, figsize=(7.8, 3.7))
    ax = axs[0]
    for s in SK:
        _line(ax, FCS, lambda k: S[k]["aligned"]["pooled"]["rmse"], s, label=f"aligned, stress {SLAB[s]:.5g}")
        ax.plot(FCS, [S[f"fc{p:02d}_s{s}"]["raw"]["pooled"]["rmse"] for p in FCS], color=SC[s], ls=SM[s], marker="s", ms=3, lw=0.9, alpha=0.5, label=f"RAW, stress {SLAB[s]:.5g}")
    ax.axhline(S[KEYS[0]]["persistence"]["pooled"]["rmse"], color="#444444", ls=":", lw=1)
    ax.set_title("(a) all 282 values", fontsize=8.5)
    ax.set_ylabel("pooled RMSE (percentage points)", fontsize=8)
    ax.legend(fontsize=5.5, frameon=False)
    for ax in axs:
        ax.set_xlabel("uniform field capacity (vol. %)", fontsize=8)
        style(ax)
    ax = axs[1]
    dcol = {"501": "#B5651D", "502": "#1F5FA8", "503": "#7B3FA0"}
    for v in LV:
        for s in SK:
            ax.plot(FCS, [S[f"fc{p:02d}_s{s}"]["aligned"][v]["rmse"] for p in FCS], color=dcol[v], ls=SM[s], marker="o", ms=3, lw=1.2, label=f"{DEPTH[v].split(' (')[0]}, stress {SLAB[s]:.5g}")
        ax.axhline(S[KEYS[0]]["persistence"][v]["rmse"], color=dcol[v], ls=":", lw=0.8)
    ax.set_title("(b) aligned, by depth (dotted: no change)", fontsize=8.5)
    ax.set_ylabel("aligned RMSE (percentage points)", fontsize=8)
    ax.legend(fontsize=5.5, frameon=False, ncol=1)
    ax = axs[2]
    for s in SK:
        ax.plot(FCS, [PS[f"fc{p:02d}_s{s}"]["before_March1"]["raw"]["upper_two"]["rmse"] for p in FCS], color=SC[s], ls=SM[s], marker="o", ms=3.5, lw=1.3, label=f"before Mar 1, stress {SLAB[s]:.5g}")
        ax.plot(FCS, [PS[f"fc{p:02d}_s{s}"]["March1_onward"]["raw"]["upper_two"]["rmse"] for p in FCS], color=SC[s], ls=SM[s], marker="s", ms=3, lw=0.9, alpha=0.5, label=f"from Mar 1, stress {SLAB[s]:.5g}")
    ax.set_title("(c) 5.8 + 15.8 cm, RAW", fontsize=8.5)
    ax.set_ylabel("RAW RMSE (percentage points)", fontsize=8)
    ax.legend(fontsize=5.5, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig5_scores.png", dpi=150)
    plt.close(fig)


def fig_budget(B):
    fig, axs = plt.subplots(1, 3, figsize=(7.8, 3.4))
    for ax, key, lab in ((axs[0], "evap", "actual evaporation (mm)"), (axs[1], "drain", "bottom drainage (mm)"), (axs[2], "final", "final storage (mm)")):
        for s in SK:
            _line(ax, FCS, lambda k, key=key: B[k][key], s, label=f"stress {SLAB[s]:.5g}")
        ax.set_xlabel("uniform field capacity (vol. %)", fontsize=8)
        ax.set_ylabel(lab, fontsize=8)
        style(ax)
    axs[0].legend(fontsize=6.5, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig6_budgets.png", dpi=150)
    plt.close(fig)


def fig_deep(DE, B):
    fig, axs = plt.subplots(1, 3, figsize=(7.8, 3.6))
    ax = axs[0]
    for s in SK:
        _line(ax, FCS, lambda k: DE[k]["post"], s, label=f"model, Jan 24 close (storm), stress {SLAB[s]:.5g}")
        ax.plot(FCS, [DE[f"fc{p:02d}_s{s}"]["jan25_model"] for p in FCS], color=SC[s], ls=SM[s], marker="s", ms=3, lw=0.9, alpha=0.55, label=f"model, Jan 25 close, stress {SLAB[s]:.5g}")
    ax.plot(FCS, FCS, color="#888888", ls=":", lw=1, label="= field capacity")
    ax.axhline(DE[KEYS[0]]["jan25_obs"], color="black", ls="-", lw=1.2, label="sensor 25.8 cm, Jan 25 close")
    ax.set_title("(a) deep layer after the Jan 23 storm", fontsize=8.5)
    ax.set_ylabel("water content (vol. %)", fontsize=8)
    ax.legend(fontsize=5, frameon=False, loc="upper left")
    ax = axs[1]
    for s in SK:
        _line(ax, FCS, lambda k: DE[k]["peak"], s, label=f"model peak, stress {SLAB[s]:.5g}")
    ax.plot(FCS, FCS, color="#888888", ls=":", lw=1)
    ax.set_title("(b) whole-record deep model peak", fontsize=8.5)
    ax.set_ylabel("water content (vol. %)", fontsize=8)
    ax.legend(fontsize=6, frameon=False)
    ax = axs[2]
    for s in SK:
        ax.plot(FCS, [B[f"fc{p:02d}_s{s}"]["grav"][0] for p in FCS], color=SC[s], ls=SM[s], marker="o", ms=3.5, lw=1.3, label=f"upper interface, stress {SLAB[s]:.5g}")
        ax.plot(FCS, [B[f"fc{p:02d}_s{s}"]["grav"][1] for p in FCS], color=SC[s], ls=SM[s], marker="s", ms=3, lw=0.9, alpha=0.55, label=f"lower interface, stress {SLAB[s]:.5g}")
    ax.set_title("(c) gravity crossings (not additive)", fontsize=8.5)
    ax.set_ylabel("mm over 160 days", fontsize=8)
    ax.legend(fontsize=5.5, frameon=False)
    for ax in axs:
        ax.set_xlabel("uniform field capacity (vol. %)", fontsize=8)
        style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "fig7_deep_layer.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ tables
def g(x, n=3):
    return f"{x:.{n}f}"


def tables(R, S, B, PS, EP, DE):
    T = {}
    T["cases"] = (["Field capacity, all layers (vol. %)", "Stored at field capacity per 10 cm layer (mm)", "Stored at field capacity, profile (mm)", "Stress threshold (vol. %)", "Stress / 0.2017"],
                  [[f"{p}", f"{p:d}", f"{3 * p:d}", g(100 * SLAB[s], 3), "1.0" if s == "1" else "1.5"] for s in SLAB for p in FCS])
    pet = R[KEYS[0]]["res"]["plan"]["pet_mm_day"]["sum"]
    for s in SK:
        rows = [[f"{p}", g(B[f"fc{p:02d}_s{s}"]["evap"], 2), g(B[f"fc{p:02d}_s{s}"]["evap"] / pet * 100, 1), g(B[f"fc{p:02d}_s{s}"]["drain"], 3), g(B[f"fc{p:02d}_s{s}"]["final"], 2),
                 g(B[f"fc{p:02d}_s{s}"]["grav"][0], 2), g(B[f"fc{p:02d}_s{s}"]["grav"][1], 2)] for p in FCS]
        T[f"budget_{s}"] = (["Field capacity (vol. %)", "Actual E (mm)", "Actual E / PET (%)", "Bottom drainage (mm)", "Final storage (mm)", "Upper-interface crossing (mm)", "Lower-interface crossing (mm)"], rows)
    cols = ["5.8 cm (n = 56)", "15.8 cm (n = 147)", "25.8 cm (n = 79)", "All pooled (n = 282)"]
    vs = LV + ["pooled"]
    for kind, nm in (("aligned", "scores_aligned"), ("raw", "scores_raw")):
        rows = [[LAB(k)] + [g(S[k][kind][v]["rmse"]) for v in vs] for k in KEYS] + [["No change (day-0 sensor value held)"] + [g(S[KEYS[0]]["persistence"][v]["rmse"]) for v in vs]]
        T[nm] = (["Setting"] + cols, rows)
    pc = lambda k, pn, kind, sub: PS[k][pn][kind][sub]
    T["exchange"] = (["Field capacity (vol. %)", "Stress", "Upper interface, net exchange (mm)", "Lower interface, net exchange (mm)", "Deep layer peak (vol. %)", "Deep layer at the end (vol. %)"],
                     [[f"{fcp(k)}", g(SLAB[sk(k)], 5), f"{B[k]['diff_down'][0]:+.3f}", f"{B[k]['diff_down'][1]:+.3f}", g(DE[k]["peak"], 2), g(B[k]["final_theta"][2], 2)] for k in KEYS])
    T["period_top"] = (["Setting", "Before Mar 1 RAW", "Before Mar 1 aligned", "From Mar 1 RAW", "From Mar 1 aligned"],
                       [[LAB(k)] + [g(pc(k, pn, kind, "upper_two")["rmse"]) for pn in ("before_March1", "March1_onward") for kind in ("raw", "aligned")] for k in KEYS])
    T["period_bias"] = (["RAW bias, model minus sensor (percentage points)", "Upper two, before Mar 1", "Upper two, from Mar 1", "All three, before Mar 1", "All three, from Mar 1"],
                        [[LAB(k)] + [f"{pc(k, pn, 'raw', sub)['bias']:+.2f}" for sub in ("upper_two", "all_three") for pn in ("before_March1", "March1_onward")] for k in KEYS])
    e0 = EP[KEYS[0]]
    T["endpoint"] = (["Setting (vol. %)"] + [f"{DEPTH[e['level']].split(' (')[0]}, {e['close'][:10]}" for e in e0],
                     [["Observed (last valid sensor value)"] + [g(e["obs"], 2) for e in e0]] + [[LAB(k)] + [g(e["model"], 2) for e in EP[k]] for k in KEYS])
    T["deep"] = (["Setting", "Deep layer before storm, Jan 23 close (vol. %)", "Deep layer after storm, Jan 24 close (vol. %)", "Deep layer, Jan 25 close (obs 6.34) (vol. %)", "Whole-record deep peak (vol. %)", "Date of peak (close)"],
                 [[LAB(k), g(DE[k]["pre"], 2), g(DE[k]["post"], 2), g(DE[k]["jan25_model"], 2), g(DE[k]["peak"], 2), DE[k]["peak_close"][:10]] for k in KEYS])
    rows, rows2 = [], []
    for k in KEYS:
        tm = R[k]["tm"]
        cl = R[k]["res"]["cumulative_closure"]
        rows.append([LAB(k), g(tm["integration_s"], 4), g(tm["operator_prepare_s"], 2), g(tm["source_checks_and_input_preparation_s"], 2), g(R[k]["wall"]["seconds"], 2)])
        rows2.append([LAB(k), f"{abs(cl['soil_residual_m3']):.1e} / {cl['soil_bound_m3']:.2e}", f"{max(B[k]['d900_pp']):.5f}", f"{B[k]['dec_max_err_m']:.1e}", "0 (exact)"])
    T["timing"] = (["Run", "Integration (s)", "Operator prep + JIT (s)", "Source checks + input prep (s)", "Full fresh command (s)"], rows)
    T["checks"] = (["Run", "Closure residual / native bound (m³)", "900 s step: max change (percentage points)", "90-digit check: max error (m)", "NumPy vs Numba max difference"], rows2)
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
        (4, "NEON (RELEASE-2026). Soil water content and water salinity, DP1.00094.001 (Jornada, JORN, plot 004, sensors 501-503).", "https://data.neonscience.org/data-products/DP1.00094.001/RELEASE-2026"),
        (5, "NEON (RELEASE-2026). Precipitation, DP1.00045.001.", "https://data.neonscience.org/data-products/DP1.00045.001/RELEASE-2026"),
        (6, "NEON (RELEASE-2026). Single aspirated air temperature, DP1.00002.001 (air temperature only; no soil temperature is used by the evaporation demand).", "https://data.neonscience.org/data-products/DP1.00002.001/RELEASE-2026"),
        (7, "NEON (RELEASE-2026). Soil physical and chemical properties, Megapit, DP1.00096.001 (Jornada, 2013-03 pit 1): texture, organic carbon and bulk density behind the layer properties. A spatial proxy for the 2024 plot 004 sensors, not demonstrably co-located.", "https://data.neonscience.org/data-products/DP1.00096.001/RELEASE-2026"),
        (8, "SYRUP project. Joint fit of the diffusion multiplier d and the evaporation-stress threshold at E = 1 (2026-11-18/19), docs/subsurface_hydrology/joint_d_stress_calibration/report.md; and the earlier SR33 field-capacity / evaporation-sensitivity reports of 2026-11-20 (SYRUP_JORN_160day_SR33_K_sensitivity_20261120.docx, SYRUP_JORN_160day_SR33_1x_evaporation_sensitivity_20261120.docx).", "")]


def trend(vals, nd=3, what="FC"):
    """Describe a seven-point series by its ACTUAL shape (monotonic, plateau at zero, single interior extremum, or irregular)."""
    vals = [float(v) for v in vals]
    d = np.diff(vals)
    tol = 1e-9
    a, b = f"{vals[0]:.{nd}f} at {what} {FCS[0]}", f"{vals[-1]:.{nd}f} at {what} {FCS[-1]}"
    if np.all(d > tol):
        return f"rises steadily from {a} to {b}"
    if np.all(d < -tol):
        return f"falls steadily from {a} to {b}"
    if np.all(d >= -tol):
        return f"rises from {a} to {b} (flat in places)"
    if np.all(d <= tol):
        z = [FCS[i] for i, v in enumerate(vals) if abs(v) < 5 * 10 ** (-nd)]
        return f"falls from {a} to {b}" + (f" and is zero from {what} {z[0]} upward" if z and z == FCS[FCS.index(z[0]):] else " (flat in places)")
    imax, imin = int(np.argmax(vals)), int(np.argmin(vals))
    if 0 < imax < len(vals) - 1 and np.all(d[:imax] > tol) and np.all(d[imax:] < -tol):
        return f"rises from {a} to a maximum of {vals[imax]:.{nd}f} at {what} {FCS[imax]}, then falls to {b}"
    if 0 < imin < len(vals) - 1 and np.all(d[:imin] < -tol) and np.all(d[imin:] > tol):
        return f"falls from {a} to a minimum of {vals[imin]:.{nd}f} at {what} {FCS[imin]}, then rises to {b}"
    return f"is irregular (from {a} to {b}; lowest {vals[imin]:.{nd}f} at {what} {FCS[imin]}, highest {vals[imax]:.{nd}f} at {what} {FCS[imax]})"


def blocks(R, S, B, PS, EP, DE):
    ref = R["ref"]
    fam = lambda s: [f"fc{p:02d}_s{s}" for p in FCS]
    al = lambda k, v="pooled": S[k]["aligned"][v]["rmse"]
    rw = lambda k, v="pooled": S[k]["raw"][v]["rmse"]
    top = lambda k, pn, kind: PS[k][pn][kind]["upper_two"]["rmse"]
    nochange = S[KEYS[0]]["persistence"]["pooled"]["rmse"]
    best_al = {s: min(fam(s), key=al) for s in SLAB}
    best_rw = {s: min(fam(s), key=rw) for s in SLAB}
    best_early_raw = min(KEYS, key=lambda k: top(k, "before_March1", "raw"))
    best_early_al = min(KEYS, key=lambda k: top(k, "before_March1", "aligned"))
    best_late_raw = min(KEYS, key=lambda k: top(k, "March1_onward", "raw"))
    best_late_al = min(KEYS, key=lambda k: top(k, "March1_onward", "aligned"))
    order_diff = {s_: [sorted(FAMS[s_], key=al) != sorted(FAMS[s_], key=rw)][0] for s_ in SK}
    assert all(best_al[s_] == best_rw[s_] for s_ in SK)                     # RAW and aligned agree on the lowest pooled case of each family
    n_below_nochange = sum(al(k) < nochange for k in KEYS)
    reach = sum(abs(DE[k]["post"] - fcp(k)) < 0.05 for k in KEYS)
    wetted = [k for k in KEYS if DE[k]["post"] - DE[k]["pre"] > 0.5]
    dry = [k for k in KEYS if k not in wetted]
    assert max(fcp(k) for k in wetted) < min(fcp(k) for k in dry)          # the wording 'lower / higher field capacities' is checked against the data
    rank_note = "; the order of the intermediate cases differs between the two scorings" if any(order_diff.values()) else ""
    d0off = (R["theta0"][2] - R["obs0"][2]) * 100
    hi_raw = [DE[k]["jan25_model"] - DE[k]["jan25_obs"] for k in dry]
    hi_al = [x - d0off for x in hi_raw]
    assert min(hi_raw) > 0 and min(hi_al) > 0                              # the text speaks of a positive (model-wetter) error
    dchg = lambda ks: f"{min(DE[k]['post'] - DE[k]['pre'] for k in ks):.2f}-{max(DE[k]['post'] - DE[k]['pre'] for k in ks):.2f}"
    dmax = float(np.nanmax(R["obs"][:, 2]) * 100)
    j25 = [DE[k]["jan25_model"] for k in KEYS]
    j25_obs = DE[KEYS[0]]["jan25_obs"]
    sdiff = [al(f"fc{p:02d}_s1p5") - al(f"fc{p:02d}_s1") for p in FCS]
    e_all = [B[k]["evap"] for k in KEYS]
    edif = [B[f"fc{p:02d}_s1p5"]["evap"] - B[f"fc{p:02d}_s1"]["evap"] for p in FCS]
    d_all = [B[k]["drain"] for k in KEYS]
    pet = R[KEYS[0]]["res"]["plan"]["pet_mm_day"]["sum"]
    off = R["theta0"] - R["obs0"]
    restart = R["restart"]
    if restart is not None and restart.get("status") == "PASS":
        rtxt = (f"Restart: Root's receipt root_restart_verification.json ({restart['status']}, {restart['utc'][:19].replace('T', ' ')} UTC, case {restart.get('case')}, split at nominal day {restart.get('split_day')}, fresh-process resume) reproduces the listed outputs exactly. "
                "It is a driver-level component state reconstruction only, not a native checkpoint or coupled-model acceptance, and no new run was made.")
    else:
        rtxt = "Restart: no component restart receipt existed when this report was built, so no restart claim is made; Root will update this line if a receipt is added before publication."
    dtxt = "; ".join(f"stress {SLAB[s]:.5g}: lowest at FC {fcp(best_al[s])} ({al(best_al[s]):.3f})" for s in SLAB)
    return [
        ("title", "Uniform field capacity of 9 to 15 vol. % in all layers, at two stress values: JORN, January-June 2024"),
        ("sub", "The frozen 160-day JORN column of the SR33 1xK study, re-run with one field capacity (9, 10, 11, 12, 13, 14 or 15 vol. %) imposed on all three layers and with the evaporation stress threshold 0.2017 or 0.30255: fourteen crossed cases. Explicit field-capacity overrides, not a recomputed retention curve, a new soil or a recalibration."),
        ("h1", "1. Summary"),
        ("bul", [f"**Pooled error as a function of field capacity.** Pooled aligned RMSE over 282 sensor values (percentage points) {trend([al(k) for k in fam('1')])} at stress 0.2017, and {trend([al(k) for k in fam('1p5')])} at stress 0.30255. The lowest value of each family ({dtxt}) is not a physical optimum; the no-change reference is {nochange:.3f}, and {n_below_nochange} of the 14 cases are better than it. The earlier accepted SR33 case (depth-specific field capacity 9.75 / 9.94 / 10.91 vol. %, stress 0.2017058) scored {ref['aligned']:.3f} and is shown only as context. RAW and aligned scoring agree on the lowest pooled case of each family (FC {fcp(best_rw['1'])} at stress 0.2017, FC {fcp(best_rw['1p5'])} at 0.30255){rank_note}.",
                 f"**Raising the stress threshold 1.5 times** changes the pooled aligned RMSE by {min(sdiff):+.3f} to {max(sdiff):+.3f} points depending on field capacity, and changes the actual evaporation by {min(edif):+.2f} to {max(edif):+.2f} mm (at the same field capacity): the stress and the field capacity interact, and neither can be read alone.",
                 f"**The January storm and the deep layer.** Computed, not assumed: at the 2024-01-24 07:00 UTC close (the first close after the 10 mm January 23 storm) the model 20-30 cm layer is within 0.05 points of its field capacity in {reach} of the 14 cases. Using a declared descriptive threshold of 0.5 points (not a physical classification), the storm wets the deep layer in {len(wetted)} cases (the lower field capacities, FC {min(fcp(k) for k in wetted)}-{max(fcp(k) for k in wetted)}; by {dchg(wetted)} points) and only slightly in the other {len(dry)} (FC {min(fcp(k) for k in dry)}-{max(fcp(k) for k in dry)}; by {dchg(dry)} points). The first following deep sensor reading (Jan 25 close) is {j25_obs:.2f} vol. %, against {min(j25):.2f}-{max(j25):.2f} in the model; the 25.8 cm sensor never exceeds {dmax:.2f} vol. %. The higher field capacities remove the large instantaneous storm jump and reduce the positive deep error (raw +{min(hi_raw):.2f} to +{max(hi_raw):.2f} points; +{min(hi_al):.2f} to +{max(hi_al):.2f} after subtracting the day-0 offset of {d0off:+.2f} points), but this does not establish agreement with the sensor: no measurement or support uncertainty bound is available, and a later deep wetting and drying error remains (deep aligned RMSE {min(al(k, '503') for k in dry):.2f}-{max(al(k, '503') for k in dry):.2f} against {S[KEYS[0]]['persistence']['503']['rmse']:.3f} for no change). Neither a probe fault nor a data filter is assumed.",
                 f"**Water.** All {B[KEYS[0]]['intake']:.3f} mm of rain infiltrates in every case, with no runoff. Bottom drainage {trend([B[k]['drain'] for k in fam('1')], 2)} at stress 0.2017 (range {min(d_all):.2f}-{max(d_all):.2f} mm over all cases), and actual evaporation ranges {min(e_all):.1f}-{max(e_all):.1f} mm ({100 * min(e_all) / pet:.1f}-{100 * max(e_all) / pet:.1f}% of the summed potential PET). The initial 22.9424 mm is the same in every case and is not rescaled to the field capacity.",
                 f"**Seasons.** For the two shallower sensors before March 1 the lowest RAW RMSE is {LAB(best_early_raw)} ({top(best_early_raw, 'before_March1', 'raw'):.3f}) and the lowest aligned RMSE is {LAB(best_early_al)} ({top(best_early_al, 'before_March1', 'aligned'):.3f}); from March 1 the lowest RAW RMSE is {LAB(best_late_raw)} ({top(best_late_raw, 'March1_onward', 'raw'):.3f}). These periods use the user's assumed start of transpiration as context only: the model has no transpiration, and no phenology was measured."]),
        ("h1", "2. What was run"),
        ("p", "The base is the accepted SR33 1xK run [8]: one 1 m² column of three 10 cm layers with an open bottom, all conductivities at 1x, the Tietjen-type gradient exchange [1] with d = 0.0010239461376024856, NEON rain of 25.088 mm [5], Hargreaves-Samani potential evaporation [3] from NEON air temperature [6] (E = 1, 777.3 mm summed PET), initial water 22.9424 mm from the NEON soil sensors [4], matrix-only infiltration, daily hydrology, and no transpiration, growth, plants, shading, preferential flow or routing (surface water is exported with zero residence). Evaporation and transpiration are off during rain, and no new sub-daily dry-day step is added. The clock is the earlier daily-weather approximation: 160 nominal days plus 27.5 added storm hours, ending at the 2024-06-26 07:00 UTC close. Layer properties come from the 2013 Jornada megapit [7], a spatial proxy for the 2024 plot 004 sensors, not demonstrably co-located."),
        ("p", "**What changed.** Only the field capacity and the stress threshold differ from the base (and names and provenance text). The field capacity is an explicit user override, 9 to 15 vol. % in all three layers; saturation, wilting point, residual content, Ks, the matrix and interface conductivities, d and the Green-Ampt geometry were not recomputed. These are therefore not seven self-consistent Saxton-Rawls retention curves [2] and not seven new soils. The base used depth-specific values of 9.746, 9.940 and 10.914 vol. %; the uniform values here replace them in the run (the old values remain only in the provenance text as context). The stress values 0.2017 and 0.30255 (= 1.5 x 0.2017) are rounded or scaled sensitivity settings; the earlier fitted value was 0.20170581117963332. Nothing is refit and no default changes."),
        ("tab", "cases", "Table 1. The fourteen cases (seven field capacities at each of two stress values). A field capacity of x vol. % stores x mm in a 10 cm layer and 3x mm in the 30 cm profile."),
        ("h1", "3. What field capacity does in the model"),
        ("p", f"Field capacity here is the drainage threshold of the gravity step, not a ceiling on storage: any water above it is passed down to the next layer at once, independent of the conductivities (it is not rate-limited by K), and from the bottom layer out of the open bottom. Saturation (about 46 vol. %) is not reached in these runs. At the highest field capacity (15 vol. %) a 10 cm layer holds 15 mm before passing water on; at 9 vol. % it holds 9 mm, so a larger field capacity retains more of each rain pulse in the layer where it arrives and a smaller one sends more of it deeper. The initial layer water is {R['theta0'][0] * 100:.2f}, {R['theta0'][1] * 100:.2f} and {R['theta0'][2] * 100:.2f} vol. %, below even 9 vol. %, and it is the same observed profile in all cases."),
        ("p", "Changing the field capacity therefore changes which rain pulses cascade through the two interfaces, how much water reaches the bottom, and, through the top-layer water content, the evaporation (the top layer never holds more than its field capacity at the evaporation step, and the stress threshold, 20-30 vol. %, is above every field capacity, so the quadratic reduction applies in all cases). The laws and coefficients of the Tietjen gradient exchange between layers (K, d, interface conductivities) and of the infiltration (matrix K) are unchanged, but the amounts they exchange are not: the exchange follows the moisture profiles, which the field capacity changes (Table 2c). All 25.088 mm of rain enters in every case because every rain rate is below the matrix K and nothing ponds here, not because intake would be invariant under any change of field capacity. This study does not adopt the Saxton-Rawls unsaturated conductivity."),
        ("p", "**The stress threshold.** It is the water content at which the full potential rate is still allowed; below it the demand is multiplied by (water content / threshold)², and a separate residual content stops withdrawal. It is a proxy for the rising resistance of drying soil, not a measured soil property or a retention point. The baseline 20.17 vol. % was fitted jointly with d, at E = 1 with transpiration set to zero and the earlier 6 kPa field capacity, to the 53 valid sensor values of rain-free JORN (April 8-18 and 26-30, 2025) and MOAB (May 22-28, 2025) windows [8]; if plants were active then, the fit could have absorbed the missing transpiration. It is not the Saxton-Rawls wilting point (3.95-4.63 vol. %). The value 30.26 vol. % was used here only as a sensitivity setting."),
        ("h1", "4. Water: what the field capacity changes"),
        ("tab", "budget_1", "Table 2a. Stress 0.2017. Rain = intake = 25.088 mm and runoff = 0 in every case; the initial storage is 22.9424 mm. The two interface columns are separate internal pathway counts and can include the same water: do not add them to each other or to losses and storage."),
        ("tab", "budget_1p5", "Table 2b. Stress 0.30255. Same quantities and the same caution about the interface columns."),
        ("p", f"At stress 0.2017 the actual evaporation {trend([B[k]['evap'] for k in fam('1')], 2)}, the bottom drainage {trend([B[k]['drain'] for k in fam('1')], 2)} and the final storage {trend([B[k]['final'] for k in fam('1')], 2)}; at stress 0.30255 the evaporation {trend([B[k]['evap'] for k in fam('1p5')], 2)}, the drainage {trend([B[k]['drain'] for k in fam('1p5')], 2)} and the final storage {trend([B[k]['final'] for k in fam('1p5')], 2)} (Figure 6). The crossings of the upper interface {trend([B[k]['grav'][0] for k in fam('1')], 2)} and of the lower interface {trend([B[k]['grav'][1] for k in fam('1')], 2)} at stress 0.2017. The changes in evaporation, retention and drainage cannot be attributed to the field capacity through drainage alone: the field capacity also changes the top-layer water content on which the evaporation depends."),
        ("tab", "exchange", "Table 2c. Cumulative NET gradient exchange across each interface over 160 days (mm; downward positive, upward negative), with the model deep-layer peak and end value. These are internal transfers: they are not losses and are not added to the other budget terms."),
        ("p", f"A fixed K does not mean a fixed exchange. Even where the lower interface passes no gravity water (FC 13 and above), the gradient exchange can still move water into the deep layer, so the deep layer can rise without any gravity transfer. The net integral can also be negative after an earlier wetting peak, so a negative end total is not evidence of no deep wetting: at FC 15 and stress 0.2017 the net exchange across the lower interface is {B['fc15_s1']['diff_down'][1]:+.3f} mm while the deep layer peaks at {DE['fc15_s1']['peak']:.2f} vol. % before ending at {B['fc15_s1']['final_theta'][2]:.2f}; at stress 0.30255 it is {B['fc15_s1p5']['diff_down'][1]:+.3f} mm and the peak {DE['fc15_s1p5']['peak']:.2f} vol. %. The exchange amounts therefore differ with the profile history even though the law and coefficients do not."),
        ("fig", "fig6_budgets.png", "Figure 6. Actual evaporation, bottom drainage and final storage against the uniform field capacity, at the two stress values."),
        ("h1", "5. Comparison with the sensors"),
        ("p", f"**RAW** compares model layer means with sensor point values; **aligned** subtracts one constant day-0 offset per depth ({off[0] * 100:+.2f}, {off[1] * 100:+.2f} and {off[2] * 100:+.2f} points at 5.8, 15.8 and 25.8 cm), because a 10 cm mean is not a point reading. The offsets are the same in every case; model water is never altered by the alignment. Figures 1 and 3 show the seven field capacities at stress 0.2017, Figures 2 and 4 those at 0.30255; the thin dotted lines are the field capacities in the colour of their curve."),
        ("fig", "fig1_raw_s1.png", "Figure 1. RAW water content, stress 0.2017, seven field capacities (colours and dash styles in the legend); sensor point values are black circles (gaps are not interpolated). The top panel shows observed rain."),
        ("fig", "fig2_raw_s1p5.png", "Figure 2. RAW water content, stress 0.30255. Same layout as Figure 1."),
        ("tab", "scores_aligned", "Table 3a. ALIGNED RMSE in percentage points (100 x m³/m³) by depth (n = number of sensor values) and pooled. No change = holding the day-0 sensor value."),
        ("tab", "scores_raw", "Table 3b. RAW RMSE in percentage points, same rows and columns."),
        ("fig", "fig3_aligned_s1.png", "Figure 3. Changes since day 0 (the aligned view), stress 0.2017. The dotted lines are the field capacities minus the layer's day-0 content."),
        ("fig", "fig4_aligned_s1p5.png", "Figure 4. Changes since day 0, stress 0.30255."),
        ("fig", "fig5_scores.png", "Figure 5. (a) Pooled RMSE (aligned: solid with circles; RAW: lighter with squares; dotted = pooled no change). (b) Aligned RMSE by depth, with the per-depth no-change values dotted. (c) RAW RMSE for the 5.8 and 15.8 cm sensors before and from March 1. Both stress values; no continuous fit is implied."),
        ("p", f"By depth, the aligned RMSE of the 5.8 cm sensor {trend([al(k, '501') for k in fam('1')])} at stress 0.2017, the 15.8 cm sensor {trend([al(k, '502') for k in fam('1')])}, and the 25.8 cm sensor {trend([al(k, '503') for k in fam('1')])} (no change: {S[KEYS[0]]['persistence']['501']['rmse']:.3f}, {S[KEYS[0]]['persistence']['502']['rmse']:.3f}, {S[KEYS[0]]['persistence']['503']['rmse']:.3f}). The middle sensor dominates the pooled score (n = 147). The depths therefore trade against each other, and the pooled minimum is a compromise (Figure 5). Neither the pooled error nor any depth's error is taken as the sole physical criterion; no global best is declared."),
        ("h1", "6. Before and after March 1"),
        ("p", f"Periods use the nominal local-day label (before 2024-03-01 against from 2024-03-01); each observation is at the daily close that follows its nominal day. No new calendar or transpiration rule is introduced: the split follows the user's assumed start of transpiration only as context, and no phenology was measured. For the two shallower sensors the counts are 84 before and 119 from March 1; for all three, 104 and 178. The lowest upper-two RAW RMSE before March 1 is {top(best_early_raw, 'before_March1', 'raw'):.3f} ({LAB(best_early_raw)}) and from March 1 {top(best_late_raw, 'March1_onward', 'raw'):.3f} ({LAB(best_late_raw)}); the lowest aligned values are {top(best_early_al, 'before_March1', 'aligned'):.3f} before March 1 ({LAB(best_early_al)}) and {top(best_late_al, 'March1_onward', 'aligned'):.3f} from March 1 ({LAB(best_late_al)}). Table 4b shows the RAW bias: a positive bias means the model is wetter than the sensors, which after March 1 may reflect the missing transpiration but cannot be separated from other causes with these data."),
        ("tab", "period_top", "Table 4a. RMSE (percentage points) for the 5.8 and 15.8 cm sensors, before March 1 (n = 84) and from March 1 (n = 119)."),
        ("tab", "period_bias", "Table 4b. RAW bias, model minus sensor (percentage points), for the upper two and all three sensors, by period (counts 84/119 and 104/178). RMSE for all three sensors is in the CSV tables."),
        ("h1", "7. Same-time endpoints and the deep layer"),
        ("tab", "endpoint", "Table 5. Last valid sensor value at each depth and every case at the SAME close. The shallow sensor has no value after the 2024-03-15 close and the middle sensor none at the 2024-06-26 close; the model's final inventories (Tables 2a and 2b) are valid model end states and imply no observation."),
        ("p", f"The January 23 storm (10.036 mm, nominal day January 23, closing at 2024-01-24 07:00 UTC) raises the model deep layer by {dchg(wetted)} points in the {len(wetted)} cases where it exceeds the declared descriptive threshold of 0.5 points (FC {min(fcp(k) for k in wetted)}-{max(fcp(k) for k in wetted)}; to {min(DE[k]['post'] for k in wetted):.2f}-{max(DE[k]['post'] for k in wetted):.2f} vol. %; Table 6, Figure 7a) and by {dchg(dry)} points in the other {len(dry)} (FC {min(fcp(k) for k in dry)}-{max(fcp(k) for k in dry)}); the threshold is a convenience for description, not a physical classification. The deep sensor has no value at that close (a gap, not zero); its first following value, at the Jan 25 close, is {j25_obs:.2f} vol. %, against {min(j25):.2f}-{max(j25):.2f} in the model, and its maximum over the record is {dmax:.2f} vol. %. Where the deep layer is wetted, a whole 20-30 cm layer should show up at a probe at 25.8 cm, and the sensor shows no corresponding rise. At the higher field capacities the large instantaneous storm jump is removed and the positive error is smaller: the raw error is +{min(hi_raw):.2f} to +{max(hi_raw):.2f} points, and +{min(hi_al):.2f} to +{max(hi_al):.2f} after subtracting the day-0 offset between the layer mean and the sensor point ({d0off:+.2f} points). That reduces the error but does not establish agreement with the sensor: no uncertainty bound for point-against-layer support or measurement exists here, and a later deep wetting and drying error remains (aligned RMSE of the deep sensor {min(al(k, '503') for k in dry):.2f}-{max(al(k, '503') for k in dry):.2f} for these cases, against {S[KEYS[0]]['persistence']['503']['rmse']:.3f} for no change; Table 3a). Candidate causes of the mismatch include the cascade approximation, the 2013 soil proxy and sensor or support effects; none is established. The sensor is not declared faulty, no value was removed, and the model is not a wetting-front solver."),
        ("tab", "deep", "Table 6. The model deep layer around the January 23 storm and over the whole record, at the two stress values (the sensor value at the Jan 25 close is 6.34 vol. %)."),
        ("fig", "fig7_deep_layer.png", "Figure 7. (a) Deep-layer water content after the storm against the uniform field capacity (dotted: equal to field capacity; black: the sensor at the Jan 25 close). (b) Whole-record deep-layer model peak. (c) Gravity-step crossings of the two interfaces (separate internal counts, not additive)."),
        ("h1", "8. Verification and run time"),
        ("bul", [f"Before consuming any result I re-hashed all {len(R['seal'])} files of Root's seal and checked them again afterwards. From the daily rows I recomputed the per-depth and pooled RAW, aligned and no-change scores, the period scores and counts, the endpoints, the water budgets, closure and the January deep-layer values, and compared them with Root's audit, period and deep-wetting files; all agree (relative 1e-9).",
                 f"NumPy and Numba are identical in every tracked output for all 14 cases; a 900 s storm step changes layer water by at most {max(max(B[k]['d900_pp']) for k in KEYS):.5f} percentage points. The finer controls refine the storm steps only; dry interevent evaporation remains one daily step, so no time-resolution qualification of dry-day evaporation is claimed. Root's independent 90-digit calculation (open gravity cascade, evaporation and pair relaxation, no production step call; valid here because the rows confirm no surface return or room rejection) agrees within 1.5e-17 m (largest {max(B[k]['dec_max_err_m'] for k in KEYS):.3g} m). Source packages SYRUP 180 files (4e38a472...) and MAPLE 412 files (fbc504bb...) are exact before and after every run.",
                 rtxt + " GPU, scaling, large-domain and coupled-model claims are not made."]),
        ("tab", "timing", "Table 7a. Timing, separated (separate fresh runs on a shared host with other jobs; not a controlled speed comparison)."),
        ("tab", "checks", "Table 7b. Verification per case: closure, 900 s step, 90-digit control and NumPy/Numba comparison."),
        ("p", "Integration takes a few hundredths of a second per case; the fresh-process time is dominated by one-time source verification, input preparation and compilation and says nothing about solver speed. One 1 m² CPU column."),
        ("h1", "9. Limits"),
        ("bul", ["A prescribed sensitivity: no calibration, no best parameter, no default change and no adoption. The field capacity is an override on SR33-derived conductivities, not a self-consistent retention curve; d and the stress threshold are conditional on E = 1, no transpiration and the earlier field capacity, and the stress and field capacity interact.",
                 "No transpiration, plants or shading; the observed drying after March 1 may include transpiration the model lacks. This is a conditional inference, not a measurement or a claim of plant activity. The earlier 2025 dry-period fit may have absorbed missing transpiration.",
                 "The above-field-capacity gravity cascade is instantaneous and not limited by any conductivity; Tietjen K and d (gradient exchange) are unchanged and the Saxton-Rawls unsaturated conductivity is not adopted. Persistent Green-Ampt history has no automatic reset or drying recovery; here all rain is unponded and below the matrix K, so it does not control intake.",
                 "Point sensors against 10 cm layer means, a daily clock with no new sub-daily dry-day step, a 2013 soil proxy for the 2024 plot, and the 030 against 020 air-temperature height inherited from the original run. No significance test; probe error or unrepresentativeness is neither established nor excluded."]),
        ("h1", "References"), ("refs", REFS),
        ("h1", "Appendix. Identity (compact)"),
        ("p", "Sources: SYRUP package 180 files, SHA-256 " + SYR + "; MAPLE package 412 files, SHA-256 " + MAP + ". Base case: jorn_sr33_160day_20261120/study/k1/case_numba.json, SHA-256 " + BASE_SHA + ". Case-file hashes: " + "; ".join(f"fc{fcp(k)}/{sk(k)} {sha(TASK / 'study' / k / 'case_numba.json')[:12]}" for k in KEYS)
         + ". The hash of every sealed file is in tables/provenance_seal.csv and artifact_manifest.json; daily trajectories, scores and period scores of all cases are in the CSV tables. Raw NEON files are not copied.")]
def to_docx(Bk, T, path):
    doc = Document()
    setup_styles(doc)
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Inches(8.5), Inches(11)
    sec.left_margin = sec.right_margin = Inches(0.9)
    sec.top_margin = sec.bottom_margin = Inches(0.8)
    fp = sec.footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    fr = fp.add_run("SYRUP JORN 160-day: uniform field capacity 9-15 vol. % x stress · not a recalibration · page ")
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
            p.add_run().add_picture(str(FIG / b[1]), width=Inches(6.0 if b[1] != "fig5_scores.png" else 6.7))
            runs(doc.add_paragraph(style="RepCaption"), b[2])
        elif k == "refs":
            for n, c, u in b[1]:
                p = doc.add_paragraph()
                p.paragraph_format.left_indent = Inches(0.25)
                p.paragraph_format.first_line_indent = Inches(-0.25)
                p.paragraph_format.space_after = Pt(2)
                p.add_run(f"[{n}] ").bold = True
                p.add_run(f"{c} {u}".strip()).font.size = Pt(9)
    doc.core_properties.title = "SYRUP JORN 160-day: uniform field-capacity sensitivity"
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




def restart_note(R):
    rv = R["restart"]
    assert rv is not None and rv["status"] == "PASS" and rv["case"] == "fc12_s1p5" and rv["split_day"] == 80 and rv["fresh_process_resume"] is True
    assert str(TASK / "root_restart_verification.json") in R["seal"] or R["rseal"]                 # the receipt is part of Root's seal
    return (f"Restart: Root's receipt `root_restart_verification.json` ({rv['status']}, {rv['utc'][:19].replace('T', ' ')} UTC) was read and is covered by the re-hashed seal. Case {rv['case']} (field capacity {100 * rv['field_capacity_all_layers']:g} vol. %, stress {rv['stress']}) "
            f"was split at nominal day {rv['split_day']} and resumed in a fresh process; exact equality of {', '.join(rv['exact_equal_fields'])} is reported. Scope as stated by the receipt: {rv['scope']} "
            "It is a driver-level component state reconstruction only, not a native checkpoint or coupled-model acceptance, and no new run was made. The receipt is copied to `data/`.\n\n")


def main():
    R = prep(load())
    S, B, PS, EP, DE = review(R)
    for p in list(TAB.glob("*.csv")) + list(DATA.glob("*")):
        p.unlink()
    for s in SK:
        fig_traj(R, s, False, f"fig{1 if s == '1' else 2}_raw_s{s}.png", f"RAW water content at each nominal daily close: stress {SLAB[s]:.5g}, field capacity 9-15 vol. %")
        fig_traj(R, s, True, f"fig{3 if s == '1' else 4}_aligned_s{s}.png", f"Changes since day 0 (aligned view): stress {SLAB[s]:.5g}, field capacity 9-15 vol. %")
    fig_scores(S, PS)
    fig_budget(B)
    fig_deep(DE, B)
    T = tables(R, S, B, PS, EP, DE)
    for name, (hd, rows) in T.items():
        csvw(TAB / f"{name}.csv", hd, rows)
    csvw(TAB / "provenance_seal.csv", ["file", "sha256", "bytes"], [[k, v["sha256"], v["bytes"]] for k, v in R["seal"].items()])
    csvw(TAB / "case_hashes.csv", ["case", "config_sha256_in_result", "case_numba_json_sha256"], [[k, R[k]["res"]["config_sha256"], sha(TASK / "study" / k / "case_numba.json")] for k in KEYS])
    hdr = ["day_index", "date_utc_minus7", "nominal_close_utc", "physical_t_s", "rain_mm", "pet_mm"] + [f"theta_l{j + 1}_{k}" for k in KEYS for j in range(3)] + [f"evap_mm_{k}" for k in KEYS] + ["obs_501", "obs_502", "obs_503"]
    rows = []
    for i, r0 in enumerate(R[KEYS[0]]["res"]["rows"]):
        rows.append([r0["day_index"], r0["date"], r0["nominal_close_utc"], r0["physical_t_s"], 1000 * r0["rain_m3"], r0["pet_mm_day"]] + [R[k]["res"]["rows"][i]["layer_theta"][j] for k in KEYS for j in range(3)]
                    + [1000 * R[k]["res"]["rows"][i]["evaporation_m3"] for k in KEYS] + ["" if v is None else v for v in r0["obs_theta"]])
    csvw(TAB / "daily_trajectories.csv", hdr, rows)
    prow, srow, brow, drow = [], [], [], []
    for k in KEYS:
        for pn in ("before_March1", "March1_onward"):
            for sub in ("upper_two", "all_three", "501", "502", "503"):
                for kind in ("raw", "aligned"):
                    s_ = PS[k][pn][kind][sub]
                    prow.append([k, pn, sub, kind, s_["n"], s_["rmse"], s_["bias"]])
        for kind in ("raw", "aligned", "persistence"):
            for v in LV + ["pooled"]:
                srow.append([k, kind, v, S[k][kind][v]["n"], S[k][kind][v]["rmse"], S[k][kind][v]["bias"]])
        b = B[k]
        brow.append([k, b["rain"], b["intake"], b["runoff"], b["evap"], b["drain"], b["final"], b["grav"][0], b["grav"][1], -b["diff_down"][0], -b["diff_down"][1]])
        d_ = DE[k]
        drow.append([k, d_["pre"], d_["post"], d_["jan25_model"], d_["jan25_obs"], d_["peak"], d_["peak_close"], d_["storm_grav"][0], d_["storm_grav"][1], d_["storm_drain"]])
    csvw(TAB / "period_scores_all.csv", ["case", "period", "sensor_group", "scoring", "n", "rmse_pp", "bias_pp"], prow)
    csvw(TAB / "scores_all.csv", ["case", "scoring", "sensor", "n", "rmse_pp", "bias_pp"], srow)
    csvw(TAB / "budgets_all.csv", ["case", "rain_mm", "intake_mm", "runoff_mm", "evaporation_mm", "bottom_drainage_mm", "final_storage_mm", "upper_interface_crossing_mm", "lower_interface_crossing_mm", "net_upward_diffusion_top_mid_mm", "net_upward_diffusion_mid_deep_mm"], brow)
    csvw(TAB / "deep_wetting_all.csv", ["case", "deep_prestorm_pct", "deep_poststorm_pct", "deep_jan25_model_pct", "deep_jan25_observed_pct", "deep_peak_pct", "deep_peak_close", "storm_upper_crossing_mm", "storm_lower_crossing_mm", "storm_bottom_drainage_mm"], drow)
    for k in KEYS:
        shutil.copyfile(TASK / "study" / k / "case_numba.json", DATA / f"case_numba_{k}.json")
        shutil.copyfile(TASK / "study" / k / "root_decimal_verification.json", DATA / f"root_decimal_verification_{k}.json")
        shutil.copyfile(TASK / "study" / k / "actual_numba/timing.json", DATA / f"timing_numba_{k}.json")
    extra = [n for n in ("root_restart_verification.json", "root_restart_seal.json") if (TASK / n).exists()]
    for name in ("root_result_audit.json", "root_period_scores.json", "root_deep_wetting.json", "root_independent_inputs.json", "root_parameter_verification.json", "root_report_ready.json", *extra):
        shutil.copyfile(TASK / name, DATA / name)
    blk = blocks(R, S, B, PS, EP, DE)
    to_docx(blk, T, DOCX)
    info = verify_docx(DOCX, blk)
    seal_check()                                                    # re-hash after consuming
    if R["rseal"]:
        restart_seal_check(R["rseal"])
    (HERE / "report.md").write_text(to_markdown(blk, T), newline="\n")
    (HERE / "verification.md").write_text("# Verification record\n\nBefore building, `root_report_ready.json` was required to say PASS and all files of `root_run_seal.json` were re-hashed; they were re-hashed again after the document was built. From the daily rows of all fourteen cases the builder recomputed per-depth and pooled RAW, aligned and no-change "
                                          "RMSE/bias/counts, the before/after-March-1 period scores (nominal day label < / >= 2024-03-01), the same-timestamp endpoint values, the January 23 deep-layer response and its peak, water budgets and closure, the zero runoff/ponding/onset/room-rejection "
                                          "checks, NumPy = Numba row equality, the 900 s maximum differences, the before/after source pins of all 42 runs (SYRUP 4e38a472.../180, MAPLE fbc504bb.../412), Root's 90-digit controls, the sensor values against Root's independent raw-file audit, the "
                                          "exact uniform field-capacity and stress values in every case file, unchanged conductivities, saturation and physics, and the unrescaled initial storage of 22.9424 mm; all agree with Root's audit files. Static DOCX checks: zip integrity, one media file per figure, "
                                          "table count equals repeated-header count, footer page field, no placeholder text. No Office renderer was available, so rendered pagination and legibility were not checked (figures were viewed as PNG).\n\n"
                                          + restart_note(R) + "```\n" + json.dumps(info, indent=1) + "\n```\n", newline="\n")
    man_files = {}
    for sub in (FIG, TAB, DATA):
        for p in sorted(sub.iterdir()):
            if p.is_file() and p.suffix in (".png", ".csv", ".json", ".md"):
                man_files[f"{sub.name}/{p.name}"] = sha(p)
    man = {"docx": {"file": DOCX.name, **info}, "files": man_files, "builder": sha(Path(__file__)), "report_md": sha(HERE / "report.md"), "verification_md": sha(HERE / "verification.md"), "seal_files_rechecked": len(R["seal"]),
           "restart_seal_files_rechecked": len(R["rseal"] or {}), "excluded": "cache directories and ignored files (.ruff_cache, __pycache__, temp) are never listed"}
    (HERE / "artifact_manifest.json").write_text(json.dumps(man, indent=1) + "\n", newline="\n")
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
