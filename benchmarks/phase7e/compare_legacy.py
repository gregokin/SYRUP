"""Compare the Python legacy replay with the actual Fortran reference (full-precision audit ledger + rounded outputs)
and with SYRUP characteristic runs. Reports measurements; imposes no pass mark."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

FORTRAN_LEDGER = Path("outputs/phase7b/mahleran_ledger_run_v3/Output/syrup_sediment_ledger.dat")
FORTRAN_OUTPUT = Path("outputs/phase7/mahleran_deterministic_ksat_run/Output")


def number(value):
    return float(re.sub(r"(?<=\d)([+-]\d{3})$", r"e\1", value))


def load_fortran_ledger():
    data = np.loadtxt(FORTRAN_LEDGER, converters=number).reshape(5400, 6, 14)
    names = ["pickup", "deposition_active", "deposition_outside_active", "effective_clip_source", "old_mobile",
             "new_mobile", "cn_export", "endpoint_export", "net_cell_flux", "algebra_residual", "internal_flux_residual"]
    return {name: data[:, :, i + 3] for i, name in enumerate(names)}, data[:, 0, 1]


def series_metrics(t, a, b):
    """a = reference, b = test; both per-second series."""
    cum_a, cum_b = np.cumsum(a), np.cumsum(b)
    def centroid(x):
        return float(np.dot(t, x) / x.sum()) if x.sum() > 0 else None
    def quantile_t(cum, q):
        return float(t[np.searchsorted(cum, q * cum[-1])]) if cum[-1] > 0 else None
    return {"total_ref": float(a.sum()), "total_test": float(b.sum()),
            "ratio_test_over_ref": float(b.sum() / a.sum()) if a.sum() else None,
            "peak_ref": float(a.max()), "peak_test": float(b.max()),
            "peak_time_ref_s": float(t[int(np.argmax(a))]), "peak_time_test_s": float(t[int(np.argmax(b))]),
            "centroid_ref_s": centroid(a), "centroid_test_s": centroid(b),
            "quantile_times_ref_s": {str(q): quantile_t(cum_a, q) for q in (0.1, 0.5, 0.9)},
            "quantile_times_test_s": {str(q): quantile_t(cum_b, q) for q in (0.1, 0.5, 0.9)},
            "rmse": float(np.sqrt(np.mean((a - b) ** 2))), "max_abs_diff": float(np.abs(a - b).max()),
            "cumulative_rmse": float(np.sqrt(np.mean((cum_a - cum_b) ** 2))),
            "nash_sutcliffe": float(1 - np.sum((a - b) ** 2) / np.sum((a - a.mean()) ** 2)) if a.var() > 0 else None}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--legacy", type=Path, required=True, help="run_legacy_benchmark.py output directory")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--syrup", type=Path, nargs="*", default=[], help="SYRUP benchmark run directories (hydrograph.npz)")
    args = p.parse_args()
    with np.load(args.legacy / "legacy_ledger.npz", allow_pickle=False) as d:
        t = d["t_s"]; ledger = d["ledger"]; columns = list(d["columns"]); water_q = d["water_outlet_m3_s"]
    summary = json.loads((args.legacy / "legacy_summary.json").read_text())
    F, tf = load_fortran_ledger()
    assert np.array_equal(tf, t), "time bases differ"
    col = {name: ledger[:, :, i] for i, name in enumerate(columns)}
    pairs = {"pickup": "pickup_kg", "deposition_active": "deposition_active_kg",
             "deposition_outside_active": "deposition_outside_active_kg", "effective_clip_source": "effective_clip_source_kg",
             "cn_export": "cn_export_kg", "endpoint_export": "endpoint_export_kg"}
    by_class = {}
    for f_name, p_name in pairs.items():
        ref = F[f_name].sum(0); test = col[p_name].sum(0)
        by_class[f_name] = {"fortran_kg": ref.tolist(), "python_kg": test.tolist(),
                            "ratio_python_over_fortran": [float(b / a) if a else None for a, b in zip(ref, test)],
                            "total_fortran_kg": float(ref.sum()), "total_python_kg": float(test.sum()),
                            "total_ratio": float(test.sum() / ref.sum()) if ref.sum() else None}
    final_mobile = {"fortran_kg": F["new_mobile"][-1].tolist(), "python_kg": col["new_mobile_kg"][-1].tolist()}
    # per-second outlet series: Fortran endpoint export per step (full precision) vs Python endpoint export
    series = {"endpoint_export_total": series_metrics(t, F["endpoint_export"].sum(1), col["endpoint_export_kg"].sum(1)),
              "cn_export_total": series_metrics(t, F["cn_export"].sum(1), col["cn_export_kg"].sum(1)),
              "pickup_total": series_metrics(t, F["pickup"].sum(1), col["pickup_kg"].sum(1)),
              "by_class_endpoint_export": {str(k + 1): series_metrics(t, F["endpoint_export"][:, k], col["endpoint_export_kg"][:, k])
                                           for k in range(6)}}
    # rounded legacy stock outputs (sedtr001.dat column 2 kg/s; hydro001 column 3 mm^3/s)
    sedtr = np.loadtxt(FORTRAN_OUTPUT / "sedtr001.dat"); hydro = np.loadtxt(FORTRAN_OUTPUT / "hydro001.dat")
    rounded = {"sedtr_vs_python_outlet_flux": series_metrics(t, sedtr[:, 1], col["outlet_flux_kg_s"].sum(1)),
               "water_outlet_fortran_vs_syrup_hydrology": series_metrics(t, hydro[:, 2] * 1e-9, water_q)}
    syrup = {}
    for run in args.syrup:
        with np.load(run / "hydrograph.npz") as h:
            cum = h["sed_cumulative_export_kg"]; ts = h["sed_t_s"]
        mass = np.diff(cum, prepend=0.0)
        if ts.size == t.size:
            syrup[str(run)] = {"vs_fortran_endpoint_export": series_metrics(t, F["endpoint_export"].sum(1), mass),
                               "vs_python_legacy_endpoint_export": series_metrics(t, col["endpoint_export_kg"].sum(1), mass)}
    report = {"legacy_run": str(args.legacy), "legacy_status": summary["status"], "fortran_ledger": str(FORTRAN_LEDGER),
              "depth_time_level": summary["depth_time_level"], "kernel_implementation": summary["kernel_implementation"],
              "storm_totals_by_class": by_class, "final_mobile": final_mobile, "series": series, "rounded_outputs": rounded,
              "syrup_runs": syrup, "python_identity_max_abs_residual_kg": summary["max_abs_identity_residual_kg"],
              "ring_accounting_note": "deposition_outside_active is the ring walk diagnostic, not an additional sink in the active-pool CN balance; do not add it to export.",
              "note": "Fortran values are the actual reference program (no-splash, deterministic K); Python values replay the "
                      "legacy operator on SYRUP's hydrology. Differences include hydrology (matched to -0.38% outlet integral), "
                      "and documented law-convention departures (docs/phase5/physics.md). Clipping source is an artificial mass "
                      "source in BOTH; neither is a conservation-correct target."}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"storm_totals_by_class": {k: {"total_fortran_kg": v["total_fortran_kg"], "total_python_kg": v["total_python_kg"], "total_ratio": v["total_ratio"]} for k, v in by_class.items()},
                      "endpoint_export_total": series["endpoint_export_total"], "final_mobile": final_mobile}, indent=2))


if __name__ == "__main__":
    main()
