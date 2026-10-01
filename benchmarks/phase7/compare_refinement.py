"""Timestep refinement and cost table for SYRUP frozen-geometry benchmark runs.

    python benchmarks/phase7/compare_refinement.py --runs DT1 DT0P5 DT0P25 --output NEW.json

All runs must share the case, forcing, conductivity, window, cadence and
frozen mode; only `max_dt_s` (and the hydraulic implementation) may differ.
The finest step is the numerical reference of the study, not ground truth.
Timings are observational (one run each, possibly overlapping other work).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SCHEMA = "maple-syrup-phase7-refinement/1"
SHARED = ("maple_case_identity_sha256",)
SHARED_SYRUP = ("original_rainfall_sha256", "applied_rainfall_sha256", "conductivity_mm_s", "graph_input_sha256",
                "end_s", "report_every_s", "commit", "force_final_commit", "sediment_courant_max",
                "max_transport_substeps")


def load(run: Path) -> dict:
    summary = json.loads((run / "benchmark_summary.json").read_text())
    if summary.get("schema") != "maple_syrup.plot1_matched_benchmark.v1" or not summary["mode"]["frozen_hydraulic_geometry"]:
        raise ValueError(f"{run} is not a frozen-geometry benchmark output")
    with np.load(run / "hydrograph.npz") as h:
        hydro = {k: h[k].copy() for k in ("t_s", "outlet_discharge_m3_s", "cumulative_export_m3",
                                          "sed_cumulative_export_kg", "sed_export_rate_kg_s")}
    with np.load(run / "final_state.npz") as f:
        final = {k: f[k].copy() for k in ("bed_change_kg", "peak_depth_m", "peak_velocity_m_s", "committed_elevation_m",
                                          "initial_committed_elevation_m")}
    return {"path": str(run), "summary": summary, "hydro": hydro, "final": final}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"{args.output} exists")
    runs = sorted((load(p.resolve()) for p in args.runs), key=lambda r: -r["summary"]["time"]["max_dt_s"])
    base = runs[0]["summary"]["resolved_config"]
    for r in runs[1:]:
        cfg = r["summary"]["resolved_config"]
        if any(cfg[k] != base[k] for k in SHARED) or any(cfg["syrup"][k] != base["syrup"][k] for k in SHARED_SYRUP):
            raise ValueError(f"{r['path']} differs from {runs[0]['path']} beyond the timestep")
        if not np.array_equal(r["hydro"]["t_s"], runs[0]["hydro"]["t_s"]):
            raise ValueError("runs report on different time axes")
        if not np.array_equal(r["final"]["committed_elevation_m"], r["final"]["initial_committed_elevation_m"]):
            raise ValueError(f"{r['path']} changed its committed elevation; not a frozen run")
    finest = runs[-1]
    rows = []
    for r in runs:
        s, h = r["summary"], r["hydro"]
        export_class = np.array([float(h["sed_cumulative_export_kg"][-1])] + [
            s["sediment"]["closure"]["export_actual_kg"][k] for k in range(6)])
        fine_class = np.array([float(finest["hydro"]["sed_cumulative_export_kg"][-1])] + [
            finest["summary"]["sediment"]["closure"]["export_actual_kg"][k] for k in range(6)])
        q, qf = h["outlet_discharge_m3_s"], finest["hydro"]["outlet_discharge_m3_s"]
        dz = r["final"]["bed_change_kg"].sum(axis=-1) - finest["final"]["bed_change_kg"].sum(axis=-1)
        perf = s["performance"]
        rows.append({
            "path": r["path"], "max_dt_s": s["time"]["max_dt_s"], "implementation": s["provenance"]["implementation"],
            "n_accepted_steps": s["time"]["n_accepted_steps"], "n_rejected_attempts": s["time"]["n_rejected_attempts"],
            "max_transport_substeps_used": s["time"]["max_transport_substeps_used"],
            "water_export_ledger_m3": float(h["cumulative_export_m3"][-1]),
            "water_export_ledger_relative_difference_vs_finest": float(
                (h["cumulative_export_m3"][-1] - finest["hydro"]["cumulative_export_m3"][-1])
                / finest["hydro"]["cumulative_export_m3"][-1]) if finest["hydro"]["cumulative_export_m3"][-1] else None,
            "outlet_rmse_vs_finest_m3_s": float(np.sqrt(np.mean((q - qf) ** 2))),
            "peak_outlet_m3_s": s["final_state"]["peak_outlet_discharge_m3_s"],
            "time_of_peak_outlet_s": s["final_state"]["time_of_peak_outlet_discharge_s"],
            "sediment_export_kg_total_and_classes": export_class.tolist(),
            "sediment_export_relative_difference_vs_finest": [
                float((a - b) / b) if b else None for a, b in zip(export_class, fine_class)],
            "peak_export_rate_kg_s": s["sediment"]["peak_export_rate_kg_s"],
            "time_of_peak_export_s": s["sediment"]["time_of_peak_export_s"],
            "net_bed_mass_change_kg": s["sediment"]["net_bed_mass_change_by_class_kg"],
            "max_abs_bed_change_difference_vs_finest_kg": float(np.max(np.abs(dz))),
            "max_sediment_courant": s["sediment"]["max_sediment_courant"],
            "closure_residual_max_kg": float(np.max(np.abs(s["sediment"]["closure"]["residual_kg"]))),
            "closure_tolerance_kg": s["sediment"]["closure"]["tolerance_kg"][0],
            "water_residual_m3": s["budget"]["water_residual_m3"], "water_tolerance_m3": s["budget"]["tolerance_m3"],
            "timing": {"setup_wall_s": perf["setup_and_verification"]["wall_s"],
                       "first_step_wall_s_incl_jit": perf["first_accepted_step"]["wall_s"],
                       "remaining_steps_wall_s": perf["remaining_steps"]["wall_s"],
                       "mean_wall_s_per_step": perf["remaining_steps"]["mean_wall_s_per_step"],
                       "peak_rss_kib_end": perf["peak_rss_kib"]["process_end"]},
        })
    report = {"schema": SCHEMA, "reference_rule": "finest timestep is the numerical reference, not ground truth",
              "shared_identity": {k: base[k] for k in SHARED}, "runs": rows,
              "note": "observational timings; not a controlled benchmark; GPU not exercised"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps([{k: row[k] for k in ("max_dt_s", "water_export_ledger_m3",
                                            "water_export_ledger_relative_difference_vs_finest",
                                            "sediment_export_relative_difference_vs_finest", "timing")}
                      for row in rows], indent=2))


if __name__ == "__main__":
    main()
