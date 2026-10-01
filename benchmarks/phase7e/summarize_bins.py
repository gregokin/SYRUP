"""Condense the Phase 7e characteristic-bin sweep (accepted dependency) against predeclared thresholds.

Convergence reference: the 128-bin run (finest). Production default: 32 bins. Thresholds are DIAGNOSTIC
(1% and 5% relative); the smallest bin count meeting each threshold is reported per metric. Legacy
(Fortran reference / Python replay) values are reported as a separate method comparison, never as a
convergence target (the legacy clipping source is an artificial mass source).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

BINS = (1, 2, 4, 8, 16, 32, 64, 128)
THRESHOLDS = (0.01, 0.05)


def load(root: Path, b: int):
    d = root / f"b{b}"
    summary = json.loads((d / "benchmark_summary.json").read_text())
    comp = json.loads((d / "component_times.json").read_text())
    with np.load(d / "hydrograph.npz") as h:
        t = h["sed_t_s"]; cum = h["sed_cumulative_export_kg"]
        keys = [f"sed_cumulative_export_c{i}_kg" for i in range(1, 7)]
        missing = [k for k in keys if k not in h.files]
        if missing:
            raise KeyError(f"{d}: hydrograph lacks per-class cumulative export fields {missing}")
        cum_by_class = np.stack([h[k] for k in keys], 1)
    with np.load(d / "final_state.npz") as f:
        dep = f["cumulative_deposition_kg"]; pick = f["cumulative_pickup_kg"]; bed_change = f["bed_change_kg"]
    mass = np.diff(cum, prepend=0.0)
    assert np.all(np.diff(t) == 1.0)
    windows = {}
    for w in (1, 5, 10, 30, 60):
        rate = np.convolve(mass, np.ones(w) / w, mode="valid")
        i = int(np.argmax(rate)); windows[str(w)] = {"peak_kg_s": float(rate[i]), "center_s": float(t[i] + (w - 1) / 2)}
    sed = summary["sediment"]
    return {
        "bins": b, "export_kg": sed["totals"]["export_actual"], "export_by_class_kg": sed["by_class"]["export_actual"],
        "actual_pickup_kg": sed["totals"]["actual_pickup"], "deposition_kg": sed["totals"]["deposition_actual"],
        "deposition_by_class_kg": sed["by_class"]["deposition_actual"],
        "closed": bool(sed["closure"]["closed"] and sed["closure"]["request_reconciled"]),
        "closure": sed["closure"], "final_mobile_kg": sed["final_mobile_kg"],
        "raw_peak_kg_s": sed["peak_export_rate_kg_s"], "raw_peak_time_s": sed["time_of_peak_export_s"],
        "windowed_peaks": windows,
        "centroid_s": float(np.dot(t, mass) / mass.sum()),
        "quantile_times_s": {str(q): float(t[np.searchsorted(cum, q * cum[-1])]) for q in (0.1, 0.5, 0.9)},
        "cumulative_curve": cum, "cumulative_curve_by_class": cum_by_class, "export_mass_series": mass,
        "deposition_map": dep.sum(-1), "net_bed_change_map": bed_change.sum(-1),
        "deposition_map_by_class": dep, "pickup_map": pick.sum(-1),
        "loop_wall_s": summary["performance"]["step_loop"]["wall_s"],
        "characteristic_s": comp["component_timers"]["characteristic_step"]["wall_s"],
        "maple_s": comp["component_timers"]["apply_bed_demand"]["wall_s"],
        "peak_rss_end_kib": summary["performance"]["peak_rss_kib"]["process_end"],
        "peak_rss_start_kib": summary["performance"]["peak_rss_kib"]["process_start"],
        "n_rejected": summary["time"]["n_rejected_attempts"], "max_substeps": summary["time"]["max_transport_substeps_used"],
        "provenance": {"syrup": summary["provenance"]["maple_syrup"]["package_source_digest"]["digest_sha256"],
                       "maple": summary["provenance"]["maple"]["package_source_digest"]["digest_sha256"]},
    }


def rel(a, b):
    return float(abs(a - b) / abs(b)) if b != 0 else None


def rel_l2(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b)) if np.linalg.norm(b) > 0 else None


def max_gap_over_final(a, b):
    """Maximum absolute cumulative-curve gap divided by the reference's final value (timing-sensitive)."""
    return float(np.abs(a - b).max() / b[-1]) if b[-1] > 0 else None


def legacy_series_comparison(t, test_mass, test_by_class, fortran):
    """Every bin against the actual Fortran legacy per-second endpoint export (full-precision audit ledger)."""
    ref = fortran["endpoint_export"].sum(1); ref_cum = np.cumsum(ref); test_cum = np.cumsum(test_mass)
    def windows(x):
        return {str(w): float(np.convolve(x, np.ones(w) / w, mode="valid").max()) for w in (1, 10, 60)}
    def quant(cum, q):
        return float(t[np.searchsorted(cum, q * cum[-1])])
    return {
        "legacy_total_kg": float(ref.sum()), "test_total_kg": float(test_mass.sum()), "ratio_test_over_legacy": float(test_mass.sum() / ref.sum()),
        "by_class_ratio": [float(b / a) if a > 1e-12 else None for a, b in zip(fortran["endpoint_export"].sum(0), test_by_class[-1])],
        "legacy_by_class_kg": fortran["endpoint_export"].sum(0).tolist(), "test_by_class_kg": test_by_class[-1].tolist(),
        "centroid_legacy_s": float(np.dot(t, ref) / ref.sum()), "centroid_test_s": float(np.dot(t, test_mass) / test_mass.sum()),
        "quantile_times_legacy_s": {str(q): quant(ref_cum, q) for q in (0.1, 0.5, 0.9)},
        "quantile_times_test_s": {str(q): quant(test_cum, q) for q in (0.1, 0.5, 0.9)},
        "peak_windows_legacy_kg_s": windows(ref), "peak_windows_test_kg_s": windows(test_mass),
        "max_abs_cumulative_gap_over_legacy_final": float(np.abs(test_cum - ref_cum).max() / ref_cum[-1]),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("outputs/phase7e/sweep"))
    p.add_argument("--output", type=Path, default=Path("benchmarks/phase7e/bin_sweep_qualification.json"))
    p.add_argument("--legacy", type=Path, default=None, help="Python legacy replay dir (legacy_ledger.npz)")
    args = p.parse_args()
    runs = {b: load(args.root, b) for b in BINS}  # incomplete sweeps must fail
    ref = runs[128]; default = runs[32]
    metrics = {}
    for b, r in runs.items():
        m = {
            "export_total": rel(r["export_kg"], ref["export_kg"]),
            "export_by_class_max": max(rel(x, y) for x, y in zip(r["export_by_class_kg"], ref["export_by_class_kg"]) if y > 1e-9),
            "deposition_total": rel(r["deposition_kg"], ref["deposition_kg"]),
            "deposition_map_rel_l2": rel_l2(r["deposition_map"], ref["deposition_map"]),
            "net_bed_change_map_rel_l2": rel_l2(r["net_bed_change_map"], ref["net_bed_change_map"]),
            "cumulative_curve_rel_l2": rel_l2(r["cumulative_curve"], ref["cumulative_curve"]),
            "cumulative_curve_max_gap_over_final": max_gap_over_final(r["cumulative_curve"], ref["cumulative_curve"]),
            "cumulative_curve_by_class_max_gap_over_final": max(
                max_gap_over_final(r["cumulative_curve_by_class"][:, k], ref["cumulative_curve_by_class"][:, k])
                for k in range(6) if ref["cumulative_curve_by_class"][-1, k] > 1e-9),
            "vs_default32_cumulative_max_gap_over_final": max_gap_over_final(r["cumulative_curve"], default["cumulative_curve"]),
            "vs_default32_cumulative_rel_l2": rel_l2(r["cumulative_curve"], default["cumulative_curve"]),
            "centroid_s_abs": abs(r["centroid_s"] - ref["centroid_s"]),
            "centroid_rel": rel(r["centroid_s"], ref["centroid_s"]),
            "quantile_time_abs_s_max": max(abs(r["quantile_times_s"][q] - ref["quantile_times_s"][q]) for q in ("0.1", "0.5", "0.9")),
            "raw_peak": rel(r["raw_peak_kg_s"], ref["raw_peak_kg_s"]),
            "peak_10s_window": rel(r["windowed_peaks"]["10"]["peak_kg_s"], ref["windowed_peaks"]["10"]["peak_kg_s"]),
            "peak_60s_window": rel(r["windowed_peaks"]["60"]["peak_kg_s"], ref["windowed_peaks"]["60"]["peak_kg_s"]),
            "vs_default32_export_total": rel(r["export_kg"], default["export_kg"]),
        }
        metrics[b] = m
    smallest = {}
    metric_names = ["export_total", "export_by_class_max", "deposition_total", "deposition_map_rel_l2",
                    "net_bed_change_map_rel_l2", "cumulative_curve_rel_l2", "cumulative_curve_max_gap_over_final",
                    "cumulative_curve_by_class_max_gap_over_final", "centroid_rel", "raw_peak", "peak_10s_window", "peak_60s_window"]
    for thr in THRESHOLDS:
        smallest[str(thr)] = {}
        for name in metric_names:
            ok = [b for b in sorted(runs) if b < 128 and metrics[b][name] is not None and metrics[b][name] <= thr
                  and all(metrics[c][name] <= thr for c in sorted(runs) if b <= c < 128)]  # monotone: all coarser-than-ref bins >= b pass
            smallest[str(thr)][name] = ok[0] if ok else None
    legacy = None
    if args.legacy is not None and (args.legacy / "legacy_ledger.npz").exists():
        with np.load(args.legacy / "legacy_ledger.npz") as d:
            cols = list(d["columns"]); led = d["ledger"]
        legacy = {"python_endpoint_export_kg": float(led[:, :, cols.index("endpoint_export_kg")].sum()),
                  "python_by_class_kg": led[:, :, cols.index("endpoint_export_kg")].sum(0).tolist()}
    fortran_sums = json.loads(Path("benchmarks/phase7b/mahleran_ledger_audit.json").read_text())["sums_by_class_kg"]
    import re
    raw = np.loadtxt(Path("outputs/phase7b/mahleran_ledger_run_v3/Output/syrup_sediment_ledger.dat"),
                     converters=lambda v: float(re.sub(r"(?<=\d)([+-]\d{3})$", r"e\1", v))).reshape(5400, 6, 14)
    fortran = {"endpoint_export": raw[:, :, 10], "cn_export": raw[:, :, 9]}
    t_ref = raw[:, 0, 1]
    legacy_per_bin = {b: legacy_series_comparison(t_ref, r["export_mass_series"], r["cumulative_curve_by_class"], fortran)
                      for b, r in runs.items() if r["cumulative_curve"].size == 5400}
    report = {
        "scope": "Frozen Plot1, accepted MAPLE 72310c49 and SYRUP source as recorded; characteristic bins 1..128.",
        "thresholds": list(THRESHOLDS), "reference_bins": 128, "production_default_bins": 32,
        "runs": {b: {k: v for k, v in r.items() if not isinstance(v, np.ndarray)} for b, r in runs.items()},
        "relative_to_128": metrics, "smallest_bins_meeting_threshold": smallest,
        "all_class_budgets_closed": all(r["closed"] for r in runs.values()),
        "legacy_fortran_endpoint_export_by_class_kg": fortran_sums["endpoint_export"],
        "legacy_fortran_endpoint_export_kg": float(np.sum(fortran_sums["endpoint_export"])),
        "every_bin_vs_fortran_legacy": legacy_per_bin,
        "legacy_python_replay": legacy,
        "timing_note": "b1..b32 overlapped short profiling and unit tests of the Stage 1 candidate (seconds to tens of seconds); "
                       "b64/b128 ran alone. All timings are single sequential same-machine observations.",
        "method_note": "Bin convergence (vs 128) and legacy disagreement are different comparisons; no bin count is chosen to match legacy.",
    }
    args.output.write_text(json.dumps(report, indent=2, default=float) + "\n")
    for b in sorted(runs):
        m = metrics[b]
        print(b, {k: (round(v, 5) if isinstance(v, float) else v) for k, v in m.items() if k in ("export_total", "cumulative_curve_max_gap_over_final", "vs_default32_cumulative_max_gap_over_final", "raw_peak", "peak_10s_window", "centroid_s_abs")},
              "legacy_ratio", round(legacy_per_bin[b]["ratio_test_over_legacy"], 4) if b in legacy_per_bin else None)
    print(json.dumps(smallest, indent=1))


if __name__ == "__main__":
    main()
