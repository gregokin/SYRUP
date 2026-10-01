"""Condense the immutable-candidate Plot1 trials without changing their outputs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

RUNS = ("trial1_b32_dt1", "trial1_b64_dt1", "trial1_b128_dt1", "trial1_b32_dt0p5")
rows = []
for name in RUNS:
    root = Path("outputs/phase7b") / name
    summary = json.loads((root / "benchmark_summary.json").read_text())
    sediment = summary["sediment"]
    assert sediment["closure"]["closed"] and sediment["closure"]["request_reconciled"]
    with np.load(root / "hydrograph.npz") as data:
        t = data["sed_t_s"]
        cumulative = data["sed_cumulative_export_kg"]
        mass = np.diff(cumulative, prepend=0)
        assert np.all(np.diff(t) == 1), "diagnostics below use one-second reporting bins"
        assert np.all(mass >= 0)
        mean_t = float(np.dot(t, mass) / mass.sum())
        quantile_t = {str(q): float(t[np.searchsorted(cumulative, q * cumulative[-1])]) for q in (0.1, 0.5, 0.9)}
        rates = {}
        for width in (1, 5, 10, 30, 60):
            rate = np.convolve(mass, np.ones(width) / width, mode="valid")
            index = int(np.argmax(rate))
            rates[str(width)] = {"peak_kg_s": float(rate[index]),
                                 "window_center_s": float(t[index] + (width - 1) / 2)}
        with np.load("outputs/phase7/syrup_matched_dt1/hydrograph.npz") as baseline:
            water_keys = [key for key in baseline.files if not key.startswith("sed_")]
            # Step-count, rejection and numerical-residual diagnostics may change
            # with dt; compare physical fields only for the identical dt=1 runs.
            water_equal = ({key: bool(np.array_equal(data[key], baseline[key])) for key in water_keys}
                           if name.endswith("dt1") else None)
            if water_equal is not None:
                assert all(water_equal.values())
    rows.append({
        "run": str(root), "summary_sha256": hashlib.sha256((root / "benchmark_summary.json").read_bytes()).hexdigest(),
        "transport": sediment["transport"], "export_kg": sediment["totals"]["export_actual"],
        "export_by_class_kg": sediment["by_class"]["export_actual"],
        "actual_pickup_kg": sediment["totals"]["actual_pickup"],
        "true_peak_export_rate_kg_s": sediment["peak_export_rate_kg_s"],
        "true_peak_time_s": sediment["time_of_peak_export_s"],
        "endpoint_weighted_centroid_s": mean_t, "cumulative_quantile_times_s": quantile_t,
        "reporting_window_peak_rates": rates, "water_arrays_equal_to_original_dt1": water_equal,
        "closure": sediment["closure"], "performance_observational": summary["performance"],
        "source": summary["provenance"],
    })
reference = json.loads(Path("outputs/phase7b/trial1_comparison/comparison.json").read_text())["sediment"]
report = {
    "scope": "Frozen Plot1, original wet equations and actual MAPLE bed; characteristic transport correction.",
    "runs": rows, "reference_sediment": reference,
    "bin_32_to_128_relative_export_change": rows[2]["export_kg"] / rows[0]["export_kg"] - 1,
    "dt_half_relative_export_change_at_32_bins": rows[3]["export_kg"] / rows[0]["export_kg"] - 1,
    "sampling_note": "Centroids use one-second interval exports placed at interval endpoints; quantiles have one-second resolution. Window rates use cumulative differences, not a subsampled instantaneous rate. All raw true peaks remain reported; smoothing is diagnostic and does not replace the raw convergence test.",
    "timing_note": "Observational runs; some overlap other verification. These are not controlled CPU performance qualifications.",
    "reference_note": "Legacy clipping creates mobile mass. Its outlet export remains a comparison, not a conservation-correct calibration target.",
}
Path("benchmarks/phase7b/characteristic_qualification.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({key: report[key] for key in ("bin_32_to_128_relative_export_change", "dt_half_relative_export_change_at_32_bins")}, indent=2))
