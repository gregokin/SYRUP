"""Compare a SYRUP frozen-geometry benchmark with the whole-program MAHLERAN deterministic run.

    python benchmarks/phase7/compare_matched.py --syrup outputs/phase7/syrup_matched_dt1 \\
        --mahleran outputs/phase7/mahleran_deterministic_ksat_run --output NEW_DIR

Reads saved outputs only; executes neither model. Every MAHLERAN quantity is
a four-significant-digit endpoint sample (`e10.4`), so the comparison is
output-aware: SYRUP's instantaneous outlet discharge is compared with
MAHLERAN's `q_plot * dx` both raw and rounded to four significant digits, the
endpoint-sum integral is kept distinct from SYRUP's conservative face-volume
ledger, and peak timing accepts the rounded plateau. Legacy accumulated
detachment / net erosion (`detac001`, `neter001`: kg per cell of detachment
DEMAND and demand minus deposition) are compared with SYRUP's actual MAPLE
pickup and water class exchange AND, separately, with the actual bed
inventory change; none of these is a conservation statement about the stock
files. Targets are predeclared; a miss is reported for investigation, never
relaxed here.

`depth001.asc` / `veloc001.asc` are NOT per-cell storm maxima and are NOT
compared with SYRUP's per-cell peak depth / velocity: `output_hydro_data_xml.f90`
489-505 copies the whole-domain `d(2)` (mm) and `v` (mm/s) only when the
outlet discharge `q_plot` exceeds its running maximum, so the files hold the
synchronous snapshot at the step of the full-precision peak outlet
discharge. The comparator reports that definition, keeps both fields as
separately labelled diagnostics, and draws no depth/velocity panel. The
matching SYRUP quantity is a synchronous snapshot at its own peak-outlet
step (recorded by a read-only observer sidecar, compared separately).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np

SCHEMA = "maple-syrup-phase7-matched-comparison/1"
TARGETS = {
    "integrated_outlet_relative_difference": 0.01,
    "peak_outlet_relative_difference": 0.01,
    "peak_time_tolerance_s": 10.0,
}
MAHLERAN_INTERIOR = (60, 20)
ITERATION_RE = re.compile(r"Starting iteration\s+(\d+) rain intensity:\s+([\d.]+) time step:\s+([\d.]+)")
# Spatial fields drawn side by side. Depth / velocity are deliberately absent:
# the legacy maps are peak-outlet snapshots, not maxima (see module docstring).
SPATIAL_PANELS = (
    ("mahleran_detachment_kg", "syrup_pickup_kg", "detachment demand vs actual pickup (kg/cell)"),
    ("mahleran_net_erosion_kg", "syrup_water_net_erosion_kg", "net erosion (kg/cell)"),
)
PEAK_SNAPSHOT_DEFINITION = (
    "depth001.asc (mm) and veloc001.asc (mm/s) are the whole-domain d(2) and v copied by "
    "output_hydro_data_xml.f90 489-505 ONLY when the outlet discharge q_plot exceeds its running maximum: a "
    "synchronous snapshot at the step of the full-precision peak outlet discharge, not per-cell storm maxima"
)


def field_stats(values: np.ndarray) -> dict:
    a = np.asarray(values, dtype=np.float64)
    return {"sum": float(a.sum()), "max": float(a.max()), "mean": float(a.mean()),
            "n_positive": int(np.sum(a > 0.0))}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --- unit / orientation helpers (unit-tested) ---------------------------------------------------------------
def round_sig(values, digits: int = 4) -> np.ndarray:
    """Round like Fortran `e10.4` prints: `digits` significant figures."""
    out = np.array(values, dtype=np.float64).reshape(-1)
    for i, v in enumerate(out):
        out[i] = 0.0 if v == 0.0 else float(f"{v:.{digits - 1}E}")
    return out.reshape(np.shape(values))


def mahleran_asc_interior(values: np.ndarray, shape=MAHLERAN_INTERIOR) -> np.ndarray:
    """Crop the exterior ring and reverse north-first rows into MAPLE's
    south-first order; refuse nodata inside the physical interior."""
    a = np.asarray(values, dtype=np.float64)
    if a.shape != (shape[0] + 2, shape[1] + 2):
        raise ValueError(f"ASCII grid shape {a.shape} is not interior {shape} plus a one-cell ring")
    interior = a[1:-1, 1:-1][::-1].copy()
    if np.any(interior == -9999.0) or not np.all(np.isfinite(interior)):
        raise ValueError("nodata or non-finite value inside the physical interior")
    return interior


def read_asc(path: Path, shape=MAHLERAN_INTERIOR) -> np.ndarray:
    return mahleran_asc_interior(np.loadtxt(path, skiprows=6), shape)


def endpoint_integral_m3(t: np.ndarray, q: np.ndarray) -> float:
    """Sum of endpoint discharge times the preceding interval (the stock
    MAHLERAN accounting). Distinct from a conservative face ledger."""
    dt = np.diff(np.concatenate(([0.0], t)))
    if np.any(dt <= 0.0):
        raise ValueError("times must be strictly increasing from 0")
    return float(np.sum(q * dt))


def plateau_s(t: np.ndarray, q: np.ndarray) -> tuple[float, float]:
    """First and last time at which `q` attains its maximum (a rounded
    output shows its true peak as a plateau)."""
    top = np.flatnonzero(q == q.max())
    return float(t[top[0]]), float(t[top[-1]])


def hydrograph_metrics(t: np.ndarray, reference: np.ndarray, test: np.ndarray) -> dict:
    """Output-aware discharge comparison; `reference` is the rounded stock
    series, `test` a full-precision series on the same times."""
    if reference.shape != test.shape or t.shape != test.shape:
        raise ValueError("series must share the time axis")
    ref_int, test_int = endpoint_integral_m3(t, reference), endpoint_integral_m3(t, test)
    rounded = round_sig(test)
    ref_plateau = plateau_s(t, reference)
    test_peak_t = float(t[int(np.argmax(test))])
    rounded_plateau = plateau_s(t, rounded)
    diff = test - reference
    return {
        "reference_endpoint_integral_m3": ref_int,
        "test_endpoint_integral_m3": test_int,
        "test_rounded_endpoint_integral_m3": endpoint_integral_m3(t, rounded),
        "integrated_relative_difference": (test_int - ref_int) / ref_int if ref_int else None,
        "reference_peak_m3_s": float(reference.max()), "test_peak_m3_s": float(test.max()),
        "peak_relative_difference": (float(test.max()) - float(reference.max())) / float(reference.max())
        if reference.max() else None,
        "reference_peak_plateau_s": list(ref_plateau), "test_peak_time_s": test_peak_t,
        "test_rounded_peak_plateau_s": list(rounded_plateau),
        "peak_time_offset_from_plateau_s": max(0.0, ref_plateau[0] - test_peak_t, test_peak_t - ref_plateau[1]),
        "rmse_m3_s": float(np.sqrt(np.mean(diff ** 2))),
        "max_abs_difference_m3_s": float(np.max(np.abs(diff))),
        "time_of_max_abs_difference_s": float(t[int(np.argmax(np.abs(diff)))]),
        "rounding_floor_m3_s": float(np.max(np.abs(rounded - test))),
        "nash_sutcliffe": 1.0 - float(np.sum(diff ** 2) / np.sum((reference - reference.mean()) ** 2))
        if np.any(reference != reference.mean()) else None,
    }


def field_metrics(reference: np.ndarray, test: np.ndarray) -> dict:
    ref, tst = np.asarray(reference, dtype=np.float64), np.asarray(test, dtype=np.float64)
    if ref.shape != tst.shape:
        raise ValueError("fields must share a shape")
    diff = tst - ref
    ref_norm = float(np.sqrt(np.sum(ref ** 2)))
    corr = None
    if ref.std() > 0.0 and tst.std() > 0.0:
        corr = float(np.corrcoef(ref.ravel(), tst.ravel())[0, 1])
    return {
        "reference_sum": float(ref.sum()), "test_sum": float(tst.sum()),
        "sum_ratio_test_over_reference": float(tst.sum() / ref.sum()) if ref.sum() else None,
        "reference_max": float(ref.max()), "test_max": float(tst.max()),
        "rmse": float(np.sqrt(np.mean(diff ** 2))), "max_abs_difference": float(np.max(np.abs(diff))),
        "relative_l2": float(np.sqrt(np.sum(diff ** 2)) / ref_norm) if ref_norm else None,
        "pearson_r": corr,
        "cell_of_max_abs_difference_maple_row_col": [int(v) for v in np.unravel_index(int(np.argmax(np.abs(diff))),
                                                                                       diff.shape)],
    }


def evaluate_targets(water: dict) -> dict:
    integrated = water["integrated_relative_difference"]
    peak = water["peak_relative_difference"]
    offset = water["peak_time_offset_from_plateau_s"]
    return {
        "integrated_outlet_relative_difference": {
            "target": TARGETS["integrated_outlet_relative_difference"], "value": integrated,
            "pass": integrated is not None and abs(integrated) <= TARGETS["integrated_outlet_relative_difference"]},
        "peak_outlet_relative_difference": {
            "target": TARGETS["peak_outlet_relative_difference"], "value": peak,
            "pass": peak is not None and abs(peak) <= TARGETS["peak_outlet_relative_difference"]},
        "peak_time_within_rounded_plateau": {
            "target_s": TARGETS["peak_time_tolerance_s"], "offset_s": offset,
            "pass": offset <= TARGETS["peak_time_tolerance_s"]},
        "policy": "a missed target is investigated (timestep, forcing, solver differences), never relaxed or "
                  "matched artificially; sediment and spatial metrics are measurements without a pass mark",
    }


# --- readers --------------------------------------------------------------------------------------------------
def load_syrup(syrup_dir: Path) -> dict:
    summary = json.loads((syrup_dir / "benchmark_summary.json").read_text())
    if summary.get("schema") != "maple_syrup.plot1_matched_benchmark.v1":
        raise ValueError("not a Phase 7 SYRUP matched-benchmark output")
    if not summary["mode"]["frozen_hydraulic_geometry"] or summary["mode"]["commit"] or summary["mode"]["force_final_commit"]:
        raise ValueError("SYRUP run is not a frozen-geometry benchmark")
    for name, digest in summary["outputs"].items():
        if sha256_file(syrup_dir / name) != digest:
            raise ValueError(f"SYRUP output {name} does not match its recorded hash")
    with np.load(syrup_dir / "hydrograph.npz") as h:
        hydro = {k: h[k].copy() for k in h.files}
    with np.load(syrup_dir / "final_state.npz") as f:
        final = {k: f[k].copy() for k in f.files}
    with np.load(syrup_dir / "forcing.npz") as f:
        forcing = {k: f[k].copy() for k in f.files}
    return {"summary": summary, "hydro": hydro, "final": final, "forcing": forcing}


def load_mahleran(mahleran_dir: Path) -> dict:
    execution = json.loads((mahleran_dir / "execution.json").read_text())
    if execution.get("returncode") != 0 or not execution.get("completion_marker"):
        raise ValueError("MAHLERAN run did not complete")
    hashes = {}
    for name, entry in execution["outputs"].items():
        if name.startswith("Output/"):
            hashes[name] = sha256_file(mahleran_dir / name)
            if hashes[name] != entry["sha256"]:
                raise ValueError(f"MAHLERAN output {name} does not match its execution manifest")
    out = mahleran_dir / "Output"
    hydro = np.loadtxt(out / "hydro001.dat")
    sed = np.loadtxt(out / "sedtr001.dat")
    classes = np.loadtxt(out / "seddisch001.dat")
    for values, cols in ((hydro, 12), (sed, 16), (classes, 7)):
        if values.ndim != 2 or values.shape[1] != cols or not np.isfinite(values).all():
            raise ValueError("unexpected MAHLERAN time-series layout")
    if not (np.array_equal(hydro[:, 0], sed[:, 0]) and np.array_equal(hydro[:, 0], classes[:, 0])):
        raise ValueError("MAHLERAN time series disagree on their time axis")
    log = (mahleran_dir / "stdout.log").read_text(errors="replace")
    applied = np.array(ITERATION_RE.findall(log), dtype=np.float64)
    if applied.shape[0] != hydro.shape[0] or not np.all(applied[:, 2] == 1.0):
        raise ValueError("MAHLERAN log does not carry one applied-rainfall line per 1 s step")
    ksat = read_asc(out / "ksat_001.asc")
    xml = (mahleran_dir / "mahleran_input.xml").read_text(encoding="latin-1")
    dist = re.search(r'<finalInfiltrationRateDistribution value="([^"]*)"', xml)
    return {
        "execution": execution, "hashes": hashes,
        "t_s": hydro[:, 0], "q_m3_s": hydro[:, 2] * 1.0e-9,  # q_plot * dx in mm3/s
        "sediment_kg_s": sed[:, 1], "class_kg_s": classes[:, 1:7],
        "applied_mm_h": applied[:, 1],
        "maps": {name: read_asc(out / f"{name}001.asc") for name in ("depth", "veloc", "detac", "neter", "dschg")},
        "ksat_mm_s": ksat, "distribution": None if dist is None else dist.group(1),
    }


def syrup_applied_per_second(forcing: dict, n: int) -> np.ndarray:
    """Expand SYRUP's compressed applied schedule to one value per whole
    second `[k, k+1)`, the resolution of the legacy log."""
    edges, intensity = forcing["edges_s"], forcing["intensity_mm_per_h"]
    starts = np.arange(n, dtype=np.float64)
    k = np.searchsorted(edges, starts, side="right") - 1
    inside = (k >= 0) & (k < intensity.size)
    return np.where(inside, intensity[np.clip(k, 0, intensity.size - 1)], 0.0)


# --- comparison ------------------------------------------------------------------------------------------------
def compare(syrup_dir: Path, mahleran_dir: Path) -> tuple[dict, dict]:
    s, m = load_syrup(syrup_dir), load_mahleran(mahleran_dir)
    summary, hydro, final = s["summary"], s["hydro"], s["final"]
    t = hydro["t_s"]
    n = int(m["t_s"].size)
    if t.size != n or not np.array_equal(t, m["t_s"]):
        raise ValueError(f"time axes differ: SYRUP {t.size} rows (cadence {summary['time']['report_every_s']} s), "
                         f"MAHLERAN {n} rows; the matched benchmark needs 1 s rows over the same window")
    # forcing identity: the SYRUP run must have used this run's logged applied rainfall
    syrup_rain = syrup_applied_per_second(s["forcing"], n)
    forcing_equal = bool(np.array_equal(syrup_rain, m["applied_mm_h"]))
    ksat = summary["conductivity"]["mm_s"]
    conductivity_equal = bool(np.all(m["ksat_mm_s"] == ksat))
    # water
    q_s = hydro["outlet_discharge_m3_s"]
    water = hydrograph_metrics(t, m["q_m3_s"], q_s)
    water["syrup_conservative_export_ledger_m3"] = float(hydro["cumulative_export_m3"][-1])
    water["syrup_ledger_minus_endpoint_integral_m3"] = float(hydro["cumulative_export_m3"][-1]) - water["test_endpoint_integral_m3"]
    water["reference_integral_through_1620s_m3"] = endpoint_integral_m3(t[t <= 1620], m["q_m3_s"][t <= 1620]) if n >= 1620 else None
    water["test_integral_through_1620s_m3"] = endpoint_integral_m3(t[t <= 1620], q_s[t <= 1620]) if n >= 1620 else None
    water["accounting"] = ("reference: stock endpoint outlet discharge summed with dt = 1 s; test: SYRUP "
                           "instantaneous outlet discharge on the same rows; the conservative face ledger is "
                           "reported beside it and is not the same quantity")
    # sediment
    sed_ref_total = endpoint_integral_m3(t, m["sediment_kg_s"])
    sed_ref_class = np.array([endpoint_integral_m3(t, m["class_kg_s"][:, k]) for k in range(6)])
    sed_test_total = float(hydro["sed_cumulative_export_kg"][-1])
    sed_test_class = np.array([float(hydro[f"sed_cumulative_export_c{k + 1}_kg"][-1]) for k in range(6)])
    rate_s = hydro["sed_export_rate_kg_s"]
    ref_rate_plateau = plateau_s(t, m["sediment_kg_s"])
    sediment = {
        "reference_total_export_kg_endpoint_sum": sed_ref_total,
        "reference_total_export_kg_class_sum": float(sed_ref_class.sum()),
        "reference_class_export_kg": sed_ref_class.tolist(),
        "test_total_export_kg_ledger": sed_test_total,
        "test_class_export_kg_ledger": sed_test_class.tolist(),
        "total_ratio_test_over_reference": sed_test_total / sed_ref_total if sed_ref_total else None,
        "class_ratio_test_over_reference": [float(a / b) if b else None for a, b in zip(sed_test_class, sed_ref_class)],
        "reference_exported_fraction": (sed_ref_class / sed_ref_class.sum()).tolist() if sed_ref_class.sum() else None,
        "test_exported_fraction": (sed_test_class / sed_test_class.sum()).tolist() if sed_test_class.sum() else None,
        "reference_peak_rate_kg_s": float(m["sediment_kg_s"].max()), "reference_peak_plateau_s": list(ref_rate_plateau),
        "test_peak_rate_kg_s": float(rate_s.max()), "test_peak_time_s": float(t[int(np.argmax(rate_s))]),
        "test_true_peak_rate_kg_s": summary["sediment"]["peak_export_rate_kg_s"],
        "test_true_peak_time_s": summary["sediment"]["time_of_peak_export_s"],
        "cumulative_rmse_kg": float(np.sqrt(np.mean((np.cumsum(m["sediment_kg_s"]) - hydro["sed_cumulative_export_kg"]) ** 2))),
        "reference_cumulative_at_1620s_kg": endpoint_integral_m3(t[t <= 1620], m["sediment_kg_s"][t <= 1620]) if n >= 1620 else None,
        "test_cumulative_at_1620s_kg": float(hydro["sed_cumulative_export_kg"][t <= 1620][-1]) if n >= 1620 else None,
        "test_pickup_total_kg": summary["sediment"]["totals"]["actual_pickup"],
        "test_deposition_total_kg": summary["sediment"]["totals"]["deposition_actual"],
        "accounting": ("reference rates are instantaneous outlet sediment fluxes (kg/s, four digits) summed with "
                       "dt = 1 s; SYRUP class exports are the conservative MAPLE export ledger; equation-level "
                       "agreement (docs/phase5/physics.md) does not imply whole-storm agreement, which is measured "
                       "here: no pass mark"),
    }
    # legacy accumulated detachment / net erosion; depth and velocity are NOT comparable (see below)
    pickup = final["cumulative_pickup_kg"].sum(axis=-1)
    deposition = final["cumulative_deposition_kg"].sum(axis=-1)
    bed_change = final["bed_change_kg"].sum(axis=-1)
    spatial = {
        "peak_outlet_snapshot_depth_velocity": {
            "comparable": False,
            "reference_definition": PEAK_SNAPSHOT_DEFINITION,
            "reference_snapshot_time_s": (f"within the rounded peak-outlet plateau {list(water['reference_peak_plateau_s'])}"
                                          " (the exact step is not recoverable from the rounded stock output)"),
            "test_definition": "SYRUP peak_depth_m / peak_velocity_m_s are per-cell maxima over every accepted step",
            "matching_quantity": "a synchronous SYRUP snapshot of routed depth and velocity at its own peak-outlet "
                                 "step (read-only observer sidecar); compared separately, not here",
            "reference_snapshot_stats": {"depth_mm": field_stats(m["maps"]["depth"]),
                                         "velocity_mm_s": field_stats(m["maps"]["veloc"])},
            "test_storm_maxima_stats": {"depth_mm": field_stats(final["peak_depth_m"] * 1.0e3),
                                        "velocity_mm_s": field_stats(final["peak_velocity_m_s"] * 1.0e3)},
            "note": "the two statistics blocks describe different quantities and carry no error metric",
        },
        "detachment_kg_per_cell": {
            **field_metrics(m["maps"]["detac"], pickup),
            "meaning": "reference: accumulated legacy detachment DEMAND (detach_tot * dx^2 rho dt); test: actual "
                       "MAPLE removal (supply-limited); not the same quantity"},
        "net_erosion_kg_per_cell_water_exchange": {
            **field_metrics(m["maps"]["neter"], pickup - deposition),
            "meaning": "reference: (detach_tot - depos_tot) legacy rates; test: actual MAPLE pickup minus deposition "
                       "(water class exchange only)"},
        "net_erosion_kg_per_cell_actual_bed": {
            **field_metrics(m["maps"]["neter"], -bed_change),
            "meaning": "test: minus the actual MAPLE bed inventory change (voxel + active), which the frozen "
                       "elevation does not reflect"},
        "cumulative_cell_discharge_m3": {"reference_sum": float(m["maps"]["dschg"].sum()),
                                         "test": None,
                                         "meaning": "SYRUP records no per-cell cumulative throughflow; not compared"},
        "orientation": "MAHLERAN ASCII cropped of its ring and row-reversed to MAPLE south-first order",
    }
    report = {
        "schema": SCHEMA,
        "syrup_dir": str(syrup_dir), "mahleran_dir": str(mahleran_dir),
        "syrup_summary_sha256": sha256_file(syrup_dir / "benchmark_summary.json"),
        "syrup_outputs_sha256": summary["outputs"], "mahleran_outputs_sha256": m["hashes"],
        "syrup_source_sha256": summary["provenance"]["maple_syrup"]["package_source_digest"]["digest_sha256"],
        "maple_source_sha256": summary["provenance"]["maple"]["package_source_digest"]["digest_sha256"],
        "mahleran_executable_sha256": m["execution"]["executable_sha256"],
        "matched_configuration": {
            "forcing_identical_per_second": forcing_equal,
            "conductivity_mm_s": ksat, "reference_conductivity_uniform": conductivity_equal,
            "reference_distribution": m["distribution"],
            "syrup_max_dt_s": summary["time"]["max_dt_s"], "syrup_implementation": summary["provenance"]["implementation"],
            "window_s": [0.0, float(t[-1])], "rows": n,
            "frozen_geometry_checks": summary["mode"]["checks"]["checks"],
        },
        "water": water, "targets": evaluate_targets(water), "sediment": sediment, "spatial": spatial,
        "spatial_panels": [label for _, _, label in SPATIAL_PANELS],
        "syrup_budgets": {"water": summary["budget"], "sediment_closure": summary["sediment"]["closure"]},
        "limitations": [
            "MAHLERAN files print four significant digits; no bitwise or conservation comparison is possible from them",
            "endpoint-sum integrals and the SYRUP conservative export ledger are different quantities",
            ("legacy detachment/net erosion maps are demand-based rates; MAPLE pickup and bed change are actual "
             "supply-limited inventories"),
            ("depth001/veloc001 are synchronous peak-outlet snapshots (output_hydro_data_xml.f90 489-505), not "
             "per-cell maxima; they are not compared with SYRUP's per-cell peak depth/velocity"),
            "reference splash disabled by patch; SYRUP has no splash; other legacy bookkeeping unchanged",
            "SYRUP hydraulic geometry frozen for the comparison; its normal mode evolves terrain",
        ],
    }
    if not forcing_equal:
        report["limitations"].append("FORCING MISMATCH: SYRUP applied rainfall differs from this reference log")
    arrays = {"t_s": t, "mahleran_q_m3_s": m["q_m3_s"], "syrup_q_m3_s": q_s,
              "syrup_export_ledger_m3": hydro["cumulative_export_m3"],
              "mahleran_sediment_kg_s": m["sediment_kg_s"], "syrup_export_rate_kg_s": rate_s,
              "mahleran_class_kg_s": m["class_kg_s"],
              "syrup_cumulative_export_kg": hydro["sed_cumulative_export_kg"],
              # retained as separately labelled diagnostics; different definitions, never compared or co-plotted
              "mahleran_peak_outlet_snapshot_depth_mm": m["maps"]["depth"],
              "mahleran_peak_outlet_snapshot_velocity_mm_s": m["maps"]["veloc"],
              "syrup_storm_max_depth_mm": final["peak_depth_m"] * 1.0e3,
              "syrup_storm_max_velocity_mm_s": final["peak_velocity_m_s"] * 1.0e3,
              "mahleran_detachment_kg": m["maps"]["detac"], "syrup_pickup_kg": pickup,
              "mahleran_net_erosion_kg": m["maps"]["neter"], "syrup_water_net_erosion_kg": pickup - deposition,
              "syrup_actual_bed_change_kg": bed_change}
    return report, arrays


def plot(arrays: dict, output: Path) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    t = arrays["t_s"]
    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    axes[0].plot(t, arrays["mahleran_q_m3_s"] * 1e3, label="MAHLERAN (deterministic K, no splash)", linewidth=1.2)
    axes[0].plot(t, arrays["syrup_q_m3_s"] * 1e3, label="SYRUP frozen geometry", linewidth=1.0)
    axes[0].set_ylabel("Outlet discharge (L/s)")
    axes[1].plot(t, arrays["mahleran_sediment_kg_s"] * 1e3, linewidth=1.2)
    axes[1].plot(t, arrays["syrup_export_rate_kg_s"] * 1e3, linewidth=1.0)
    axes[1].set_ylabel("Sediment export (g/s)")
    axes[2].plot(t, np.cumsum(arrays["mahleran_sediment_kg_s"]) * 1e3, linewidth=1.2)
    axes[2].plot(t, arrays["syrup_cumulative_export_kg"] * 1e3, linewidth=1.0)
    axes[2].set_ylabel("Cumulative export (g)")
    axes[2].set_xlabel("Time (s)")
    for ax in axes:
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.suptitle("Plot 1 matched benchmark: outlet water and sediment")
    fig.tight_layout()
    fig.savefig(output / "series.png", dpi=150)
    plt.close(fig)
    # Only fields with a shared definition are drawn side by side (SPATIAL_PANELS). The legacy
    # peak-outlet depth/velocity snapshots are not maxima and get no panel.
    fig, axes = plt.subplots(2, len(SPATIAL_PANELS), figsize=(5 * len(SPATIAL_PANELS), 9), squeeze=False)
    for col, (a, b, label) in enumerate(SPATIAL_PANELS):
        signed = bool(np.any(arrays[a] < 0.0) or np.any(arrays[b] < 0.0))
        vmax = max(float(np.max(np.abs(arrays[a]))), float(np.max(np.abs(arrays[b]))), 1e-300)
        for row, name in enumerate((a, b)):
            im = axes[row, col].imshow(arrays[name], origin="lower", vmin=-vmax if signed else 0.0, vmax=vmax,
                                       cmap="RdBu_r" if signed else "viridis")
            axes[row, col].set_title(f"{'MAHLERAN' if row == 0 else 'SYRUP'} {label}", fontsize=9)
            fig.colorbar(im, ax=axes[row, col], fraction=0.05)
    fig.tight_layout()
    fig.savefig(output / "spatial.png", dpi=150)
    plt.close(fig)
    return ["series.png", "spatial.png"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--syrup", type=Path, required=True)
    parser.add_argument("--mahleran", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="NEW directory for JSON, arrays and plots.")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"output {args.output} exists; use a new directory")
    report, arrays = compare(args.syrup.resolve(), args.mahleran.resolve())
    args.output.mkdir(parents=True)
    np.savez(args.output / "series.npz", **arrays)
    report["plots"] = plot(arrays, args.output)
    (args.output / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: report[k] for k in ("matched_configuration", "targets")}, indent=2))
    print(json.dumps({k: report["water"][k] for k in ("integrated_relative_difference", "peak_relative_difference",
                                                        "reference_peak_plateau_s", "test_peak_time_s", "rmse_m3_s")},
                     indent=2))
    print(json.dumps({k: report["sediment"][k] for k in ("total_ratio_test_over_reference",
                                                           "class_ratio_test_over_reference")}, indent=2))


if __name__ == "__main__":
    main()
