"""Phase 7i: compare the SYRUP prepared hydrology with the heterogeneous whole-MAHLERAN reference. Reads saved outputs
only; executes neither model.

    python benchmarks/phase7i/compare_hydrology.py --capture-run RUN --syrup dt1=DIR [--syrup dt0.5=DIR ...] \\
        [--controlled dt1=DIR ...] [--earlier-run outputs/phase7/mahleran_fixed_no_splash] --output NEW_DIR

PREDECLARED TARGETS (fixed before any result; a miss is reported, never relaxed, re-defined or tuned away):
  integrated endpoint outlet runoff   |relative difference| <= 1 %      (sum Q dt, the stock-output quantity)
  peak outlet discharge               |relative difference| <= 1 %
  peak time                           inside the reference peak plateau +/- 10 s
      (for the rounded stock series the plateau is the run of equal rounded maxima; for the full-precision series it
       is the single exact peak step)
  spatial fields (each model at ITS OWN peak-outlet step for depth/velocity; final surface depth; final soil water;
  cumulative drainage)                relative L2 <= 1 %  (aim; reported with max abs error, RMSE, correlation)
Hydrograph NSE/RMSE, budgets and the conservative face export are measurements without a pass mark except that the
SYRUP water budget must close inside its MAPLE-derived bound and the prepared/reference parity inside the phase 7h
water bound (rtol 2e-12, atol 1e-14). Budget tolerances of the original program are not altered: the legacy residual
is reported and decomposed (stale inflow, Crank-Nicolson closure) where the controlled driver was run.

The comparison uses the application's FULL-PRECISION capture. The stock four-digit outputs are compared only for
the explicit consistency checks (rounded capture == stock file, derivative outputs == earlier heterogeneous run).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "phase7"))
import capture_data as cd
import compare_matched as cm
import run_controlled_fortran as rcf

SCHEMA = "maple-syrup-phase7i-hydrology-comparison/1"
TARGETS = {**cm.TARGETS, "spatial_relative_l2": 0.01}
NUMERIC_OUTPUTS = ("SedChange.dat", "aspct001.asc", "depth001.asc", "detac001.asc", "dschg001.asc", "hydro001.dat",
                   "ksat_001.asc", "neter001.asc", "pave.asc", "seddisch001.dat", "sedtr001.asc", "sedtr001.dat",
                   "theta001.asc", "veloc001.asc")


# --- consistency of the derivative with the earlier heterogeneous run --------------------------------------------
def compare_original_outputs(run: Path, earlier: Path) -> dict:
    """Byte identity of the 14 numerical stock outputs of the capture derivative with the earlier heterogeneous
    run. Identity of bytes implies the sampled realization (and everything else the program prints) is unchanged.
    No claim is made from rounded values alone: if any differs the derivative's OWN exact field is the reference."""
    same, different, missing = [], [], []
    for name in NUMERIC_OUTPUTS:
        a, b = run / "Output" / name, earlier / "Output" / name
        if not a.is_file() or not b.is_file():
            missing.append(name)
        elif cd.sha256_file(a) == cd.sha256_file(b):
            same.append(name)
        else:
            different.append(name)
    return {"identical": same, "different": different, "missing": missing,
            "all_numeric_outputs_identical": bool(not different and not missing),
            "interpretation": ("byte-identical rounded outputs: the derivative reproduces the earlier run; the captured "
                               "field is that run's realization (bitwise equivalence of the unprinted state is not "
                               "inferred from rounded files, only supported by it)") if not different and not missing else
                              ("OUTPUTS DIFFER from the earlier run: the sampled realization or run changed; the "
                               "derivative's own captured field is used consistently and the earlier outputs are not "
                               "used as reference")}


def rounded_map_check(capture: np.ndarray, printed: np.ndarray) -> dict:
    """Does the full-precision capture round to a printed 4-significant-digit map? Relative difference on the
    cells printed non-zero must be within half a unit of the last printed digit (worst case mantissa 0.1), and a
    printed zero must not hide a materially non-zero capture (absolute floor 5e-5 x the map maximum)."""
    nonzero = printed != 0.0
    rel = np.abs(capture - printed)[nonzero] / np.abs(printed[nonzero]) if nonzero.any() else np.zeros(1)
    zero_hidden = int(np.sum(~nonzero & (np.abs(capture) > 5.0e-5 * max(float(np.abs(printed).max()), 1e-300))))
    return {"max_relative_difference": float(rel.max()), "bound": cd.ROUNDED_MAP_RELATIVE_BOUND,
            "printed_zero_cells_hiding_nonzero_capture": zero_hidden,
            "consistent": bool(rel.max() <= cd.ROUNDED_MAP_RELATIVE_BOUND and zero_hidden == 0)}


def stock_consistency(run: Path, static: cd.Capture, steps: dict, final: cd.Capture, nr: int, nc: int) -> dict:
    """The full-precision capture must round to the stock printed files of its own run."""
    out = run / "Output"
    dx = np.float32(static.scalars["dx_mm"])
    hydro = np.loadtxt(out / "hydro001.dat")
    q_f32 = steps["q_plot_single_mm2_s"].astype(np.float32) * dx  # the model's own `q_plot * dx` product
    printed = cm.round_sig(q_f32.astype(np.float64))
    mismatch = int(np.sum(printed != hydro[:, 2]))
    shape = (nr - 1, nc - 1)
    ksat = rounded_map_check(cd.interior_south_first(static.arrays["ksat"], nr, nc), cm.read_asc(out / "ksat_001.asc", shape))
    depth = rounded_map_check(cd.interior_south_first(final.arrays["model_dmax_mm"], nr, nc),
                              cm.read_asc(out / "depth001.asc", shape))
    velocity = rounded_map_check(cd.interior_south_first(final.arrays["model_vmax_mm_s"], nr, nc),
                                 cm.read_asc(out / "veloc001.asc", shape))
    return {
        "hydro001_outlet_rounded_mismatches": mismatch, "hydro001_rows": int(hydro.shape[0]),
        "ksat_map": ksat, "peak_depth_map": depth, "peak_velocity_map": velocity,
        "ksat_map_consistent": ksat["consistent"], "peak_maps_consistent": bool(depth["consistent"] and velocity["consistent"]),
        "model_dmax_minus_hook_copy_mm": final.scalars["max_abs_model_dmax_minus_capture_mm"],
        "model_vmax_minus_hook_copy_mm_s": final.scalars["max_abs_model_vmax_minus_capture_mm_s"],
        "hook_peak_copy_equals_model_maps": bool(final.scalars["max_abs_model_dmax_minus_capture_mm"] == 0.0
                                                 and final.scalars["max_abs_model_vmax_minus_capture_mm_s"] == 0.0),
    }


# --- metrics ----------------------------------------------------------------------------------------------------
def spatial_pair(legacy: np.ndarray, syrup: np.ndarray, target: float = TARGETS["spatial_relative_l2"]) -> dict:
    """`cm.field_metrics` plus the predeclared relative-L2 verdict (not evaluable when the reference norm is 0)."""
    metrics = cm.field_metrics(legacy, syrup)
    rel = metrics["relative_l2"]
    metrics["target_relative_l2"] = target
    if not np.any(np.asarray(legacy) != 0.0):  # zero reference: a relative error does not exist; report a zero match
        metrics.update(evaluable=False, reference_is_zero=True, zero_match=bool(not np.any(np.asarray(syrup) != 0.0)),
                       note="the legacy reference field is identically zero: no relative assessment; compare max_abs_difference")
        metrics["pass"] = None
        return metrics
    metrics["evaluable"] = rel is not None
    metrics["pass"] = bool(rel is not None and rel <= target)
    return metrics


def legacy_budget(static: cd.Capture, steps: dict, nr: int, nc: int) -> dict:
    """Water budget of the application from the per-step capture, in m3 (mm depth sums x cell area).
    `export_*` come both ways: the stock endpoint sum Q dt and the conservative Crank-Nicolson face export."""
    s = static.scalars
    c = s["dx_mm"] * s["dy_mm"] * cd.MM3_TO_M3  # m3 per mm of depth in one cell
    dt = s["dt_s"]
    active = static.arrays["rmask"][1:nr, 1:nc] >= 0.0
    soil0 = float(static.arrays["cum_inf"][1:nr, 1:nc][active].sum()) * c
    rain = float(np.sum(steps["sum_r2_active_mm_s"] * dt)) * c
    surface, soil, drain = (float(steps[k][-1]) * c for k in ("surface_sum_mm", "soil_sum_mm", "drain_sum_mm"))
    q = steps["q_outlet_double_mm2_s"] * s["dx_mm"] * cd.MM3_TO_M3
    endpoint = float(np.sum(q * dt))
    cn = float(np.sum(steps["cn_export_step_m3"]))
    stored = surface + soil + drain - soil0
    return {"units": "m3", "rain": rain, "soil_initial": soil0, "surface_final": surface, "soil_final": soil,
            "drainage": drain, "net_infiltration": soil - soil0 + drain, "export_endpoint_sum_q_dt": endpoint,
            "export_cn_face": cn, "residual_with_cn_export": stored + cn - rain,
            "residual_with_endpoint_export": stored + endpoint - rain,
            "note": "the original program keeps no water ledger: this is reconstructed from the hook's per-step sums"}


def relative(a: float, b: float):
    return None if b == 0.0 else (a - b) / b


def hydro_block(t: np.ndarray, reference: np.ndarray, test: np.ndarray) -> dict:
    block = cm.hydrograph_metrics(t, reference, test)
    block["targets"] = cm.evaluate_targets(block)
    # EXTRA diagnostic (not the predeclared criterion, which stays the rounded plateau +/- 10 s): offset of the test peak
    # from the reference's actual first-maximum time
    block["diagnostic_peak_time_offset_from_reference_argmax_s"] = abs(
        block["test_peak_time_s"] - float(t[int(np.argmax(reference))]))
    return block


def load_syrup(directory: Path) -> dict:
    summary = json.loads((directory / "hydrology_summary.json").read_text())
    if summary.get("schema") != "maple_syrup.phase7i.hydrology.v1":
        raise ValueError(f"{directory} is not a phase 7i SYRUP hydrology output")
    for name, digest in summary["outputs"].items():
        if cd.sha256_file(directory / name) != digest:
            raise ValueError(f"{directory / name} does not match its recorded hash")
    with np.load(directory / "history.npz") as h, np.load(directory / "fields.npz") as f:
        return {"summary": summary, "history": {k: h[k].copy() for k in h.files}, "fields": {k: f[k].copy() for k in f.files}}


def load_controlled(directory: Path, n_seconds: int) -> dict:
    record = json.loads((directory / "controlled_summary.json").read_text())
    if record.get("status") != "complete":
        raise ValueError(f"{directory}: controlled run is not complete")
    for name, digest in record["output_sha256"].items():
        if cd.sha256_file(directory / name) != digest:
            raise ValueError(f"{directory / name} does not match its recorded hash")
    m = round(1.0 / record["dt_s"])
    data = rcf.parse_history((directory / "hydrograph.dat").read_text(), n_seconds * m)
    return {"record": record, "data": data, "m": m}


def sensitivity_row(t, legacy_q, q_test, peak_t, export, label, native_integral) -> dict:
    """`one_second_sample_*` use the outlet discharge sampled at each whole second (what the 1 s legacy output can
    be compared with); `native_dt_endpoint_integral_m3` is the sum of Q dt over EVERY step of that run's own dt. They are
    different quantities for dt < 1 s and are never substituted for one another."""
    block = cm.hydrograph_metrics(t, legacy_q, q_test)
    legacy_integral = float(np.sum(legacy_q * np.diff(np.concatenate(([0.0], t)))))
    return {"label": label, "one_second_sample_endpoint_integral_m3": block["test_endpoint_integral_m3"],
            "native_dt_endpoint_integral_m3": native_integral,
            "native_dt_integral_relative_difference_vs_legacy_endpoint": relative(native_integral, legacy_integral),
            "integrated_relative_difference": block["integrated_relative_difference"],
            "peak_m3_s": block["test_peak_m3_s"], "peak_relative_difference": block["peak_relative_difference"],
            "peak_time_s": peak_t, "conservative_export_m3": export, "rmse_m3_s": block["rmse_m3_s"],
            "nash_sutcliffe": block["nash_sutcliffe"]}


def compare(run: Path, syrup_dirs: dict[str, Path], controlled_dirs: dict[str, Path], earlier: Path | None) -> tuple[dict, dict]:
    record = cd.verify_capture_run(run)
    static = cd.load_capture(run / "Output" / cd.STATIC_NAME, "static")
    steps = cd.load_steps(run / "Output" / cd.STEPS_NAME)
    final = cd.load_capture(run / "Output" / cd.FINAL_NAME, "final")
    nr, nc = int(static.scalars["nr"]), int(static.scalars["nc"])
    n = int(steps["iter"].size)
    t = steps["t_s"]
    dx = static.scalars["dx_mm"]
    q_full = steps["q_outlet_double_mm2_s"] * dx * cd.MM3_TO_M3
    q_single = steps["q_plot_single_mm2_s"] * dx * cd.MM3_TO_M3
    stock = np.loadtxt(run / "Output" / "hydro001.dat")[:, 2] * 1.0e-9
    if stock.shape != (n,):
        raise ValueError("stock hydrograph does not match the capture's step count")
    iter_single, iter_double = int(final.scalars["peak_iter_single"]), int(final.scalars["peak_iter_double"])
    exact_single_t, exact_double_t = float(t[iter_single - 1]), float(t[iter_double - 1])
    legacy = {
        "peak_time_single_precision_logic_s": exact_single_t, "peak_time_double_precision_s": exact_double_t,
        "peak_discharge_single_m3_s": float(q_single.max()), "peak_discharge_double_m3_s": float(q_full.max()),
        "argmax_single_equals_model_peak": bool(int(np.argmax(q_single)) + 1 == iter_single),
        "argmax_double_equals_hook_peak": bool(int(np.argmax(q_full)) + 1 == iter_double),
        "single_vs_double_max_relative_difference": float(np.max(np.abs(q_single - q_full) / np.maximum(q_full, 1e-300)
                                                              * (q_full > 0))),
        "endpoint_integral_double_m3": float(q_full.sum()), "endpoint_integral_single_m3": float(q_single.sum()),
        "endpoint_integral_stock_rounded_m3": float(stock.sum()),
    }
    legacy_bud = legacy_budget(static, steps, nr, nc)
    d_leg = {k: cd.interior_south_first(final.arrays[k], nr, nc) * cd.MM_TO_M for k in (
        "d_peak_single_mm", "d_peak_double_mm", "d_final_mm", "cum_inf_final_mm", "cum_drain_final_mm")}
    v_leg = {k: cd.interior_south_first(final.arrays[k], nr, nc) * cd.MM_TO_M for k in ("v_peak_single_mm_s", "v_peak_double_mm_s")}
    report = {
        "schema": SCHEMA, "targets_predeclared": TARGETS,
        "capture_run": str(run), "capture_executable_sha256": record["executable_sha256"],
        "capture_files_sha256": {name: record["outputs"][name]["sha256"] for name in cd.CAPTURE_FILES},
        "earlier_run_identity": None if earlier is None else compare_original_outputs(run, earlier),
        "stock_consistency": stock_consistency(run, static, steps, final, nr, nc),
        "comparison_sources_sha256": {name: cd.sha256_file(Path(__file__).resolve().parent / name) for name in (
            "compare_hydrology.py", "capture_data.py", "run_controlled_fortran.py")},
        "legacy": legacy, "legacy_budget": legacy_bud,
        "legacy_budget_reading": ("the conservative CN face export closes the legacy budget to the stale-inflow gain and "
                                  "the Crank-Nicolson closure error; endpoint sums are the stock quantity"),
        "syrup": {}, "sensitivity": [], "controlled": {},
    }
    series = {"t_s": t, "mahleran_q_full_double_m3_s": q_full, "mahleran_q_single_m3_s": q_single, "mahleran_q_stock_rounded_m3_s": stock}
    sensitivity = [{"label": "MAHLERAN application dt=1 (full precision)",
                    "one_second_sample_endpoint_integral_m3": float(q_full.sum()),
                    "native_dt_endpoint_integral_m3": float(q_full.sum()),
                    "peak_m3_s": float(q_full.max()), "peak_time_s": exact_double_t, "conservative_export_m3": legacy_bud["export_cn_face"]}]
    for label, directory in syrup_dirs.items():
        s = load_syrup(directory)
        summary, h, f = s["summary"], s["history"], s["fields"]
        m = int(summary["substeps"])
        if int(summary["n_seconds"]) != n:
            raise ValueError(f"{label}: SYRUP run has {summary['n_seconds']} seconds, the capture {n}")
        for key, want in (("capture_executable_sha256", record["executable_sha256"]),):
            if summary["inputs"][key] != want:
                raise ValueError(f"{label}: SYRUP run was built from a different capture ({key})")
        if summary["inputs"]["capture_files_sha256"] != {name: record["outputs"][name]["sha256"] for name in cd.CAPTURE_FILES}:
            raise ValueError(f"{label}: SYRUP run used different capture files")
        q_s = h["outlet_m3_s"][m - 1::m]
        peak = summary["peak"]
        block_stock = hydro_block(t, stock, q_s)
        block_full = hydro_block(t, q_full, q_s)
        block_single = hydro_block(t, q_single, q_s)
        export = float(h["export_m3"].sum())
        native_integral = float(h["outlet_m3_s"].sum()) / m  # sum Q dt over every step of this run's own dt
        entry = {
            "directory": str(directory), "substeps": m, "implementation": summary["implementation"],
            "own_peak_time_s": peak["own_peak_time_s"], "own_peak_outlet_m3_s": peak["own_outlet_peak_m3_s"],
            "own_peak_time_offset_from_legacy_exact_single_s": abs(peak["own_peak_time_s"] - exact_single_t),
            "own_peak_time_offset_from_legacy_exact_double_s": abs(peak["own_peak_time_s"] - exact_double_t),
            "hydrograph_vs_stock_rounded": block_stock, "hydrograph_vs_full_precision": block_full,
            "hydrograph_vs_single_precision": block_single,
            "conservative_face_export_m3": export,
            "one_second_sample_endpoint_integral_m3": block_full["test_endpoint_integral_m3"],
            "native_dt_endpoint_integral_m3": native_integral,
            "endpoint_vs_conservative_export_m3": native_integral - export,
            "budget": summary["budget"], "parity_prepared_vs_reference": summary["parity_prepared_vs_reference"],
            "conductivity": summary["conductivity"], "forcing": summary["forcing"],
        }
        syr_budget = summary["budget"]
        entry["syrup_vs_legacy_totals"] = {
            "rain_m3": {"syrup": syr_budget["rain"], "mahleran": legacy_bud["rain"], "relative": relative(syr_budget["rain"], legacy_bud["rain"])},
            "drainage_m3": {"syrup": syr_budget["drainage"], "mahleran": legacy_bud["drainage"], "relative": relative(syr_budget["drainage"], legacy_bud["drainage"])},
            "net_infiltration_m3": {"syrup": syr_budget["intake"] - syr_budget["return"], "mahleran": legacy_bud["net_infiltration"],
                                     "relative": relative(syr_budget["intake"] - syr_budget["return"], legacy_bud["net_infiltration"])},
            "surface_final_m3": {"syrup": syr_budget["surface_final"], "mahleran": legacy_bud["surface_final"], "relative": relative(syr_budget["surface_final"], legacy_bud["surface_final"])},
            "soil_final_m3": {"syrup": syr_budget["soil_final"], "mahleran": legacy_bud["soil_final"], "relative": relative(syr_budget["soil_final"], legacy_bud["soil_final"])},
            "conservative_export_m3": {"syrup": export, "mahleran_cn_face": legacy_bud["export_cn_face"], "relative": relative(export, legacy_bud["export_cn_face"])},
            "one_second_sample_endpoint_integral_m3": {
                "syrup": block_full["test_endpoint_integral_m3"], "mahleran": legacy_bud["export_endpoint_sum_q_dt"],
                "relative": relative(block_full["test_endpoint_integral_m3"], legacy_bud["export_endpoint_sum_q_dt"])},
            "native_dt_endpoint_integral_m3": {
                "syrup": native_integral, "mahleran": legacy_bud["export_endpoint_sum_q_dt"],
                "relative": relative(native_integral, legacy_bud["export_endpoint_sum_q_dt"])},
        }
        entry["spatial"] = {
            "target_definition": "each model at its OWN peak-outlet step (legacy: the model's strict-greater single-precision "
                                 "running maximum, replicated by the hook); depth m, velocity m/s",
            "own_peak_depth": spatial_pair(d_leg["d_peak_single_mm"], f["own_peak_depth_m"]),
            "own_peak_velocity": spatial_pair(v_leg["v_peak_single_mm_s"], f["own_peak_velocity_m_s"]),
            "own_peak_depth_vs_legacy_double_peak": spatial_pair(d_leg["d_peak_double_mm"], f["own_peak_depth_m"]),
            "own_peak_velocity_vs_legacy_double_peak": spatial_pair(v_leg["v_peak_double_mm_s"], f["own_peak_velocity_m_s"]),
            "final_surface_depth": spatial_pair(d_leg["d_final_mm"], f["final_depth_m"]),
            "final_soil_water": spatial_pair(d_leg["cum_inf_final_mm"], f["final_soil_water_m"]),
            "cumulative_drainage": spatial_pair(d_leg["cum_drain_final_mm"], f["cum_drainage_m"]),
        }
        for tag, key in (("single", iter_single), ("double", iter_double)):
            name = f"snapshot_s{key}_depth_m"
            if name in f:
                entry["spatial"][f"synchronous_depth_at_legacy_{tag}_peak_step_diagnostic"] = spatial_pair(
                    d_leg[f"d_peak_{tag}_mm"], f[name])
                entry["spatial"][f"synchronous_velocity_at_legacy_{tag}_peak_step_diagnostic"] = spatial_pair(
                    v_leg[f"v_peak_{tag}_mm_s"], f[f"snapshot_s{key}_velocity_m_s"])
        report["syrup"][label] = entry
        series[f"syrup_{label}_q_m3_s"] = q_s
        sensitivity.append({"label": f"SYRUP {summary['implementation']} dt={1.0 / m:g}", **sensitivity_row(
            t, q_full, q_s, peak["own_peak_time_s"], export, label, native_integral)})
    for label, directory in controlled_dirs.items():
        c = load_controlled(directory, n)
        m, data = c["m"], c["data"]
        q_c = data[m - 1::m, 1]
        record_c = c["record"]
        entry = {"directory": str(directory), "dt_s": record_c["dt_s"], "executable_sha256": record_c["executable_sha256"],
                 "driver_source_sha256": record_c["driver_source_sha256"],
                 "hydrograph_vs_full_precision": hydro_block(t, q_full, q_c),
                 "final_budget_residual_m3": record_c["final_budget_residual_m3"],
                 "final_stale_gain_m3": record_c["final_stale_gain_m3"],
                 "final_closure_sum_m3": record_c["final_closure_sum_m3"],
                 "residual_minus_stale_minus_closure_m3": record_c["residual_minus_stale_minus_closure_m3"],
                 "driver_export_m3": float(data[-1, 2]), "driver_bracket_end_count": float(data[-1, 11]),
                 "max_closure_m": float(data[:, 10].max())}
        if m == 1:
            entry["application_vs_driver_max_abs_outlet_m3_s"] = float(np.max(np.abs(q_c - q_full)))
            entry["application_vs_driver_max_relative_outlet"] = float(np.max(np.abs(q_c - q_full) / np.maximum(q_full, 1e-300) * (q_full > 0)))
            entry["application_cn_export_vs_driver_export_m3"] = float(data[-1, 2] - legacy_bud["export_cn_face"])
        report["controlled"][label] = entry
        series[f"controlled_{label}_q_m3_s"] = q_c
        sensitivity.append({"label": f"original-routine driver dt={record_c['dt_s']:g}", **sensitivity_row(
            t, q_full, q_c, float(data[np.argmax(data[:, 1]), 0]), float(data[-1, 2]), label,
            float(data[:, 1].sum()) / m)})
    report["sensitivity"] = sensitivity
    return report, series


def parse_labelled(values: list[str] | None) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for item in values or []:
        label, sep, path = item.partition("=")
        if not sep or not label or not path or label in out:
            raise SystemExit(f"expected unique LABEL=DIR, got {item!r}")
        out[label] = Path(path).resolve()
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--capture-run", type=Path, required=True)
    parser.add_argument("--syrup", action="append", help="LABEL=DIR of a run_syrup_hydrology output (first is primary)")
    parser.add_argument("--controlled", action="append", help="LABEL=DIR of a run_controlled_fortran output")
    parser.add_argument("--earlier-run", type=Path, default=Path("outputs/phase7/mahleran_fixed_no_splash"))
    parser.add_argument("--output", type=Path, required=True, help="NEW directory")
    args = parser.parse_args(argv)
    syrup, controlled = parse_labelled(args.syrup), parse_labelled(args.controlled)
    if not syrup:
        raise SystemExit("at least one --syrup run is required")
    run_record = cd.verify_capture_run(args.capture_run)
    out = cd.refuse_output(args.output, cd.protected_paths(
        run_record, ("capture run", args.capture_run), ("earlier run", args.earlier_run),
        *((f"syrup {k}", v) for k, v in syrup.items()), *((f"controlled {k}", v) for k, v in controlled.items())))
    report, series = compare(args.capture_run.resolve(), syrup, controlled,
                             args.earlier_run.resolve() if args.earlier_run.is_dir() else None)
    out.mkdir(parents=True)
    np.savez(out / "series.npz", **series)
    (out / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False, default=float) + "\n")
    primary = next(iter(report["syrup"].values()))
    print(json.dumps({"earlier_run_identity": report["earlier_run_identity"] and report["earlier_run_identity"]["all_numeric_outputs_identical"],
                      "targets_vs_stock_rounded": primary["hydrograph_vs_stock_rounded"]["targets"],
                      "targets_vs_full_precision": primary["hydrograph_vs_full_precision"]["targets"],
                      "spatial_pass": {k: v["pass"] for k, v in primary["spatial"].items() if isinstance(v, dict) and "pass" in v}},
                     indent=2, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
