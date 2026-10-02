"""Phase 7i: SYRUP hydrology-only replay of the heterogeneous Plot1 storm with the CAPTURED conductivity realization.

    python benchmarks/phase7i/run_syrup_hydrology.py --capture-run outputs/phase7i/mahleran_capture_run \\
        --output outputs/phase7i/syrup_hydrology_dt1 [--substeps 1|2|4] [--allow-maple-source-change]

Frozen elevation and routing, 5400 s, no splash, no evapotranspiration, no dry reset, NO sediment: the actual pinned
MAPLE case, import and bed-initialisation machinery is used only to build the SYRUP grid, routing graph and column
parameters (`verify_plot1_case`, `prepare_verified_sediment_case`); the MAPLE bed is never written (its digest is
compared before and after). The conductivity realization is the full-precision field of the application's own
setup (not a constant mean, not an RNG sample): it is validated, converted mm/s -> m/s and injected into
`column_parameters`; every other column/geometry quantity comes from the case and is checked explicitly against the
application's post-setup state BEFORE any step. Forcing is the exact per-step rate the application applied (read
before `infilt`, hence including its one-second switching lag and single-precision `rval`), not the 0.01 mm/h log.

The hydrology is `hydrology_numba.prepare_hydrology` / `prepared_coupled_step` (default) or the reference
`storm.coupled_step`; `--parity reference` (default for the prepared path) additionally advances the reference
path as its own trajectory and compares EVERY public field of each step within the phase 7h water bound
(rtol 2e-12, atol 1e-14). A step the existing strict roundoff guard (hpre vs the column depth) refuses is NOT
fixed, relaxed or skipped: the input is saved reproducibly, the run exits with status 3 and the report asks Codex.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import re
import resource
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import capture_data as cd

WATER_RTOL = 2.0e-12
WATER_ATOL = 1.0e-14
SUBSTEPS_ALLOWED = (1, 2, 4, 8)
ITERATION_RE = re.compile(r"Starting iteration\s+(\d+) rain intensity:\s+([\d.]+) time step:\s+([\d.]+)")
SCHEMA = "maple_syrup.phase7i.hydrology.v1"
HISTORY_COLUMNS = ("t_s", "rate_m_s", "outlet_m3_s", "export_m3", "rain_m3", "intake_m3", "return_m3", "drainage_m3",
                   "surface_m3", "soil_m3", "max_depth_m", "max_velocity_m_s", "budget_residual_m3",
                   "max_cell_balance_residual_m", "n_no_runon", "n_partial_runon", "n_complete_runon")


class HydrologyStepFailure(RuntimeError):
    """A hydrology step was refused. `record` is JSON data; `inputs` the arrays needed to reproduce it."""

    def __init__(self, message: str, record: dict, inputs: dict):
        super().__init__(message)
        self.record = record
        self.inputs = inputs


# --- public-output parity (prepared vs reference) -----------------------------------------------------------------
def _walk(ref, new, path: str, found: list) -> None:
    if dataclasses.is_dataclass(ref) and not isinstance(ref, type):
        if type(new) is not type(ref):
            found.append((path, float("inf")))
            return
        for f in dataclasses.fields(ref):
            _walk(getattr(ref, f.name), getattr(new, f.name), f"{path}.{f.name}", found)
        return
    if ref is None or new is None:
        found.append((path, 0.0 if ref is new else float("inf")))
        return
    if isinstance(ref, (bool, str, int, np.integer, np.bool_)):
        found.append((path, 0.0 if type(new) is type(ref) and new == ref else float("inf")))
        return
    a, b = np.asarray(ref), np.asarray(new)
    if a.shape != b.shape or a.dtype != b.dtype:
        found.append((path, float("inf")))
    elif a.dtype.kind != "f":
        found.append((path, 0.0 if np.array_equal(a, b) else float("inf")))
    elif not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        found.append((path, float("inf")))
    else:
        found.append((path, float(np.max(np.abs(b - a) / (WATER_ATOL + WATER_RTOL * np.abs(a))))) if a.size else (path, 0.0))


class ParityTracker:
    """Running maximum, per public field path, of |new - ref| / (atol + rtol |ref|); <= 1 passes the water bound.
    Structural differences (type, shape, dtype, ints, strings, non-finite) are +inf."""

    def __init__(self) -> None:
        self.max_excess: dict[str, float] = {}
        self.steps = 0

    def update(self, ref, new) -> None:
        found: list = []
        _walk(ref, new, "step", found)
        self.steps += 1
        for path, value in found:
            self.max_excess[path] = max(self.max_excess.get(path, 0.0), value)

    def summary(self) -> dict:
        """JSON-safe: a structural mismatch (+inf) is listed in `structural_mismatch_fields` and reported as null."""
        worst = max(self.max_excess.values(), default=0.0)
        def finite(v):
            return v if math.isfinite(v) else None

        return {"compared_steps": self.steps, "n_public_fields": len(self.max_excess), "rtol": WATER_RTOL,
                "atol": WATER_ATOL, "max_normalised_excess": finite(worst), "pass": bool(worst <= 1.0),
                "fields_over_bound": sorted(p for p, v in self.max_excess.items() if v > 1.0),
                "structural_mismatch_fields": sorted(p for p, v in self.max_excess.items() if not math.isfinite(v)),
                "max_normalised_excess_by_field": {p: finite(v) for p, v in sorted(self.max_excess.items())}}


# --- the replay ------------------------------------------------------------------------------------------------------
def _exc_record(exc: BaseException | None) -> dict | None:
    return None if exc is None else {"class": type(exc).__name__, "message": str(exc)}


def simulate(graph, column, field, soil0, rates_m_s, *, substeps: int = 1, implementation: str = "prepared",
             parity: bool = False, snapshot_seconds=()) -> dict:
    """Advance the frozen-geometry hydrology through `rates_m_s` (one constant rate per whole second, applied
    unchanged; each second is `substeps` equal steps). Pure with respect to its inputs; no sediment, no MAPLE bed.

    Returns history arrays (one row per accepted step), per-cell cumulative fields, the own-peak synchronous
    snapshot (strict `>` first occurrence, like the legacy), requested end-of-second snapshots, the final state, the
    water budget with the MAPLE-derived bound, and the parity summary. Raises `HydrologyStepFailure` (reproducible
    input attached) if any path refuses a step."""
    from maple_syrup.conservation import volume_roundoff_bound_m3
    from maple_syrup.hydrology_numba import prepare_hydrology, prepared_coupled_step
    from maple_syrup.infiltration import InfiltrationError
    from maple_syrup.routing import RoutingError
    from maple_syrup.storm import StormControl, StormError, coupled_step, initial_state

    if substeps not in SUBSTEPS_ALLOWED:
        raise ValueError(f"substeps must be one of {SUBSTEPS_ALLOWED}")
    rates = np.asarray(rates_m_s, dtype=np.float64)
    if rates.ndim != 1 or rates.size < 1 or not np.all(np.isfinite(rates)) or np.any(rates < 0.0):
        raise ValueError("rates must be a non-empty finite non-negative 1-D array")
    if implementation not in ("prepared", "reference"):
        raise ValueError("implementation must be 'prepared' or 'reference'")
    if parity and implementation != "prepared":
        raise ValueError("the parity comparison runs the reference beside the PREPARED primary path")
    shape = tuple(graph.shape)
    n_cells = shape[0] * shape[1]
    area = float(graph.dx_m) ** 2
    dt = 1.0 / substeps
    n_steps = rates.size * substeps
    control = StormControl(max_dt_s=dt, min_dt_s=dt, max_retries=1, implementation="numba").validated()
    ctx = prepare_hydrology(graph, column) if implementation == "prepared" else None

    def primary(rate, state):
        return prepared_coupled_step(ctx, rate, state, dt, control) if ctx is not None else \
            coupled_step(graph, column, rate, state, dt, control)

    zeros = np.zeros(shape, dtype=np.float64)
    state = initial_state(graph, zeros, soil0)
    ref_state = initial_state(graph, zeros, soil0) if parity else None
    tracker = ParityTracker() if parity else None
    rate_field = np.empty(shape, dtype=np.float64)
    history = {name: np.zeros(n_steps) for name in HISTORY_COLUMNS}
    cum_rain, cum_intake, cum_return, cum_drain = (np.zeros(shape) for _ in range(4))
    wanted = {int(s) for s in snapshot_seconds}
    snapshots: dict[int, dict] = {}
    best_q, own_peak = -1.0, None
    soil0_m3 = float(np.sum(soil0)) * area
    index = 0
    for second, rate in enumerate(rates.tolist()):
        field.apply(rate, out=rate_field)
        for sub in range(substeps):
            t_end = second + (sub + 1) * dt
            p_exc = r_exc = None
            step = ref_step = None
            try:
                step = primary(rate_field, state)
            except (RoutingError, InfiltrationError, StormError) as exc:
                p_exc = exc
            if parity:
                try:
                    ref_step = coupled_step(graph, column, rate_field, ref_state, dt, control)
                except (RoutingError, InfiltrationError, StormError) as exc:
                    r_exc = exc
            if p_exc is not None or r_exc is not None:
                agree = (p_exc is not None and r_exc is not None and type(p_exc) is type(r_exc)
                         and str(p_exc) == str(r_exc))
                record = {"step_index": index, "t_start_s": t_end - dt, "dt_s": dt, "second": second,
                          "rate_m_s": rate, "primary": implementation, "primary_exception": _exc_record(p_exc),
                          "reference_exception": _exc_record(r_exc) if parity else "not run",
                          "primary_and_reference_agree": agree if parity else None,
                          "action": "NOT relaxed, skipped or fixed: reported to Codex (existing strict guard, "
                                    "docs/phase7h/performance.md 'Important existing roundoff follow-up')"}
                inputs = {"depth_m": np.array(state.depth_m), "soil_water_m": np.array(state.soil_water_m),
                          "discharge_m2_s": np.array(state.discharge_m2_s), "rain_rate_field_m_s": rate_field.copy(),
                          "ksat_m_per_s": np.array(column.ksat_m_per_s), "dt_s": np.float64(dt),
                          "t_start_s": np.float64(t_end - dt)}
                raise HydrologyStepFailure(f"step {index} refused ({record['primary_exception']}, "
                                           f"{record['reference_exception']})", record, inputs)
            if parity:
                tracker.update(ref_step, step)
                ref_state = ref_step.state
            col, route = step.column, step.route
            q_out = float(route.outlet_discharge_m3_s)
            rows = (t_end, rate, q_out, float(route.export_m3), float(col.rain_m.sum()) * area,
                    float(col.intake_m.sum()) * area, float(col.saturation_return_m.sum()) * area,
                    float(col.drainage_m.sum()) * area, float(np.sum(route.depth_m)) * area,
                    float(np.sum(col.soil_water_m)) * area, float(np.max(route.depth_m)),
                    float(np.max(route.velocity_m_s)), float(route.budget_residual_m3),
                    float(route.max_cell_balance_residual_m), int(step.n_no_runon), int(step.n_partial_runon),
                    int(step.n_complete_runon))
            for name, value in zip(HISTORY_COLUMNS, rows, strict=True):
                history[name][index] = value
            cum_rain += col.rain_m
            cum_intake += col.intake_m
            cum_return += col.saturation_return_m
            cum_drain += col.drainage_m
            if q_out > best_q:  # strict, first occurrence: the legacy running maximum
                best_q = q_out
                own_peak = {"index": index, "t_s": t_end, "outlet_m3_s": q_out,
                            "depth_m": np.array(route.depth_m), "velocity_m_s": np.array(route.velocity_m_s)}
            if sub == substeps - 1 and (second + 1) in wanted:
                snapshots[second + 1] = {"depth_m": np.array(route.depth_m), "velocity_m_s": np.array(route.velocity_m_s)}
            state = step.state
            index += 1
    sums = {"rain": float(cum_rain.sum()) * area, "intake": float(cum_intake.sum()) * area,
            "return": float(cum_return.sum()) * area, "drainage": float(cum_drain.sum()) * area,
            "surface_final": float(np.sum(state.depth_m)) * area, "soil_final": float(np.sum(state.soil_water_m)) * area,
            "export": float(history["export_m3"].sum()), "soil_initial": soil0_m3, "surface_initial": 0.0}
    scale = max(sums["surface_final"], sums["soil_final"], sums["drainage"], sums["export"], sums["soil_initial"],
                sums["rain"])
    residual = (sums["surface_final"] + sums["soil_final"] + sums["drainage"] + sums["export"]
                - sums["surface_initial"] - sums["soil_initial"] - sums["rain"])
    bound = volume_roundoff_bound_m3(4 * n_cells * max(n_steps, 1) + 7, scale)
    budget = {"units": "m3", **sums, "water_residual_m3": residual, "bound_m3": bound,
              "bound_rule": "volume_roundoff_bound_m3(4 n_cells n_steps + 7, largest operand): the MAPLE summation "
                            "coefficient used by benchmark_experiment (src/maple_syrup/benchmark_experiment.py 633-635)",
              "closed": bool(abs(residual) <= bound)}
    return {"history": history, "substeps": substeps, "dt_s": dt, "n_steps": n_steps, "own_peak": own_peak,
            "snapshots": snapshots, "final_depth_m": np.array(state.depth_m), "final_soil_water_m": np.array(state.soil_water_m),
            "final_discharge_m2_s": np.array(state.discharge_m2_s), "cum_rain_m": cum_rain, "cum_intake_m": cum_intake,
            "cum_return_m": cum_return, "cum_drainage_m": cum_drain, "budget": budget,
            "parity": None if tracker is None else tracker.summary(), "implementation": implementation,
            "preparation": None if ctx is None else ctx.summary()}


def check_logged_forcing(rval_mm_s: np.ndarray, stdout_text: str) -> dict:
    """The application's own printed per-iteration rain (f7.2 mm/h) must be the rounding of the captured exact
    applied rate: confirms the capture is the forcing the run logged. Max |difference| <= 0.005 mm/h (+ float32)."""
    logged = np.array(ITERATION_RE.findall(stdout_text), dtype=np.float64)
    if logged.shape != (rval_mm_s.size, 3) or not np.array_equal(logged[:, 0], np.arange(1, rval_mm_s.size + 1)):
        raise cd.CaptureError("the application log does not carry one iteration line per captured step")
    diff = np.abs(rval_mm_s * 3600.0 - logged[:, 1])
    bound = 0.005 + 1.0e-6
    if not np.all(diff <= bound):
        raise cd.CaptureError(f"captured applied rain differs from the logged value by up to {diff.max()} mm/h")
    return {"n_steps": int(logged.shape[0]), "max_abs_difference_mm_h": float(diff.max()), "printing_half_ulp_mm_h": 0.005}


# --- command line ----------------------------------------------------------------------------------------------------
def load_capture_run(run_dir: Path) -> dict:
    record = cd.verify_capture_run(run_dir)
    out = run_dir / "Output"
    static = cd.load_capture(out / cd.STATIC_NAME, "static")
    steps = cd.load_steps(out / cd.STEPS_NAME)
    final = cd.load_capture(out / cd.FINAL_NAME, "final")
    if int(final.scalars["n_steps_written"]) != steps["iter"].size or int(static.scalars["nit"]) != steps["iter"].size:
        raise cd.CaptureError("step count of the static, steps and final captures disagree")
    return {"record": record, "static": static, "steps": steps, "final": final}


def _write_json(path: Path, payload: dict) -> None:
    from maple_syrup.case_import import _json_safe

    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--capture-run", type=Path, required=True, help="completed run of the capture derivative")
    parser.add_argument("--case", type=Path, default=Path("outputs/plot1"))
    parser.add_argument("--output", type=Path, required=True, help="NEW directory")
    parser.add_argument("--substeps", type=int, default=1, choices=SUBSTEPS_ALLOWED, help="equal steps per second")
    parser.add_argument("--hydrology-implementation", choices=("prepared", "reference"), default="prepared")
    parser.add_argument("--parity", choices=("reference", "none"), default=None,
                        help="default: reference beside the prepared path, none for the reference path")
    parser.add_argument("--allow-maple-source-change", action="store_true",
                        help="the selected optimized MAPLE package differs from the import's recorded source")
    args = parser.parse_args(argv)
    parity = (args.parity or ("reference" if args.hydrology_implementation == "prepared" else "none")) == "reference"
    if parity and args.hydrology_implementation != "prepared":
        parser.error("--parity reference requires --hydrology-implementation prepared")

    from maple_syrup.case_import import verify_plot1_case
    from maple_syrup.column_experiment import (
        _bed_digest,
        _source_digests,
        _syrup_provenance,
    )
    from maple_syrup.infiltration import column_parameters
    from maple_syrup.provenance import environment_record
    from maple_syrup.sediment_experiment import prepare_verified_sediment_case

    t_wall = time.perf_counter()
    capture = load_capture_run(args.capture_run.resolve())
    capture_before = cd.capture_digest(args.capture_run)
    script_hashes = {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                     for name in ("run_syrup_hydrology.py", "capture_data.py")}
    static, steps = capture["static"], capture["steps"]
    nr, nc = int(static.scalars["nr"]), int(static.scalars["nc"])
    verified = verify_plot1_case(args.case, allow_maple_source_change=args.allow_maple_source_change)
    protected = cd.protected_paths(
        capture["record"], ("MAPLE source", verified.maple_dependency.source_root),
        ("MAPLE package", verified.maple_dependency.package_dir),
        ("MAHLERAN", verified.report["recipe"]["mahleran_root"]), ("case", args.case),
        ("capture run", args.capture_run), ("SYRUP src", HERE.parents[1] / "src"),
        ("earlier phase 7 outputs", HERE.parents[1] / "outputs/phase7"))
    out = cd.refuse_output(args.output, protected)
    syrup_prov = _syrup_provenance()
    digests_before = _source_digests(Path(syrup_prov["package_dir"]), verified.maple_dependency.package_dir)
    n_seconds = int(steps["iter"].size)
    prepared = prepare_verified_sediment_case(verified, end_s=float(n_seconds))
    case, graph, host, field, soil0 = (prepared[k] for k in ("case", "graph", "host", "field", "soil0"))
    bed_before = _bed_digest(case)

    # --- validate the captured setup against the SYRUP case BEFORE anything runs --------------------------------
    ksat_mm_s = cd.validate_ksat_mm_s(static.arrays["ksat"], nr, nc)
    if ksat_mm_s.shape != host["ksat_m_per_s"].shape:
        raise cd.CaptureError(f"conductivity grid {ksat_mm_s.shape} != case grid {host['ksat_m_per_s'].shape}")
    host, soil0, soil_binding = cd.bind_legacy_soil_initialization(static, host)
    ref_arrays = cd.reference_arrays(graph, host, np.asarray(soil0))
    ref_arrays["legacy_full_rmask"] = verified.fields["legacy_full_rainfall_scaling"][::-1].copy()
    static_report = cd.check_static_consistency(static, ref_arrays)
    static_report["legacy_soil_binding"] = soil_binding
    if static.scalars["dt_s"] != 1.0 or not np.array_equal(steps["t_s"], np.arange(1.0, n_seconds + 1.0)):
        raise cd.CaptureError("the capture is not a 1-second-step run")
    log_check = check_logged_forcing(steps["rval_applied_mm_s"], (args.capture_run / "stdout.log").read_text(errors="replace"))
    injected = cd.inject_ksat(host, ksat_mm_s)
    column = column_parameters(model="pavement_hawkins", **{k: injected[k] for k in (
        "ksat_m_per_s", "suction_m", "drainage_parameter", "theta_sat", "soil_thickness_m", "pavement_cover_fraction")})
    rates = steps["rval_applied_mm_s"] * cd.MM_TO_M  # exact per-step applied rate, mm/s -> m/s
    legacy_peaks = {int(capture["final"].scalars["peak_iter_single"]), int(capture["final"].scalars["peak_iter_double"])}
    setup_s = time.perf_counter() - t_wall

    out.parent.mkdir(parents=True, exist_ok=True)
    stage = out.parent / f".{out.name}.partial-{os.getpid()}"
    stage.mkdir()
    try:
        t_loop = time.perf_counter()
        try:
            result = simulate(graph, column, field, np.asarray(soil0), rates, substeps=args.substeps,
                              implementation=args.hydrology_implementation, parity=parity, snapshot_seconds=legacy_peaks)
        except HydrologyStepFailure as failure:
            np.savez(stage / "failure_input.npz", **failure.inputs)
            payload = {"schema": SCHEMA, "status": "FAILED: step refused", **failure.record,
                       "input_files": {"failure_input.npz": hashlib.sha256((stage / "failure_input.npz").read_bytes()).hexdigest()},
                       "capture_run": str(args.capture_run.resolve()), "case": str(args.case)}
            _write_json(stage / "step_failure.json", payload)
            stage.rename(out)
            print(json.dumps(payload, indent=2), file=sys.stderr)
            return 3
        loop_s = time.perf_counter() - t_loop
        if _bed_digest(case) != bed_before:
            raise RuntimeError("the MAPLE bed digest changed during a hydrology-only run")
        capture_after = cd.capture_digest(args.capture_run)
        if capture_after != capture_before:
            raise RuntimeError("the capture run's files changed while the replay ran")
        if {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in script_hashes} != script_hashes:
            raise RuntimeError("a benchmark script changed while the replay ran")
        digests_after = _source_digests(Path(syrup_prov["package_dir"]), verified.maple_dependency.package_dir)
        if digests_after != digests_before:
            raise RuntimeError(f"source changed during the run: {digests_before} -> {digests_after}")
        h = result["history"]
        peak = result["own_peak"]
        np.savez(stage / "history.npz", **h)
        np.savez(stage / "forcing.npz", rval_applied_mm_s=steps["rval_applied_mm_s"], rate_m_s=rates)
        fields_out = {"ksat_m_per_s": injected["ksat_m_per_s"], "own_peak_depth_m": peak["depth_m"],
                      "own_peak_velocity_m_s": peak["velocity_m_s"], "final_depth_m": result["final_depth_m"],
                      "final_soil_water_m": result["final_soil_water_m"], "cum_rain_m": result["cum_rain_m"],
                      "cum_intake_m": result["cum_intake_m"], "cum_return_m": result["cum_return_m"],
                      "cum_drainage_m": result["cum_drainage_m"]}
        for sec, snap in result["snapshots"].items():
            fields_out[f"snapshot_s{sec}_depth_m"] = snap["depth_m"]
            fields_out[f"snapshot_s{sec}_velocity_m_s"] = snap["velocity_m_s"]
        np.savez(stage / "fields.npz", **fields_out)
        kernels = None
        if args.hydrology_implementation == "prepared":
            from maple_syrup.hydrology_numba import kernel_provenance

            kernels = kernel_provenance()
        n_per = args.substeps
        sampled_q = h["outlet_m3_s"][n_per - 1::n_per]
        summary = {
            "schema": SCHEMA, "status": "completed hydrology-only replay (frozen terrain/routing, no sediment, no ET, no reset)",
            "implementation": args.hydrology_implementation, "substeps": args.substeps, "dt_s": result["dt_s"],
            "n_seconds": n_seconds, "n_steps": result["n_steps"],
            "conductivity": {"source": "full-precision realization captured from the application's own setup",
                             "min_mm_s": float(ksat_mm_s.min()), "max_mm_s": float(ksat_mm_s.max()),
                             "mean_mm_s": float(ksat_mm_s.mean()), "std_mm_s": float(ksat_mm_s.std()),
                             "n_cells": int(ksat_mm_s.size), "sha256_of_m_per_s_bytes": hashlib.sha256(
                                 np.ascontiguousarray(injected["ksat_m_per_s"]).tobytes()).hexdigest(),
                             "nominal_xml_mean_mm_s": host["ksat_m_per_s"].mean() * 1.0e3,
                             "note": "the positive-truncated sample mean exceeds the nominal location parameter"},
            "forcing": {"source": "exact per-step applied rate (single-precision rval widened), read before infilt",
                        "total_depth_mm": float(np.sum(steps["rval_applied_mm_s"])), "distinct_rates": int(np.unique(steps["rval_applied_mm_s"]).size),
                        "log_consistency": log_check, "sha256": hashlib.sha256(rates.tobytes()).hexdigest()},
            "static_consistency": static_report,
            "peak": {"own_outlet_peak_m3_s": peak["outlet_m3_s"], "own_peak_time_s": peak["t_s"],
                     "own_peak_step_index": peak["index"], "sampled_second_peak_m3_s": float(sampled_q.max()),
                     "legacy_peak_iterations_snapshotted": sorted(legacy_peaks)},
            "budget": result["budget"], "parity_prepared_vs_reference": result["parity"],
            "hydrology_context": result["preparation"], "kernels": kernels,
            "maple_bed": {"digest_before": bed_before, "digest_after": _bed_digest(case), "written": False},
            "inputs": {"capture_run": str(args.capture_run.resolve()), "case": str(args.case),
                       "capture_files_sha256": {name: capture["record"]["outputs"][name]["sha256"] for name in cd.CAPTURE_FILES},
                       "capture_executable_sha256": capture["record"]["executable_sha256"]},
            "performance": {"setup_s": setup_s, "loop_wall_s": loop_s, "whole_process_wall_s": time.perf_counter() - t_wall,
                            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                            "note": "one run; first step includes JIT; the reference lockstep roughly doubles loop time"},
            "provenance": {"maple_syrup": syrup_prov, "maple": verified.maple_provenance,
                           "source_digests_before": digests_before, "source_digests_after": digests_after,
                           "environment": environment_record(), "script_sha256": script_hashes["run_syrup_hydrology.py"],
                           "capture_data_sha256": script_hashes["capture_data.py"],
                           "capture_digest_before": capture_before, "capture_digest_after": capture_after,
                           "capture_unchanged": True},
            "outputs": {},
        }
        for name in ("history.npz", "forcing.npz", "fields.npz"):
            summary["outputs"][name] = hashlib.sha256((stage / name).read_bytes()).hexdigest()
        _write_json(stage / "hydrology_summary.json", summary)
        stage.rename(out)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    print(json.dumps({"status": summary["status"], "peak": summary["peak"], "budget_closed": summary["budget"]["closed"],
                      "water_residual_m3": summary["budget"]["water_residual_m3"], "bound_m3": summary["budget"]["bound_m3"],
                      "parity": None if result["parity"] is None else {k: result["parity"][k] for k in (
                          "compared_steps", "max_normalised_excess", "pass")}}, indent=2))
    return 0 if (summary["budget"]["closed"] and (result["parity"] is None or result["parity"]["pass"])) else 4


if __name__ == "__main__":
    sys.exit(main())
