"""Phase 4c: Plot 1 coupled water-only storm and recession on the actual MAPLE case.

    python -m maple_syrup.storm_experiment --case-dir outputs/plot1 \\
        --output-dir outputs/plot1_storm_dt1 --max-dt-s 1 --implementation numba

Rainfall/infiltration (accepted Phase 3 columns) and method-5 routing
(accepted Phase 4b) are coupled by `storm.evolve` from t = 0 to `--end-s`
(default: the legacy `stormlength`, 5400 s; the Plot 1 record ends at
1620 s, so the run includes the dry recession). The bed is fixed, there is
no sediment, no evapotranspiration and no dry reset: all residual water is
retained, and the run ends at the configured time, which is NOT an event
completion even if the flow is tiny.

The case is re-verified first (`case_import.verify_plot1_case`); the
rainfall is parsed from its hash-checked staged copy; parameters come from
the verified legacy settings and sidecar arrays (`plot1_parameters`); the
graph is `plot1_routing_graph`. The MAPLE sediment-side state is digested
before and after and must be unchanged; the final depth is returned as a
MAPLE `WaterState` (the loaded case's mobile array, depth replaced) and
validated. Outputs (NEW directory, never inside the case, MAPLE, MAHLERAN
or recipe trees; written only after every balance and source-stability
check passes): `storm_summary.json`, `final_water.npz`, `hydrograph.csv`
and `hydrograph.npz`.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from maple_syrup.case_import import (
    Plot1ImportError,
    VerifiedPlot1Case,
    _json_safe,
    _refuse_output,
    _write_new_json,
    verify_plot1_case,
)
from maple_syrup.column_experiment import (
    _bed_digest,
    _Clock,
    _source_digests,
    _syrup_provenance,
    plot1_parameters,
)
from maple_syrup.dependency import MapleDependencyError
from maple_syrup.infiltration import (
    LOCAL_BALANCE_RTOL,
    InfiltrationError,
    column_parameters,
    initial_soil_water_m,
)
from maple_syrup.provenance import environment_record
from maple_syrup.rainfall import (
    RainfallError,
    parse_legacy_rainfall_file,
    rainfall_field,
)
from maple_syrup.routing import (
    BALANCE_RTOL,
    IMPLEMENTATIONS,
    RoutingError,
    RoutingGraphError,
    plot1_routing_graph,
)
from maple_syrup.storm import (
    HYDROGRAPH_COLUMNS,
    StormControl,
    StormError,
    evolve,
    initial_state,
)

__all__ = ["FINAL_NAME", "HYDROGRAPH_CSV", "HYDROGRAPH_NPZ", "SUMMARY_NAME", "StormRun", "main", "run_plot1_storm"]

SUMMARY_SCHEMA = "maple_syrup.plot1_storm.v1"
SUMMARY_NAME = "storm_summary.json"
FINAL_NAME = "final_water.npz"
HYDROGRAPH_CSV = "hydrograph.csv"
HYDROGRAPH_NPZ = "hydrograph.npz"
_EPS = float(np.finfo(np.float64).eps)
# Column and routing per-step tolerances combined, per step and per cell.
_STORM_RTOL = LOCAL_BALANCE_RTOL + BALANCE_RTOL


@dataclass(frozen=True)
class StormRun:
    summary: dict[str, Any]
    verified: VerifiedPlot1Case
    water: Any  # MAPLE WaterState: final depth, the loaded case's mobile array
    soil_water_m: np.ndarray
    hydrograph: np.ndarray  # (n_rows, len(HYDROGRAPH_COLUMNS)) host


def _positive(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not (math.isfinite(value) and value > 0.0):
        raise StormError(f"{name} must be finite and > 0, got {value!r}")
    return float(value)


def run_plot1_storm(
    case_dir: str | Path,
    output_dir: str | Path,
    *,
    max_dt_s: float = 1.0,
    end_s: float | None = None,
    implementation: str = "numba",
    backend: str = "numpy",
    report_every_s: float = 60.0,
    min_dt_s: float = 1.0 / 1024.0,
    max_retries: int = 10,
    max_steps: int = 10_000_000,
    max_report_rows: int = 100_000,
    mahleran_root: str | Path | None = None,
    expected_maple_root: str | Path | None = None,
    allow_maple_source_change: bool = False,
) -> StormRun:
    from maple.core.backend import (
        read_transfer_counters,
        resolve_backend,
        synchronize,
        to_device,
        to_host,
    )
    from maple.water import validate_water_state

    from maple_syrup import routing_numba

    max_dt = _positive(max_dt_s, "max_dt_s")
    cadence = _positive(report_every_s, "report_every_s")
    if end_s is not None:
        _positive(end_s, "end_s")
    # Strict guard validation on the ORIGINAL values: no int()/float() coercion
    # of bools or non-integers before the checks.
    control = StormControl(max_dt_s=max_dt_s, min_dt_s=min_dt_s, max_retries=max_retries, max_steps=max_steps,
                           implementation=implementation).validated()
    if isinstance(max_report_rows, bool) or not isinstance(max_report_rows, int) or max_report_rows < 1:
        raise StormError(f"max_report_rows must be a positive int, got {max_report_rows!r}")
    if implementation not in IMPLEMENTATIONS:
        raise StormError(f"implementation must be one of {IMPLEMENTATIONS}, got {implementation!r}")
    if implementation == "numba" and backend != "numpy":
        raise StormError("implementation 'numba' runs on the numpy backend only; use --implementation array "
                         "for other backends (no transfer, no fallback)")
    if implementation == "numba" and not routing_numba.numba_available():
        raise StormError("implementation 'numba' requested but Numba is not importable (optional extra "
                         "maple-syrup[numba]); there is no fallback -- pass --implementation array explicitly")
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise StormError(f"refusing to write into existing path {output_dir}")
    if output_dir.is_relative_to(Path(case_dir).resolve()):
        raise StormError("refusing to write inside the bound case directory")

    clock = _Clock()
    syrup_provenance = _syrup_provenance()
    verified = verify_plot1_case(case_dir, mahleran_root=mahleran_root, expected_maple_root=expected_maple_root,
                                 allow_maple_source_change=allow_maple_source_change)
    dependency = verified.maple_dependency
    recipe_record = verified.report["recipe"]
    _refuse_output(output_dir, {
        "MAPLE source": dependency.source_root,
        "MAPLE package": dependency.package_dir,
        "MAHLERAN": Path(mahleran_root or recipe_record["mahleran_root"]).resolve(),
        "recorded MAHLERAN": Path(recipe_record["mahleran_root"]).resolve(),
        "recipe": Path(recipe_record["recipe_path"]).resolve().parent,
    })
    digests_before = {
        "maple_syrup": syrup_provenance["package_source_digest"]["digest_sha256"],
        "maple": verified.maple_provenance["package_source_digest"]["digest_sha256"],
    }
    case = verified.case
    settings = verified.report["legacy_options"]["settings"]
    schedule = parse_legacy_rainfall_file(verified.rainfall_path)
    if schedule.provenance.sha256 != verified.report["rainfall"]["sha256"]:
        raise StormError("parsed rainfall bytes differ from the verified rainfall hash")
    end = float(settings["stormlength_s"]) if end_s is None else float(end_s)
    host, parameter_record = plot1_parameters(verified.report, verified.fields)
    g = case.config.geometry
    ny, nx, n_classes = g.ny, g.nx, len(case.config.grain_classes.classes)
    if host["theta_sat"].shape != (ny, nx) or g.dx_m != g.dy_m:
        raise StormError("sidecar fields do not match the MAPLE grid, or cells are not square")
    area = g.dx_m * g.dy_m
    bed_before = _bed_digest(case)

    resolved = resolve_backend(backend)
    xp = resolved.xp
    graph = plot1_routing_graph(verified.fields, verified.report, xp=xp)
    if graph.dx_m != g.dx_m:
        raise StormError("routing graph spacing differs from the MAPLE geometry")
    dev = {name: to_device(array, xp) for name, array in host.items()}
    params = column_parameters(
        model="pavement_hawkins",
        **{k: dev[k] for k in ("ksat_m_per_s", "suction_m", "drainage_parameter", "theta_sat",
                               "soil_thickness_m", "pavement_cover_fraction")},
    )
    field = rainfall_field(ny, nx, scale=dev["rainfall_scale"])
    depth0 = to_device(np.array(case.water.depth_m, dtype=np.float64, copy=True), xp)
    soil0 = initial_soil_water_m(params, dev["initial_theta"])
    state0 = initial_state(graph, depth0, soil0, t_s=0.0)
    setup = clock.lap()

    # --- coupled loop: arrays stay in xp; two validating flag reads per attempt ---------
    counters_before = read_transfer_counters()
    result = evolve(graph, params, field, schedule, state0, end, control,
                    report_every_s=cadence, max_report_rows=max_report_rows)
    synchronize(xp)
    loop = clock.lap()
    transfers = dataclasses.asdict(read_transfer_counters().delta(counters_before))

    # --- reporting boundary: one stacked read of every scalar ---------------------------
    final = result.state
    names = ["surface_initial", "soil_initial", "rain", "intake", "saturation_return", "drainage",
             "surface_final", "soil_final"]
    grids = [depth0, soil0, result.cumulative_rain_m, result.cumulative_intake_m,
             result.cumulative_saturation_return_m, result.cumulative_drainage_m, final.depth_m, final.soil_water_m]
    scalars = to_host(xp.stack([xp.asarray(a.sum(), dtype=np.float64) for a in grids] + [
        xp.asarray(v, dtype=np.float64) for v in (
            result.cumulative_export_m3, result.peak_depth_m.max(), result.peak_velocity_m_s.max(),
            final.depth_m.max(), (final.depth_m > 0.0).sum(), final.soil_water_m.min(), final.soil_water_m.max(),
            result.max_courant_old, result.max_courant_new, result.cell_steps_no_runon,
            result.cell_steps_partial_runon, result.cell_steps_complete_runon,
            result.peak_outlet_discharge_m3_s, result.time_of_peak_outlet_s,
        )
    ]))
    sums = dict(zip(names, scalars[: len(names)].tolist()))
    (export_m3, peak_depth, peak_velocity, final_max_depth, ponded_cells, soil_min, soil_max,
     cr_old, cr_new, cells_no, cells_partial, cells_complete, true_peak_q, true_peak_t) = scalars[len(names):].tolist()
    hydrograph = to_host(result.hydrograph).copy()
    final_depth, final_soil = to_host(final.depth_m).copy(), to_host(final.soil_water_m).copy()
    final_q = to_host(final.discharge_m2_s).copy()
    grids_out = {
        "depth_m": final_depth, "soil_water_m": final_soil, "discharge_m2_s": final_q,
        "velocity_m_s": to_host(result.last_velocity_m_s).copy(),
        "peak_depth_m": to_host(result.peak_depth_m).copy(),
        "peak_velocity_m_s": to_host(result.peak_velocity_m_s).copy(),
        "cumulative_rain_m": to_host(result.cumulative_rain_m).copy(),
        "cumulative_intake_m": to_host(result.cumulative_intake_m).copy(),
        "cumulative_return_m": to_host(result.cumulative_saturation_return_m).copy(),
        "cumulative_drainage_m": to_host(result.cumulative_drainage_m).copy(),
    }
    if not all(np.all(np.isfinite(v)) for v in grids_out.values()) or not np.all(np.isfinite(hydrograph)):
        raise StormError("non-finite final grids or hydrograph; no output written")
    if np.any(final_depth < 0.0) or np.any(final_soil < 0.0) or np.any(final_q < 0.0):
        raise StormError("negative final inventory; no output written")

    water = dataclasses.replace(case.water, depth_m=final_depth)
    validate_water_state(water, ny, nx, n_classes)
    bed_after = _bed_digest(case)
    if bed_after != bed_before:
        raise StormError("MAPLE sediment-side state changed during a water-only run")

    # --- budget (depth sums over cells; volumes = x area) ------------------------------------
    n_steps, n_cells = result.n_accepted_steps, ny * nx
    export_depth = export_m3 / area
    inventory = sums["surface_initial"] + sums["soil_initial"] + sums["rain"]
    magnitude = inventory + sums["intake"] + sums["drainage"] + sums["saturation_return"] + export_depth
    tolerance = _STORM_RTOL * (n_steps + n_cells) * magnitude
    water_residual = (sums["surface_final"] + sums["soil_final"] + sums["drainage"] + export_depth) - inventory
    surface_residual = sums["surface_final"] - (sums["surface_initial"] + sums["rain"] - sums["intake"]
                                                + sums["saturation_return"] - export_depth)
    soil_residual = sums["soil_final"] - (sums["soil_initial"] + sums["intake"] - sums["drainage"]
                                          - sums["saturation_return"])
    rain_expected = schedule.depth_m(0.0, end) * float(host["rainfall_scale"].sum())
    rain_tolerance = 4.0 * _EPS * (n_steps + n_cells) * max(rain_expected, sums["rain"])
    for label, residual, tol in (("water", water_residual, tolerance), ("surface", surface_residual, tolerance),
                                 ("soil", soil_residual, tolerance),
                                 ("rainfall integral", sums["rain"] - rain_expected, rain_tolerance)):
        if not abs(residual) <= tol:
            raise StormError(f"{label} budget residual {residual} m exceeds tolerance {tol} m; no output written")
    rows = {name: hydrograph[:, i] for i, name in enumerate(HYDROGRAPH_COLUMNS)}
    row_residual = (rows["surface_storage_m3"] + rows["soil_storage_m3"] + rows["cumulative_drainage_m3"]
                    + rows["cumulative_export_m3"]) - (
        (sums["surface_initial"] + sums["soil_initial"]) * area + rows["cumulative_rain_m3"])
    if not np.all(np.abs(row_residual) <= tolerance * area):
        raise StormError("a hydrograph row violates the water balance; no output written")

    digests_after = _source_digests(Path(syrup_provenance["package_dir"]), dependency.package_dir)
    changed = [name for name in digests_before if digests_after[name] != digests_before[name]]
    if changed:
        raise StormError(f"source changed during the run ({', '.join(changed)}); no output written")

    graph_summary = graph.summary()
    graph_summary.pop("level_widths", None)
    summary = {
        "schema": SUMMARY_SCHEMA,
        "status": (f"configured_end_reached at t = {end} s; NOT an event completion: residual surface water "
                   f"{sums['surface_final'] * area} m3 and outlet discharge {rows['outlet_discharge_m3_s'][-1]} m3/s "
                   "are retained; no dry reset, no flow-stop criterion"),
        "case": {
            "case_dir": str(verified.case_dir),
            "maple_case_identity_sha256": verified.binding["maple_case_identity_sha256"],
            "maple_artifact_sha256": verified.binding["maple_artifact_sha256"],
            "verification": verified.checks,
        },
        "sources": {
            "rainfall": {"path": str(verified.rainfall_path), "sha256": schedule.provenance.sha256,
                         "n_records": schedule.provenance.n_records, "start_clock": schedule.provenance.start_clock,
                         "convention": schedule.provenance.convention},
            "mahleran_xml_sha256": verified.report["mahleran"]["xml_sha256"],
            "mahleran_git": verified.report["mahleran"]["git"],
            "syrup_report_sha256": verified.binding["syrup_report_sha256"],
            "syrup_fields_sha256": verified.binding["syrup_fields_sha256"],
        },
        "provenance": {
            "maple_syrup": syrup_provenance,
            "maple": verified.maple_provenance,
            "maple_matches_import_binding": not verified.checks["maple_source_changed_since_import"],
            "source_stability": {"before": digests_before, "after": digests_after, "stable": True,
                                 "rule": "package source digests taken before verification and after the loop; "
                                         "any change aborts before outputs are written"},
            "environment": environment_record(),
            "numba": routing_numba.numba_versions(),
            "implementation": implementation,
        },
        "domain": {"ny": ny, "nx": nx, "dx_m": g.dx_m, "cell_area_m2": area, "area_m2": area * n_cells,
                   "active_cells": graph.n_active, "graph": graph_summary,
                   "boundaries": "legacy rmask < 0 ring cells export (south edge outlets); no other lateral exchange"},
        "time": {
            "start_s": 0.0, "end_s": end, "rainfall_end_s": schedule.end_s,
            "legacy_stormlength_s": float(settings["stormlength_s"]),
            "recession_included": end > schedule.end_s,
            "max_dt_s": max_dt, "min_dt_s": control.min_dt_s, "n_accepted_steps": n_steps,
            "n_rejected_attempts": result.n_rejected_attempts, "rejections_recorded": list(result.rejections),
            "dt_min_accepted_s": result.min_accepted_dt_s, "dt_max_accepted_s": result.max_accepted_dt_s,
            "n_boundaries": int(result.boundaries.size), "report_every_s": cadence,
            "policy": "boundaries = rainfall knots in (0, end) + reporting times + end (float-noise "
                      "coincidences merged onto the exact forcing edge); dt <= max_dt_s; a forced slice to a "
                      "boundary may be shorter than min_dt_s; Courant/negative-RHS rejection halves dt and "
                      "recomputes the same state, refused below the retry floor min_dt_s; no other exception "
                      "is retried; failed attempts accumulate nothing",
        },
        "parameters": parameter_record,
        "routing": {
            "method": "MAHLERAN method 5 (Crank-Nicolson, bisection on [0, R], coherent old inflow)",
            "implementation": implementation, "courant_max": control.courant_max,
            "bisection_iterations": control.bisection_iterations, "root_tolerance_m": control.root_tolerance_m,
            "friction_factor": graph_summary["friction_factor_range"],
            "max_courant_old": cr_old, "max_courant_new": cr_new,
            "max_cell_balance_residual_m": float(rows["max_routing_cell_balance_residual_m"].max()),
            "max_constitutive_residual_m": float(rows["max_constitutive_residual_m"].max()),
            "old_flux_rule": "q_prev if intake <= rain; 0 if intake >= h + rain; else k hpre^1.5, "
                             "hpre = max(h - max(intake - rain, 0), 0) (legacy post-infilt d(1)); "
                             "receiver old inflow = donor sum of the same q_old (coherent; no stale qin)",
            "cell_steps": {"no_runon": cells_no, "partial_runon": cells_partial, "complete_runon": cells_complete},
        },
        "budget": {
            "units": "m3",
            "surface_initial_m3": sums["surface_initial"] * area,
            "soil_initial_m3": sums["soil_initial"] * area,
            "rain_m3": sums["rain"] * area,
            "rain_expected_m3": rain_expected * area,
            "intake_m3": sums["intake"] * area,
            "saturation_return_m3": sums["saturation_return"] * area,
            "net_infiltration_m3": (sums["intake"] - sums["saturation_return"]) * area,
            "drainage_m3": sums["drainage"] * area,
            "export_m3": export_m3,
            "surface_final_m3": sums["surface_final"] * area,
            "soil_final_m3": sums["soil_final"] * area,
            "water_residual_m3": water_residual * area,
            "surface_residual_m3": surface_residual * area,
            "soil_residual_m3": soil_residual * area,
            "rainfall_integral_residual_m3": (sums["rain"] - rain_expected) * area,
            "tolerance_m3": tolerance * area,
            "rainfall_tolerance_m3": rain_tolerance * area,
            "tolerance_rule": "(16 + 32) eps (n_steps + n_cells) (initial + rain + intake + drainage + return "
                              "+ export) depth sums; per-step column and routing FP64 tolerances combined",
            "identity": "surface_final + soil_final + drainage + export = surface_initial + soil_initial + rain",
            "export_meaning": "time-integrated Crank-Nicolson face volume through outlet cells; the hydrograph's "
                              "outlet_discharge_m3_s is the instantaneous end-of-step value (legacy q_plot dx)",
            "runoff_coefficient_export_over_rain": export_m3 / (sums["rain"] * area) if sums["rain"] > 0 else None,
        },
        "final_state": {
            "ponded_cells": int(ponded_cells), "max_depth_m": final_max_depth,
            "mean_depth_m": sums["surface_final"] / n_cells,
            "peak_depth_m": peak_depth, "peak_velocity_m_s": peak_velocity,
            "final_outlet_discharge_m3_s": float(rows["outlet_discharge_m3_s"][-1]),
            "peak_outlet_discharge_m3_s": true_peak_q,
            "time_of_peak_outlet_discharge_s": true_peak_t,
            "peak_note": "true numerical peak of the instantaneous outlet discharge over every accepted step "
                         "(tracked as device scalars); the hydrograph rows are samples at the reporting cadence",
            "sampled_peak_outlet_discharge_m3_s": float(rows["outlet_discharge_m3_s"].max()),
            "sampled_time_of_peak_outlet_discharge_s": float(rows["t_s"][int(np.argmax(rows["outlet_discharge_m3_s"]))]),
            "soil_water_range_m": [soil_min, soil_max],
            "residual_water": "retained in MAPLE WaterState.depth_m; configured end is not event end",
        },
        "maple_state": {
            "sediment_digest_before": bed_before, "sediment_digest_after": bed_after, "unchanged": True,
            "covers": "voxel column, active layer, availability, water mobile mass, ledger, topography",
            "water_state": "depth_m replaced by the final surface depth; mobile array is the loaded one",
        },
        "backend": {
            "backend": resolved.backend.value, "device_id": resolved.device_id,
            "fingerprint": json.loads(json.dumps(resolved.fingerprint, default=str)),
            "loop_transfer_counters": transfers,
            "validation": "two batched flag reads per attempt (column, routing); hydrograph rows written into a "
                          "device buffer; grids copied to the host once at the end",
            "gpu": "not exercised unless backend is cupy on an available device; no GPU claim",
        },
        "timings": {
            "setup_and_verification": setup,
            "first_accepted_step": {"wall_s": result.first_step_wall_s, "cpu_s": result.first_step_cpu_s,
                                    "note": "includes lazy import/JIT compilation for the numba implementation"},
            "remaining_steps": {"wall_s": result.remaining_wall_s, "cpu_s": result.remaining_cpu_s,
                                "n_steps": max(n_steps - 1, 0)},
            "step_loop": loop,
            "note": "process CPU and wall seconds of one run; not a controlled benchmark",
        },
        "hydrograph": {"columns": list(HYDROGRAPH_COLUMNS), "n_rows": int(hydrograph.shape[0]),
                       "files": [HYDROGRAPH_CSV, HYDROGRAPH_NPZ],
                       "interval_columns_in_csv": "rain/intake/return/drainage/export differences between rows"},
        "assumptions": [
            "fixed bed and routing graph (no sediment exchange, no terrain refresh)",
            "constant friction factor 21.45 (legacy type 1); draining D4 terrain, 10 south outlets",
            "rain excess and run-on infiltration per the accepted Phase 3 column (deterministic Ksat mean)",
            "coherent old face flux (accepted conservation correction; legacy stale qin(1) not used)",
            "no evapotranspiration, dry reset, plants, splash or sediment",
        ],
        "limitations": [
            "not a MAHLERAN Fortran storm benchmark (the coupled original harness is a separate task)",
            "configured end is not event completion; residual water and discharge are reported, not resolved",
            "no GPU result unless backend is cupy on an available device; no performance claim",
            "no production restart; final grids written once",
        ],
    }

    output_dir.mkdir(parents=True)
    with (output_dir / FINAL_NAME).open("xb") as handle:
        np.savez(handle, **grids_out)
    interval = {name: np.diff(rows[name], prepend=0.0) for name in (
        "cumulative_rain_m3", "cumulative_intake_m3", "cumulative_saturation_return_m3",
        "cumulative_drainage_m3", "cumulative_export_m3")}
    with (output_dir / HYDROGRAPH_NPZ).open("xb") as handle:
        np.savez(handle, **rows, **{f"interval_{k[len('cumulative_'):]}": v for k, v in interval.items()},
                 row_water_residual_m3=row_residual)
    with (output_dir / HYDROGRAPH_CSV).open("x", newline="", encoding="ascii") as handle:
        writer = csv.writer(handle)
        header = list(HYDROGRAPH_COLUMNS) + [f"interval_{k[len('cumulative_'):]}" for k in interval] + [
            "row_water_residual_m3"]
        writer.writerow(header)
        for i in range(hydrograph.shape[0]):
            writer.writerow([repr(float(v)) for v in hydrograph[i]] + [repr(float(interval[k][i])) for k in interval]
                            + [repr(float(row_residual[i]))])
    summary["outputs"] = {name: hashlib.sha256((output_dir / name).read_bytes()).hexdigest()
                          for name in (FINAL_NAME, HYDROGRAPH_NPZ, HYDROGRAPH_CSV)}
    summary["timings"]["reporting"] = clock.lap()
    _write_new_json(output_dir / SUMMARY_NAME, summary)
    return StormRun(summary=summary, verified=verified, water=water, soil_water_m=final_soil, hydrograph=hydrograph)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m maple_syrup.storm_experiment",
        description="MAPLE-SYRUP Phase 4c: Plot 1 coupled water-only storm and recession (no sediment, no reset).",
    )
    parser.add_argument("--case-dir", required=True, help="Verified Phase 2 case, e.g. outputs/plot1")
    parser.add_argument("--output-dir", required=True, help="NEW directory for summary, grids and hydrograph.")
    parser.add_argument("--max-dt-s", type=float, default=1.0, help="Largest step (s); knots always split.")
    parser.add_argument("--end-s", type=float, default=None,
                        help="Simulated end (s); default the legacy stormlength (5400 s for Plot 1).")
    parser.add_argument("--implementation", default="numba", choices=IMPLEMENTATIONS,
                        help="Ordered-sweep implementation (default numba; missing Numba is an error, no fallback).")
    parser.add_argument("--backend", default="numpy", choices=("numpy", "cupy"))
    parser.add_argument("--report-every-s", type=float, default=60.0, help="Hydrograph row cadence (s).")
    parser.add_argument("--min-dt-s", type=float, default=1.0 / 1024.0)
    parser.add_argument("--max-retries", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=10_000_000)
    parser.add_argument("--mahleran-root", help="Override the recipe's MAHLERAN root (read-only).")
    parser.add_argument("--expected-maple-root", help="Refuse any other MAPLE source root.")
    parser.add_argument("--allow-maple-source-change", action="store_true",
                        help="Run even if MAPLE's source digest differs from the import's (recorded).")
    args = parser.parse_args(argv)
    try:
        run = run_plot1_storm(
            args.case_dir, args.output_dir, max_dt_s=args.max_dt_s, end_s=args.end_s,
            implementation=args.implementation, backend=args.backend, report_every_s=args.report_every_s,
            min_dt_s=args.min_dt_s, max_retries=args.max_retries, max_steps=args.max_steps,
            mahleran_root=args.mahleran_root, expected_maple_root=args.expected_maple_root,
            allow_maple_source_change=args.allow_maple_source_change,
        )
    except MapleDependencyError as exc:
        print(f"MAPLE dependency check failed: {exc}", file=sys.stderr)
        return 2
    except (Plot1ImportError, StormError, RoutingError, RoutingGraphError, InfiltrationError, RainfallError,
            ValueError) as exc:
        print(f"plot1 storm failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(_json_safe({k: run.summary[k] for k in ("status", "time", "budget", "final_state", "timings")}),
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
