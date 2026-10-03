"""EXPERIMENTAL water-only comparison runs of the two hydraulic alternatives on the actual imported Plot 1 case.

    python -m maple_syrup.experimental_experiment --case-dir outputs/plot1 --output-dir outputs/exp_explicit \\
        --solver explicit --backend numpy --max-dt-s 1 --end-s 5400 --allow-maple-source-change
    python -m maple_syrup.experimental_experiment --case-dir outputs/plot1 --output-dir outputs/exp_li_gpu \\
        --solver local_inertial --backend cupy --max-dt-s 1 --limiter off

`--solver explicit|local_inertial` selects the candidate (`experimental_hydrology` documents the equations, CFL/positivity
bounds, the Darcy normal-flow outlet boundary of the local-inertial run and the optional donor limiter); `--backend numpy` runs
the pure NumPy reference, `--backend cupy` the CUDA form (CuPy and a device required, never Numba, no fallback).

What is reused unchanged: the verified MAPLE case (`case_import.verify_plot1_case`, same source/bed protection and refusal of
output inside protected trees), the same column parameters, rainfall schedule/field, antecedent soil water, depth and graph as
`storm_experiment` (so the forcing is identical), `storm.plan_boundaries`, and the accepted column physics. The bed, voxel state
and sediment are never touched (their digest is compared before and after); there is no sediment, splash, evapotranspiration, dry
reset or plant growth, and the run ends at the configured time, which is NOT an event completion (residual water is retained and
reported). Fixed terrain: elevation and (for the explicit method) the D4 routing of the legacy graph are frozen.

Outputs (NEW directory only, written after every budget and source-stability check passes): `experiment_summary.json`,
`final_state.npz`, `hydrograph.npz`, `hydrograph.csv` and, when `--snapshot-times-s` is given, `snapshots.npz`
(synchronous depth/soil/velocity (and face-flux) maps at those boundaries). The event water budget uses the MAPLE-derived
`conservation.volume_roundoff_bound_m3` (no widened tolerance); the per-step balances inside the solvers use the existing
`routing.BALANCE_RTOL`. Neither candidate is claimed equivalent to the legacy MAHLERAN hydraulics.

Nothing here was run by its author (file-only tools); Codex records results.
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
from types import SimpleNamespace
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
from maple_syrup.conservation import volume_roundoff_bound_m3
from maple_syrup.dependency import MapleDependencyError
from maple_syrup.experimental_hydrology import (
    DEFAULT_CFL_MAX,
    LIMITERS,
    METHODS,
    CpuHydraulicSolver,
    ExperimentalHydrologyError,
    HydraulicControl,
    build_local_inertial_geometry,
    open_faces_from_graph,
)
from maple_syrup.experimental_storm import (
    EXPERIMENT_HYDROGRAPH_COLUMNS,
    ExperimentalControl,
    evolve_experimental,
)
from maple_syrup.infiltration import (
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
from maple_syrup.routing import RoutingError, RoutingGraphError, plot1_routing_graph
from maple_syrup.storm import StormError

__all__ = ["IMPLEMENTATIONS", "ExperimentRun", "main", "resolve_implementation", "run_plot1_experiment"]

SUMMARY_SCHEMA = "maple_syrup.experimental_hydraulics.v1"
SUMMARY_NAME = "experiment_summary.json"
FINAL_NAME = "final_state.npz"
HYDROGRAPH_NPZ = "hydrograph.npz"
HYDROGRAPH_CSV = "hydrograph.csv"
SNAPSHOTS_NAME = "snapshots.npz"
_EPS = float(np.finfo(np.float64).eps)


@dataclass(frozen=True)
class ExperimentRun:
    summary: dict[str, Any]
    verified: VerifiedPlot1Case
    hydrograph: np.ndarray
    final_depth_m: np.ndarray
    final_soil_water_m: np.ndarray


def plot1_inputs(verified: VerifiedPlot1Case, backend: str, *, with_geometry: bool) -> SimpleNamespace:
    """The Plot 1 water inputs shared by the CLI and the comparison harness (so both always use the identical forcing,
    parameters, antecedent state and graph as `storm_experiment`): parsed rainfall schedule, host/device column parameters,
    rainfall field, initial depth and soil water, the routing graph, and optionally the local-inertial geometry. `backend` is
    "numpy" or "cupy" (arrays are created in that namespace; nothing is run)."""
    from maple.core.backend import resolve_backend, to_device

    case = verified.case
    settings = verified.report["legacy_options"]["settings"]
    schedule = parse_legacy_rainfall_file(verified.rainfall_path)
    if schedule.provenance.sha256 != verified.report["rainfall"]["sha256"]:
        raise StormError("parsed rainfall bytes differ from the verified rainfall hash")
    host, parameter_record = plot1_parameters(verified.report, verified.fields)
    g = case.config.geometry
    ny, nx, n_classes = g.ny, g.nx, len(case.config.grain_classes.classes)
    if host["theta_sat"].shape != (ny, nx) or g.dx_m != g.dy_m:
        raise StormError("sidecar fields do not match the MAPLE grid, or cells are not square")
    resolved = resolve_backend(backend)
    xp = resolved.xp
    graph = plot1_routing_graph(verified.fields, verified.report, xp=xp)
    if graph.dx_m != g.dx_m:
        raise StormError("routing graph spacing differs from the MAPLE geometry")
    dev = {name: to_device(array, xp) for name, array in host.items()}
    params = column_parameters(
        model="pavement_hawkins",
        **{k: dev[k] for k in ("ksat_m_per_s", "suction_m", "drainage_parameter", "theta_sat", "soil_thickness_m",
                               "pavement_cover_fraction")},
    )
    field = rainfall_field(ny, nx, scale=dev["rainfall_scale"])
    depth0 = to_device(np.array(case.water.depth_m, dtype=np.float64, copy=True), xp)
    soil0 = initial_soil_water_m(params, dev["initial_theta"])
    geometry = None
    if with_geometry:
        full = np.asarray(verified.fields["legacy_full_elevation_m"], dtype=np.float64)
        interior = np.asarray(verified.fields["elevation_source_m"], dtype=np.float64)
        if full.shape != (ny + 2, nx + 2) or not np.array_equal(full[1:-1, 1:-1], interior):
            raise StormError("legacy_full_elevation_m interior differs from elevation_source_m")
        geometry = build_local_inertial_geometry(full, np.asarray(graph.active), np.asarray(graph.friction_factor),
                                                 graph.dx_m, open_faces_from_graph(graph))
    return SimpleNamespace(
        case=case, settings=settings, stormlength_s=float(settings["stormlength_s"]), schedule=schedule, host=host,
        parameter_record=parameter_record, grid=g, ny=ny, nx=nx, n_classes=n_classes, area=g.dx_m * g.dy_m,
        resolved=resolved, xp=xp, graph=graph, params=params, field=field, depth0=depth0, soil0=soil0, geometry=geometry)


def _positive(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not (math.isfinite(value) and value > 0.0):
        raise StormError(f"{name} must be finite and > 0, got {value!r}")
    return float(value)


IMPLEMENTATIONS = ("numpy", "numba", "cuda")
_BACKEND_IMPLEMENTATIONS = {"numpy": ("numpy", "numba"), "cupy": ("cuda",)}


def resolve_implementation(backend: str | None, implementation: str | None) -> tuple[str, str]:
    """Resolve the explicit `(backend, implementation)` pair BEFORE anything is read or built. `backend` is the array namespace
    ("numpy" | "cupy"), `implementation` the solver form ("numpy" reference | "numba" compiled CPU | "cuda"). Omitted values keep
    the historical backend-dependent selection: backend numpy -> implementation numpy, backend cupy -> implementation cuda, and
    `implementation="cuda"` alone implies backend cupy. Valid combinations: numpy+numpy, numpy+numba, cupy+cuda; every other one
    is refused (there is no silent substitution). Raises `StormError`."""
    if implementation is not None and implementation not in IMPLEMENTATIONS:
        raise StormError(f"implementation must be one of {IMPLEMENTATIONS}, got {implementation!r}")
    if backend is not None and backend not in _BACKEND_IMPLEMENTATIONS:
        raise StormError(f"backend must be 'numpy' or 'cupy', got {backend!r}")
    if backend is None:
        backend = "cupy" if implementation == "cuda" else "numpy"
    if implementation is None:
        implementation = _BACKEND_IMPLEMENTATIONS[backend][0] if backend == "numpy" else "cuda"
    if implementation not in _BACKEND_IMPLEMENTATIONS[backend]:
        raise StormError(f"implementation {implementation!r} needs backend "
                         f"{'cupy' if implementation == 'cuda' else 'numpy'}, not {backend!r} (valid pairs: numpy+numpy, "
                         "numpy+numba, cupy+cuda)")
    return backend, implementation


def run_plot1_experiment(
    case_dir: str | Path,
    output_dir: str | Path,
    *,
    solver: str = "explicit",
    backend: str | None = None,
    max_dt_s: float = 1.0,
    end_s: float | None = None,
    report_every_s: float = 60.0,
    cfl_max: float = DEFAULT_CFL_MAX,
    limiter: str = "off",
    min_dt_s: float = 1.0 / 1024.0,
    max_retries: int = 10,
    max_steps: int = 10_000_000,
    max_report_rows: int = 100_000,
    snapshot_times_s: tuple = (),
    mahleran_root: str | Path | None = None,
    expected_maple_root: str | Path | None = None,
    allow_maple_source_change: bool = False,
    implementation: str | None = None,
) -> ExperimentRun:
    from maple.core.backend import (
        gpu_execution_available,
        read_transfer_counters,
        synchronize,
        to_host,
    )
    from maple.water import validate_water_state

    if solver not in METHODS:
        raise StormError(f"solver must be one of {METHODS}, got {solver!r}")
    if backend is not None and backend not in ("numpy", "cupy"):
        raise StormError(f"backend must be 'numpy' or 'cupy', got {backend!r}")
    requested = {"backend": backend, "implementation": implementation}  # what the caller asked for (None = defaulted)
    backend, implementation = resolve_implementation(backend, implementation)
    if limiter not in LIMITERS:
        raise StormError(f"limiter must be one of {LIMITERS}, got {limiter!r}")
    max_dt = _positive(max_dt_s, "max_dt_s")
    cadence = _positive(report_every_s, "report_every_s")
    if end_s is not None:
        _positive(end_s, "end_s")
    control = ExperimentalControl(max_dt_s=max_dt_s, min_dt_s=min_dt_s, max_retries=max_retries,
                                  max_steps=max_steps).validated()
    hydraulic = HydraulicControl(cfl_max=cfl_max, limiter=limiter).validated()
    if solver == "explicit" and limiter != "off":
        raise StormError("--limiter donor belongs to the local-inertial solver")
    if isinstance(max_report_rows, bool) or not isinstance(max_report_rows, int) or max_report_rows < 1:
        raise StormError(f"max_report_rows must be a positive int, got {max_report_rows!r}")
    if backend == "cupy" and not gpu_execution_available():
        raise StormError("--backend cupy requested but CuPy or a CUDA device is unavailable; there is no fallback to the "
                         "NumPy reference (use --backend numpy explicitly)")
    if implementation == "numba":
        from maple_syrup.routing_numba import numba_available

        if not numba_available():
            raise StormError("--implementation numba requested but Numba is not importable (optional extra "
                             "maple-syrup[numba]); there is no fallback to the NumPy reference (use --implementation numpy "
                             "explicitly)")
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
    inputs = plot1_inputs(verified, backend, with_geometry=solver == "local_inertial")
    case, settings, schedule, host, parameter_record = (inputs.case, inputs.settings, inputs.schedule, inputs.host,
                                                        inputs.parameter_record)
    end = inputs.stormlength_s if end_s is None else float(end_s)
    g, ny, nx, n_classes, area = inputs.grid, inputs.ny, inputs.nx, inputs.n_classes, inputs.area
    bed_before = _bed_digest(case)
    resolved, xp, graph, params, field = inputs.resolved, inputs.xp, inputs.graph, inputs.params, inputs.field
    depth0, soil0, geometry = inputs.depth0, inputs.soil0, inputs.geometry
    setup = clock.lap()

    # --- the solver (CUDA: compile/load, validation, one-time static transfers; counted and timed apart) -----------------
    prepare_before = read_transfer_counters()
    if implementation == "cuda":
        from maple_syrup.experimental_cuda import CudaHydraulicSolver

        hydraulic_solver = CudaHydraulicSolver(solver, graph, params, geometry=geometry, control=hydraulic)
    elif implementation == "numba":
        from maple_syrup.experimental_numba import NumbaHydraulicSolver

        hydraulic_solver = NumbaHydraulicSolver(solver, graph, params, geometry=geometry, control=hydraulic)
    else:
        hydraulic_solver = CpuHydraulicSolver(solver, graph, params, geometry=geometry, control=hydraulic)
    state0 = hydraulic_solver.initial_state(depth0, soil0, t_s=0.0)
    preparation = {"timing": clock.lap(),
                   "transfer_counters": dataclasses.asdict(read_transfer_counters().delta(prepare_before))}

    # --- the loop (CUDA: two counted packet reads per attempted step; arrays stay on the device) -----------------------
    counters_before = read_transfer_counters()
    result = evolve_experimental(hydraulic_solver, field, schedule, state0, end, control, report_every_s=cadence,
                                 max_report_rows=max_report_rows, snapshot_times_s=tuple(snapshot_times_s))
    synchronize(xp)
    loop = clock.lap()
    counters_after_loop = read_transfer_counters()
    loop_transfers = dataclasses.asdict(counters_after_loop.delta(counters_before))

    # --- reporting boundary: explicit final downloads, counted apart -------------------------------------------------
    final = result.state
    names = ["surface_initial", "soil_initial", "rain", "intake", "saturation_return", "drainage", "surface_final",
             "soil_final"]
    grids = [depth0, soil0, result.cumulative_rain_m, result.cumulative_intake_m, result.cumulative_saturation_return_m,
             result.cumulative_drainage_m, final.depth_m, final.soil_water_m]
    scalars = to_host(xp.stack([xp.asarray(a.sum(), dtype=np.float64) for a in grids] + [
        xp.asarray(v, dtype=np.float64) for v in (
            result.cumulative_export_m3, result.peak_depth_m.max(), result.peak_velocity_m_s.max(),
            final.depth_m.max(), (final.depth_m > 0.0).sum(), final.soil_water_m.min(), final.soil_water_m.max(),
            result.max_cfl, result.limited_cells_total, result.limited_volume_total_m3,
            result.peak_outlet_discharge_m3_s, result.time_of_peak_outlet_s)]))
    sums = dict(zip(names, scalars[: len(names)].tolist(), strict=True))
    (export_m3, peak_depth, peak_velocity, final_max_depth, ponded_cells, soil_min, soil_max, max_cfl, limited_cells,
     limited_volume, true_peak_q, true_peak_t) = scalars[len(names):].tolist()
    hydrograph = to_host(result.hydrograph).copy()
    grids_out = {
        "depth_m": to_host(final.depth_m).copy(), "soil_water_m": to_host(final.soil_water_m).copy(),
        "velocity_m_s": to_host(result.last_velocity_m_s).copy(), "peak_depth_m": to_host(result.peak_depth_m).copy(),
        "peak_velocity_m_s": to_host(result.peak_velocity_m_s).copy(),
        "cumulative_rain_m": to_host(result.cumulative_rain_m).copy(),
        "cumulative_intake_m": to_host(result.cumulative_intake_m).copy(),
        "cumulative_return_m": to_host(result.cumulative_saturation_return_m).copy(),
        "cumulative_drainage_m": to_host(result.cumulative_drainage_m).copy(),
    }
    if final.qx_m2_s is not None:
        grids_out["qx_m2_s"] = to_host(final.qx_m2_s).copy()
        grids_out["qy_m2_s"] = to_host(final.qy_m2_s).copy()
    snapshot_out = {}
    for time_s, snap in sorted(result.snapshots.items()):
        for key, value in snap.items():
            snapshot_out[f"t{time_s:g}_{key}"] = np.asarray(to_host(value)).copy() if key != "t_s" else np.float64(value)
    reporting_transfers = dataclasses.asdict(read_transfer_counters().delta(counters_after_loop))
    if not all(np.all(np.isfinite(v)) for v in grids_out.values()) or not np.all(np.isfinite(hydrograph)):
        raise StormError("non-finite final grids or hydrograph; no output written")
    if np.any(grids_out["depth_m"] < 0.0) or np.any(grids_out["soil_water_m"] < 0.0):
        raise StormError("negative final inventory; no output written")

    water = dataclasses.replace(case.water, depth_m=grids_out["depth_m"])
    validate_water_state(water, ny, nx, n_classes)
    bed_after = _bed_digest(case)
    if bed_after != bed_before:
        raise StormError("MAPLE sediment-side state changed during a water-only run")

    # --- budget (depth sums over cells x area = m3; MAPLE-derived roundoff bound, no widened tolerance) ------------------
    n_steps, n_cells = result.n_accepted_steps, ny * nx
    volumes = {k: v * area for k, v in sums.items()}
    volumes["export"] = export_m3
    residual = (volumes["surface_final"] + volumes["soil_final"] + volumes["drainage"] + volumes["export"]
                - volumes["surface_initial"] - volumes["soil_initial"] - volumes["rain"])
    surface_residual = volumes["surface_final"] - (volumes["surface_initial"] + volumes["rain"] - volumes["intake"]
                                                   + volumes["saturation_return"] - volumes["export"])
    soil_residual = volumes["soil_final"] - (volumes["soil_initial"] + volumes["intake"] - volumes["drainage"]
                                             - volumes["saturation_return"])
    scale = max(abs(v) for v in volumes.values())
    bound = volume_roundoff_bound_m3(4 * n_cells * max(n_steps, 1) + 7, scale)
    rain_expected = schedule.depth_m(0.0, end) * float(host["rainfall_scale"].sum()) * area
    rain_bound = volume_roundoff_bound_m3(4 * n_cells * max(n_steps, 1) + 7, max(rain_expected, volumes["rain"]))
    for label, value, tol in (("water", residual, bound), ("surface", surface_residual, bound),
                              ("soil", soil_residual, bound), ("rainfall integral", volumes["rain"] - rain_expected,
                                                               rain_bound)):
        if not abs(value) <= tol:
            raise StormError(f"{label} budget residual {value} m3 exceeds the MAPLE-derived bound {tol} m3; no output written")
    columns = list(EXPERIMENT_HYDROGRAPH_COLUMNS)
    rows = {name: hydrograph[:, i] for i, name in enumerate(columns)}
    row_residual = (rows["surface_storage_m3"] + rows["soil_storage_m3"] + rows["cumulative_export_m3"]
                    + rows["cumulative_drainage_m3"]) - ((sums["surface_initial"] + sums["soil_initial"]) * area
                                                         + rows["cumulative_rain_m3"])
    if not np.all(np.abs(row_residual) <= bound):
        raise StormError("a hydrograph row violates the water balance; no output written")

    digests_after = _source_digests(Path(syrup_provenance["package_dir"]), dependency.package_dir)
    changed = [name for name in digests_before if digests_after[name] != digests_before[name]]
    if changed:
        raise StormError(f"source changed during the run ({', '.join(changed)}); no output written")

    graph_summary = graph.summary()
    graph_summary.pop("level_widths", None)
    counted_reads_per_attempt = 2 if backend == "cupy" else None
    summary = {
        "schema": SUMMARY_SCHEMA,
        "status": (f"EXPERIMENTAL water-only run of the {solver} candidate reached the configured end t = {end} s; NOT an "
                   f"event completion: residual surface water {volumes['surface_final']} m3 and outlet discharge "
                   f"{rows['outlet_discharge_m3_s'][-1]} m3/s are retained; no dry reset, no sediment, no ET, no splash; "
                   "not equivalent to the legacy MAHLERAN hydraulics"),
        "solver": solver,
        "case": {"case_dir": str(verified.case_dir),
                 "maple_case_identity_sha256": verified.binding["maple_case_identity_sha256"],
                 "maple_artifact_sha256": verified.binding["maple_artifact_sha256"], "verification": verified.checks},
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
            "maple_syrup": syrup_provenance, "maple": verified.maple_provenance,
            "maple_matches_import_binding": not verified.checks["maple_source_changed_since_import"],
            "source_stability": {"before": digests_before, "after": digests_after, "stable": True},
            "environment": environment_record(), "implementation": hydraulic_solver.implementation,
        },
        "domain": {"ny": ny, "nx": nx, "dx_m": g.dx_m, "cell_area_m2": area, "area_m2": area * n_cells,
                   "active_cells": graph.n_active, "graph": graph_summary},
        "hydraulics": hydraulic_solver.describe(),
        "boundary": ("explicit: legacy D4 outlets export the used face volume; local_inertial: ONLY the legacy outlet "
                     "faces are open with a Darcy normal-flow boundary q = k_b h^(3/2) from the DEM bed drop to the ring "
                     "cell, every other ring face and every face touching an inactive cell is closed; both differ from the "
                     "interior physics and from the legacy edge rule"),
        "time": {
            "start_s": 0.0, "end_s": end, "rainfall_end_s": schedule.end_s, "legacy_stormlength_s": float(settings["stormlength_s"]),
            "max_dt_s": max_dt, "min_dt_s": control.min_dt_s, "n_accepted_steps": n_steps,
            "n_rejected_attempts": result.n_rejected_attempts, "rejections_recorded": list(result.rejections),
            "dt_min_accepted_s": result.min_accepted_dt_s, "dt_max_accepted_s": result.max_accepted_dt_s,
            "n_boundaries": int(result.boundaries.size), "report_every_s": cadence,
            "snapshot_times_s": sorted(float(k) for k in result.snapshots),
            "policy": "boundaries = rainfall knots + reporting times + requested snapshot times + end; a rejected step "
                      "(CFL bound, or negative depth without the limiter) halves dt from the unchanged state, the cap "
                      "doubles after every clean full-size step; no clipping",
        },
        "parameters": parameter_record,
        "numerics": {"cfl_max": hydraulic.cfl_max, "limiter": hydraulic.limiter, "max_cfl_reached": max_cfl,
                     "cfl_kind": hydraulic_solver.describe()["cfl_kind"], "limited_cells_total": limited_cells,
                     "limited_volume_total_m3": limited_volume},
        "budget": {
            "units": "m3", **{f"{k}_m3": v for k, v in volumes.items()}, "rain_expected_m3": rain_expected,
            "water_residual_m3": residual, "surface_residual_m3": surface_residual, "soil_residual_m3": soil_residual,
            "bound_m3": bound, "rainfall_bound_m3": rain_bound,
            "bound_rule": "conservation.volume_roundoff_bound_m3(4 n_cells n_steps + 7, largest operand): the MAPLE "
                          "summation coefficient (as the Phase 7i benchmark); the solvers' own per-step balances use "
                          "routing.BALANCE_RTOL",
            "identity": "surface_final + soil_final + drainage + export = surface_initial + soil_initial + rain",
            "export_meaning": "time-integrated face volume through the outlet faces; outlet_discharge_m3_s in the "
                              "hydrograph is the instantaneous end-of-step value",
            "runoff_coefficient_export_over_rain": export_m3 / volumes["rain"] if volumes["rain"] > 0 else None,
        },
        "final_state": {
            "ponded_cells": int(ponded_cells), "max_depth_m": final_max_depth,
            "peak_depth_m": peak_depth, "peak_velocity_m_s": peak_velocity,
            "final_outlet_discharge_m3_s": float(rows["outlet_discharge_m3_s"][-1]),
            "peak_outlet_discharge_m3_s": true_peak_q, "time_of_peak_outlet_discharge_s": true_peak_t,
            "soil_water_range_m": [soil_min, soil_max],
            "continuation": {
                "state_t_s": float(final.t_s), "state_next_dt_cap_s": float(final.next_dt_cap_s),
                "numerical_state": ("depth_m, soil_water_m" + (", qx_m2_s, qy_m2_s" if final.qx_m2_s is not None else "")
                                    + " (final_state.npz) plus state_t_s and the adaptive step cap state_next_dt_cap_s"),
                "why": "the adaptive step cap shapes the step sequence and therefore the trajectory; resuming from the arrays "
                       "alone would restart the cap at max_dt_s and follow a different path",
                "disk_restart": "not implemented; in memory, passing result.state back to evolve_experimental continues exactly",
            },
            "field_meaning": "peak depth/velocity = maxima over accepted steps of the END-of-step values of the method; no "
                             "pickup, travel distance or detachment is computed (water only)",
        },
        "maple_state": {"sediment_digest_before": bed_before, "sediment_digest_after": bed_after, "unchanged": True,
                        "covers": "voxel column, active layer, availability, water mobile mass, ledger, topography"},
        "backend": {
            "backend": resolved.backend.value, "device_id": resolved.device_id,
            "fingerprint": json.loads(json.dumps(resolved.fingerprint, default=str)),
            "preparation": preparation, "loop_transfer_counters": loop_transfers,
            "reporting_transfer_counters": reporting_transfers,
            "counted_packet_reads_per_attempt": counted_reads_per_attempt,
            "attempts": n_steps + result.n_rejected_attempts,
            "transfer_scope": hydraulic_solver.describe()["transfer_scope"],
            "requested": requested, "resolved": {"backend": backend, "implementation": implementation},
            "gpu": ("experimental CUDA candidate (water only); not a production qualification and no performance claim"
                    if implementation == "cuda" else
                    "compiled Numba CPU form of the NumPy reference (new, unqualified beyond its tests); no GPU involved"
                    if implementation == "numba" else "NumPy reference; no GPU involved"),
        },
        "timings": {
            "setup_and_verification": setup, "preparation_and_state_check": preparation["timing"],
            "first_accepted_step": {"wall_s": result.first_step_wall_s, "cpu_s": result.first_step_cpu_s},
            "remaining_steps": {"wall_s": result.remaining_wall_s, "cpu_s": result.remaining_cpu_s,
                                "n_steps": max(n_steps - 1, 0)},
            "step_loop": loop, "note": "process CPU and wall seconds of one run; not a controlled benchmark",
        },
        "hydrograph": {"columns": columns, "n_rows": int(hydrograph.shape[0]), "files": [HYDROGRAPH_CSV, HYDROGRAPH_NPZ]},
        "assumptions": [
            "fixed bed, elevation and (explicit) routing graph; no sediment exchange, no terrain refresh",
            "constant friction factor from the legacy type-1 mean; deterministic Ksat mean (as storm_experiment)",
            "accepted MAHLERAN-inspired column physics (infiltration, drainage, saturation return) unchanged",
            "no evapotranspiration, dry reset, plants, splash or sediment",
        ],
        "limitations": [
            "an experimental alternative: not equivalent to the legacy hydraulics, not an event completion",
            "first-order explicit / local-inertial approximations; scalar directional friction (own face component only)",
            ("no restart to disk; final grids written once with state_t_s and state_next_dt_cap_s, the numerical adaptive "
             "history needed to continue exactly in memory"),
            ("the local-inertial backend is NOT field-equivalent over a full 5400 s storm at the 2e-12 / 1e-14 bounds "
             "(root trialC2 CUDA and the compiled Numba form, both against the NumPy reference: water closes and export agrees "
             "to ~5e-14 m3, but some fields exceed the bounds); see "
             "docs/hydraulic_candidates/README.md, 'Qualification status'"),
            ("local inertia: velocity_m_s / peak_velocity_m_s are RECONSTRUCTED cell speeds (face-averaged new flux / "
             "END-of-step depth) and are unbounded near drying cells (measured Fr up to ~4e4 at h ~ 1e-11 m in the root's "
             "trialC1 diagnostics); nothing is clipped, they are NOT qualified for detachment or transport. "
             "Stage-consistent face velocity/Froude (experimental_hydrology.stage_face_diagnostics) separate the cell "
             "reconstruction from the face flows, but are NOT a fix: the root's trialC2 stage diagnostics show face Fr up to "
             "~2 at 600 s and up to ~0.4% of wet faces above 0.5, outside the low-Froude range where local inertia is "
             "accurate (advection, directional damping, wetting/drying and outlet treatment are not qualified)"),
        ],
    }
    output_dir.mkdir(parents=True)
    with (output_dir / FINAL_NAME).open("xb") as handle:
        np.savez(handle, **grids_out, state_t_s=np.float64(final.t_s),
                 state_next_dt_cap_s=np.float64(final.next_dt_cap_s))  # the portable numeric state is not only the arrays
    interval_names = ("cumulative_rain_m3", "cumulative_intake_m3", "cumulative_saturation_return_m3",
                      "cumulative_drainage_m3", "cumulative_export_m3")
    interval = {name: np.diff(rows[name], prepend=0.0) for name in interval_names}
    with (output_dir / HYDROGRAPH_NPZ).open("xb") as handle:
        np.savez(handle, **rows, **{f"interval_{k[len('cumulative_'):]}": v for k, v in interval.items()},
                 row_water_residual_m3=row_residual)
    with (output_dir / HYDROGRAPH_CSV).open("x", newline="", encoding="ascii") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns + [f"interval_{k[len('cumulative_'):]}" for k in interval] + ["row_water_residual_m3"])
        for i in range(hydrograph.shape[0]):
            writer.writerow([repr(float(v)) for v in hydrograph[i]] + [repr(float(interval[k][i])) for k in interval]
                            + [repr(float(row_residual[i]))])
    produced = [FINAL_NAME, HYDROGRAPH_NPZ, HYDROGRAPH_CSV]
    if snapshot_out:
        with (output_dir / SNAPSHOTS_NAME).open("xb") as handle:
            np.savez(handle, **snapshot_out)
        produced.append(SNAPSHOTS_NAME)
    summary["outputs"] = {name: hashlib.sha256((output_dir / name).read_bytes()).hexdigest() for name in produced}
    summary["timings"]["reporting"] = clock.lap()
    _write_new_json(output_dir / SUMMARY_NAME, summary)
    return ExperimentRun(summary=summary, verified=verified, hydrograph=hydrograph,
                         final_depth_m=grids_out["depth_m"], final_soil_water_m=grids_out["soil_water_m"])


def _snapshot_times(text: str | None) -> tuple:
    if not text:
        return ()
    try:
        return tuple(float(t) for t in text.split(","))
    except ValueError as exc:
        raise StormError(f"--snapshot-times-s must be a comma-separated list of seconds, got {text!r}") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m maple_syrup.experimental_experiment",
        description="EXPERIMENTAL water-only explicit-kinematic / local-inertial runs on the imported Plot 1 case "
                    "(no sediment, no reset; not the legacy hydraulics).")
    parser.add_argument("--case-dir", required=True, help="Verified Phase 2 case, e.g. outputs/plot1")
    parser.add_argument("--output-dir", required=True, help="NEW directory for summary, grids, hydrograph and snapshots.")
    parser.add_argument("--solver", required=True, choices=METHODS)
    parser.add_argument("--backend", default=None, choices=("numpy", "cupy"),
                        help="array namespace: numpy (default) or cupy (needs CuPy + device; never Numba; no fallback)")
    parser.add_argument("--implementation", default=None, choices=IMPLEMENTATIONS,
                        help="solver form: numpy = NumPy reference, numba = compiled CPU form (needs numpy backend and Numba), "
                             "cuda = CUDA form (needs cupy). Omitted: numpy for --backend numpy, cuda for --backend cupy. "
                             "Invalid pairs are refused before the case is read; there is never a silent fallback")
    parser.add_argument("--max-dt-s", type=float, default=1.0)
    parser.add_argument("--end-s", type=float, default=None, help="default: the legacy stormlength (5400 s for Plot 1)")
    parser.add_argument("--report-every-s", type=float, default=60.0)
    parser.add_argument("--cfl-max", type=float, default=DEFAULT_CFL_MAX, help="CFL bound in (0, 0.5]")
    parser.add_argument("--limiter", default="off", choices=LIMITERS,
                        help="'donor' = documented conservative positivity limiter (local inertia only); default off")
    parser.add_argument("--snapshot-times-s", default=None, help="comma-separated times for synchronous snapshot maps")
    parser.add_argument("--min-dt-s", type=float, default=1.0 / 1024.0)
    parser.add_argument("--max-retries", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=10_000_000)
    parser.add_argument("--mahleran-root", help="Override the recipe's MAHLERAN root (read-only).")
    parser.add_argument("--expected-maple-root", help="Refuse any other MAPLE source root.")
    parser.add_argument("--allow-maple-source-change", action="store_true",
                        help="Run even if MAPLE's source digest differs from the import's (recorded).")
    args = parser.parse_args(argv)
    try:
        run = run_plot1_experiment(
            args.case_dir, args.output_dir, solver=args.solver, backend=args.backend, max_dt_s=args.max_dt_s,
            end_s=args.end_s, report_every_s=args.report_every_s, cfl_max=args.cfl_max, limiter=args.limiter,
            min_dt_s=args.min_dt_s, max_retries=args.max_retries, max_steps=args.max_steps,
            snapshot_times_s=_snapshot_times(args.snapshot_times_s), mahleran_root=args.mahleran_root,
            expected_maple_root=args.expected_maple_root, allow_maple_source_change=args.allow_maple_source_change,
            implementation=args.implementation)
    except MapleDependencyError as exc:
        print(f"MAPLE dependency check failed: {exc}", file=sys.stderr)
        return 2
    except (Plot1ImportError, StormError, RoutingError, RoutingGraphError, InfiltrationError, RainfallError,
            ExperimentalHydrologyError, ValueError) as exc:
        print(f"experimental run failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(_json_safe({k: run.summary[k] for k in ("status", "solver", "time", "numerics", "budget",
                                                              "final_state", "timings")}), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
