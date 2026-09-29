"""Phase 3b: Plot 1 no-routing infiltration column diagnostic.

Every cell of the verified Phase 2 Plot 1 case is an independent soil-water
column under the Plot 1 rainfall, from t = 0 to the end of the rainfall
record. There is NO routing, no sediment exchange, no evapotranspiration
and no dry reset: ponded water stays where it was generated, and the run
ends at the end of rainfall, which is not the end of an event.

    python -m maple_syrup.column_experiment --case-dir outputs/plot1 \\
        --max-dt-s 1 --output-dir outputs/plot1_columns

The case is re-verified first (`case_import.verify_plot1_case`), the
rainfall is parsed from its hash-checked staged copy, and parameters come
from the verified legacy settings and sidecar arrays. Steps split at every
rainfall knot and at `--max-dt-s`. The MAPLE bed, availability, mobile
sediment and ledger are never passed to the kernel; their digests are
compared before and after. The final surface depth is returned as a MAPLE
`WaterState` (the loaded case's own mobile array, depth replaced).

Outputs (NEW directory, never inside the case, MAPLE, MAHLERAN or recipe
trees): `column_summary.json` (budget, parameters, provenance, timings)
and `final_columns.npz` (final grids only; no per-step history). The
SYRUP and MAPLE package source digests are recorded (with git state) and
must be unchanged from before verification to after the step loop, or
nothing is written.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from maple_syrup.case_import import (
    FIELDS_NAME,
    Plot1ImportError,
    VerifiedPlot1Case,
    _json_safe,
    _refuse_output,
    _write_new_json,
    verify_plot1_case,
)
from maple_syrup.dependency import MapleDependencyError
from maple_syrup.infiltration import (
    LOCAL_BALANCE_RTOL,
    InfiltrationError,
    column_parameters,
    column_step,
    initial_soil_water_m,
)
from maple_syrup.provenance import (
    capture_syrup_provenance,
    environment_record,
    read_git_head,
    scoped_git_status,
    source_tree_digest,
)
from maple_syrup.rainfall import (
    RainfallError,
    parse_legacy_rainfall_file,
    rainfall_field,
)

__all__ = ["ColumnExperimentError", "ColumnRun", "main", "plot1_parameters", "run_plot1_columns"]

SUMMARY_SCHEMA = "maple_syrup.plot1_columns.v1"
SUMMARY_NAME = "column_summary.json"
FINAL_NAME = "final_columns.npz"
_EPS = float(np.finfo(np.float64).eps)


class ColumnExperimentError(RuntimeError):
    """The case, its settings or the run request cannot be used."""


# --------------------------------------------------------------------------
# Parameters from the verified legacy settings
# --------------------------------------------------------------------------
def plot1_parameters(report: dict[str, Any], fields: dict[str, np.ndarray]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Host FP64 `(ny, nx)` inputs for `column_parameters` plus the initial
    moisture and rainfall scale, and a record of where each came from.

    Only the configuration Plot 1 actually uses is accepted: infiltration
    model 2, parameter type 2, one surface type, rainfall type 2, no
    suction/drainage/initial-moisture/final-infiltration maps, a
    saturated-moisture map. Ksat is the XML mean (the configured 'normal'
    draw is replaced by its positive mean; no random sampling)."""
    s = report["legacy_options"]["settings"]
    use, dist = s["use_flags"], s["distributions"]
    expected = {
        "infiltration_model": 2, "infiltration_parameter_type": 2, "rain_type": 2, "number_of_surface_types": 1,
    }
    for key, value in expected.items():
        if s[key] != value:
            raise ColumnExperimentError(f"legacy {key} = {s[key]!r}; this diagnostic supports only {value!r}")
    for flag, value in (("use_final_infiltration_map", False), ("use_suction_map", False),
                        ("use_drainage_map", False), ("use_initial_soil_moisture_map", False),
                        ("use_saturated_soil_moisture_map", True)):
        if use[flag] is not value:
            raise ColumnExperimentError(f"legacy {flag} = {use[flag]!r}; expected {value!r}")
    for key in ("wettingFrontSuctionDistribution", "drainageParameterDistribution",
                "initialSoilMoistureDistribution"):
        if dist[key] != "deterministic":
            raise ColumnExperimentError(f"legacy {key} = {dist[key]!r}; only 'deterministic' is supported")
    stype = fields["surface_type_resolved"]
    if not np.all(stype == 1):
        raise ColumnExperimentError("surface types other than 1 are present")

    def type1(tag: str) -> float:
        value = float(s["by_surface_type"][tag]["type_1"])
        if not math.isfinite(value):
            raise ColumnExperimentError(f"legacy {tag} is not finite")
        return value

    ksat_mm_s = type1("final_infiltration_rate_mean")
    suction_mm = type1("wetting_front_suction_mean")
    drain = type1("drainage_parameter_mean")
    theta0 = type1("initial_soil_moisture_mean")
    thickness_m = type1("soil_thickness")
    theta_sat = np.asarray(fields["saturated_soil_moisture"], dtype=np.float64)
    shape = theta_sat.shape

    def full(value: float) -> np.ndarray:
        return np.full(shape, value, dtype=np.float64)

    inputs = {
        "ksat_m_per_s": full(ksat_mm_s * 1.0e-3),
        "suction_m": full(suction_mm * 1.0e-3),
        "drainage_parameter": full(drain),
        "theta_sat": theta_sat,
        "soil_thickness_m": full(thickness_m),
        "pavement_cover_fraction": np.asarray(fields["pavement_cover_fraction"], dtype=np.float64),
    }
    extra = {"initial_theta": full(theta0),
             "rainfall_scale": np.asarray(fields["rainfall_scaling"], dtype=np.float64)}
    record = {
        "model": "pavement_hawkins (legacy infiltration_model 2, infiltration-parameter_type 2)",
        "ksat_m_per_s": {
            "value": ksat_mm_s * 1.0e-3, "xml_mean_mm_per_s": ksat_mm_s,
            "xml_std_dev_mm_per_s": s["by_surface_type"]["final_infiltration_std_dev"]["type_1"],
            "xml_distribution": dist["finalInfiltrationRateDistribution"],
            "decision": "deterministic XML mean; the configured normal draw (std > mean) could be "
                        "negative and is not sampled. Not a reproduction of a legacy realization.",
            "used_for": "K when local rain is zero; drainage demand",
        },
        "suction_m": {"value": suction_mm * 1.0e-3, "xml_mean_mm": suction_mm, "psi_mod": 1.0},
        "drainage_parameter": {"value": drain},
        "initial_theta": {"value": theta0, "retained_soil_water_m": theta0 * thickness_m},
        "soil_thickness_m": {"value": thickness_m,
                             "note": "soil-water depth only; unrelated to the MAPLE sediment column depth"},
        "theta_sat": {"source": f"sidecar {FIELDS_NAME}: saturated_soil_moisture (legacy map)",
                      "range": [float(theta_sat.min()), float(theta_sat.max())]},
        "pavement_cover_fraction": {"source": f"sidecar {FIELDS_NAME}: pavement_cover_fraction",
                                    "range": [float(inputs["pavement_cover_fraction"].min()),
                                              float(inputs["pavement_cover_fraction"].max())],
                                    "legacy_pave": "fraction * 1e-2 (percent * 1e-4)"},
        "rainfall_scale": {"source": f"sidecar {FIELDS_NAME}: rainfall_scaling (legacy rmask interior)",
                           "unique": sorted({float(v) for v in np.unique(extra["rainfall_scale"])})},
        "calibration": "ksat_mod = psi_mod = 1 (no calib.dat; checked)",
    }
    return {**inputs, **extra}, record


# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ColumnRun:
    summary: dict[str, Any]
    verified: VerifiedPlot1Case
    water: Any  # MAPLE WaterState: final depth, the loaded case's mobile array
    soil_water_m: np.ndarray  # final retained soil water (host)


def _substeps(schedule, max_dt_s: float) -> list[tuple[float, int, float]]:
    """(dt, n, rate) per constant-rate segment from t = 0 to the end of
    rainfall: every knot is a step boundary and every dt <= max_dt_s."""
    edges, rates = schedule.edges_s.tolist(), schedule.rate_m_per_s.tolist()
    segments = [(0.0, edges[0], 0.0)] if edges[0] > 0.0 else []
    segments += [(edges[k], edges[k + 1], rates[k]) for k in range(len(rates))]
    plan = []
    for start, end, rate in segments:
        span = end - start
        n = max(1, math.ceil(span / max_dt_s))
        while span / n > max_dt_s:
            n += 1
        plan.append((span / n, n, rate))
    return plan


def _bed_digest(case: Any) -> str:
    """SHA-256 over every array of the MAPLE sediment-side state."""
    from maple.core.backend import array_leaves, to_host

    hasher = hashlib.sha256()
    for array in array_leaves(case.voxel_column, case.active_layer, case.sediment_availability,
                              case.water.mobile_mass_by_cell_class_kg, case.sediment_ledger,
                              case.topography_result):
        host = np.ascontiguousarray(to_host(array))
        hasher.update(f"{host.dtype.str}{host.shape}".encode())
        hasher.update(host.tobytes())
    return hasher.hexdigest()


# SYRUP paths whose git state is recorded with the run (src layout: repo = package_dir/../..).
_SYRUP_GIT_SCOPE = ("src", "tests", "docs", "cases", "pyproject.toml")


def _syrup_provenance() -> dict[str, Any]:
    """`capture_syrup_provenance` (package source digest) plus the repository
    HEAD and a scoped `git status`, so an uncommitted implementation is
    identified by its content digest and flagged as dirty."""
    record = capture_syrup_provenance()
    repo = Path(record["package_dir"]).parents[1]
    git = read_git_head(repo)
    if git["status"] == "ok":
        git["scoped_status"] = scoped_git_status(repo, _SYRUP_GIT_SCOPE)
        git["head_describes_source"] = git["scoped_status"]["status"] == "clean"
    record["repository_root"] = str(repo)
    record["git"] = git
    return record


def _source_digests(syrup_package_dir: Path, maple_package_dir: Path) -> dict[str, str]:
    """Current content digests of the SYRUP and imported MAPLE packages."""
    return {
        "maple_syrup": source_tree_digest(syrup_package_dir).digest_sha256,
        "maple": source_tree_digest(maple_package_dir).digest_sha256,
    }


class _Clock:
    def __init__(self) -> None:
        self.cpu, self.wall = time.process_time(), time.perf_counter()

    def lap(self) -> dict[str, float]:
        cpu, wall = time.process_time(), time.perf_counter()
        record = {"cpu_s": cpu - self.cpu, "wall_s": wall - self.wall}
        self.cpu, self.wall = cpu, wall
        return record


def run_plot1_columns(
    case_dir: str | Path,
    output_dir: str | Path,
    *,
    max_dt_s: float,
    backend: str = "numpy",
    mahleran_root: str | Path | None = None,
    expected_maple_root: str | Path | None = None,
    allow_maple_source_change: bool = False,
) -> ColumnRun:
    from maple.core.backend import (
        read_transfer_counters,
        resolve_backend,
        synchronize,
        to_device,
        to_host,
    )
    from maple.water import validate_water_state

    if isinstance(max_dt_s, bool) or not isinstance(max_dt_s, (int, float)) or not (
        math.isfinite(max_dt_s) and max_dt_s > 0.0
    ):
        raise ColumnExperimentError(f"max_dt_s must be finite and > 0, got {max_dt_s!r}")
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise ColumnExperimentError(f"refusing to write into existing path {output_dir}")
    if output_dir.is_relative_to(Path(case_dir).resolve()):
        raise ColumnExperimentError("refusing to write inside the bound case directory")

    clock = _Clock()
    syrup_provenance = _syrup_provenance()
    verified = verify_plot1_case(case_dir, mahleran_root=mahleran_root, expected_maple_root=expected_maple_root,
                                 allow_maple_source_change=allow_maple_source_change)
    dependency = verified.maple_dependency
    recipe_record = verified.report["recipe"]
    # The actual roots in use, not only the optional expected-root flags.
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
    schedule = parse_legacy_rainfall_file(verified.rainfall_path)
    if schedule.provenance.sha256 != verified.report["rainfall"]["sha256"]:
        raise ColumnExperimentError("parsed rainfall bytes differ from the verified rainfall hash")
    host, parameter_record = plot1_parameters(verified.report, verified.fields)
    g = case.config.geometry
    ny, nx, n_classes = g.ny, g.nx, len(case.config.grain_classes.classes)
    if host["theta_sat"].shape != (ny, nx):
        raise ColumnExperimentError("sidecar fields do not match the MAPLE grid")
    cell_area = g.dx_m * g.dy_m
    bed_before = _bed_digest(case)

    resolved = resolve_backend(backend)
    xp = resolved.xp
    dev = {name: to_device(array, xp) for name, array in host.items()}
    params = column_parameters(
        model="pavement_hawkins",
        **{k: dev[k] for k in ("ksat_m_per_s", "suction_m", "drainage_parameter", "theta_sat",
                               "soil_thickness_m", "pavement_cover_fraction")},
    )
    field = rainfall_field(ny, nx, scale=dev["rainfall_scale"])
    depth0 = to_device(np.array(case.water.depth_m, dtype=np.float64, copy=True), xp)
    soil0 = initial_soil_water_m(params, dev["initial_theta"])
    plan = _substeps(schedule, float(max_dt_s))
    setup = clock.lap()

    # --- step loop: arrays stay in xp; one batched flag read per step ---------
    counters_before = read_transfer_counters()
    depth, soil = depth0, soil0
    totals = {k: xp.zeros((ny, nx), dtype=np.float64) for k in ("rain", "intake", "return", "drainage")}
    rate = xp.empty((ny, nx), dtype=np.float64)
    n_steps = 0
    for dt, n, rate_value in plan:
        field.apply(rate_value, out=rate)
        for _ in range(n):
            step = column_step(params, depth, soil, rate, dt)
            depth, soil = step.depth_m, step.soil_water_m
            totals["rain"] += step.rain_m
            totals["intake"] += step.intake_m
            totals["return"] += step.saturation_return_m
            totals["drainage"] += step.drainage_m
            n_steps += 1
    synchronize(xp)
    loop = clock.lap()
    transfers = dataclasses.asdict(read_transfer_counters().delta(counters_before))

    # --- reporting boundary: one stacked read of every scalar ----------------
    cell_residual = (depth + soil + totals["drainage"]) - (depth0 + soil0 + totals["rain"])
    names = ["surface_initial", "soil_initial", "rain", "intake", "saturation_return", "drainage",
             "surface_final", "soil_final"]
    arrays = [depth0, soil0, totals["rain"], totals["intake"], totals["return"], totals["drainage"], depth, soil]
    scalars = to_host(xp.stack([a.sum() for a in arrays] + [
        xp.abs(cell_residual).max(), depth.max(), (depth > 0.0).sum().astype(np.float64),
        (soil / params.soil_thickness_m).min(), (soil / params.soil_thickness_m).max(),
        totals["return"].max(), totals["drainage"].max(),
    ]))
    sums = dict(zip(names, scalars[: len(names)].tolist()))
    (cell_residual_max, depth_max, ponded_cells, theta_min, theta_max,
     return_max, drainage_max) = scalars[len(names):].tolist()
    final_depth, final_soil = to_host(depth).copy(), to_host(soil).copy()
    final_totals = {k: to_host(v).copy() for k, v in totals.items()}

    water = dataclasses.replace(case.water, depth_m=final_depth)
    validate_water_state(water, ny, nx, n_classes)
    bed_after = _bed_digest(case)
    if bed_after != bed_before:
        raise ColumnExperimentError("MAPLE sediment-side state changed during a no-exchange run")

    # --- budget ---------------------------------------------------------------
    inventory = sums["surface_initial"] + sums["soil_initial"] + sums["rain"]
    magnitude = inventory + sums["intake"] + sums["drainage"] + sums["saturation_return"]
    tolerance = LOCAL_BALANCE_RTOL * (n_steps + ny * nx) * magnitude
    water_residual = (sums["surface_final"] + sums["soil_final"] + sums["drainage"]) - inventory
    surface_residual = sums["surface_final"] - (sums["surface_initial"] + sums["rain"] - sums["intake"]
                                                + sums["saturation_return"])
    soil_residual = sums["soil_final"] - (sums["soil_initial"] + sums["intake"] - sums["drainage"]
                                          - sums["saturation_return"])
    rain_expected = schedule.total_depth_m() * float(host["rainfall_scale"].sum())
    rain_tolerance = 4.0 * _EPS * (n_steps + ny * nx) * rain_expected
    for label, residual, tol in (("water", water_residual, tolerance), ("surface", surface_residual, tolerance),
                                 ("soil", soil_residual, tolerance),
                                 ("rainfall integral", sums["rain"] - rain_expected, rain_tolerance)):
        if not abs(residual) <= tol:
            raise ColumnExperimentError(f"{label} budget residual {residual} m exceeds tolerance {tol} m")

    # Code identity: the SYRUP and MAPLE sources that ran must be the ones recorded.
    digests_after = _source_digests(Path(syrup_provenance["package_dir"]), dependency.package_dir)
    changed = [name for name in digests_before if digests_after[name] != digests_before[name]]
    if changed:
        raise ColumnExperimentError(f"source changed during the run ({', '.join(changed)}); no output written")

    dts = [dt for dt, _, _ in plan]
    summary = {
        "schema": SUMMARY_SCHEMA,
        "status": "rainfall_window_complete; NOT an event completion (ponded water retained, no routing)",
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
            "source_stability": {
                "before": digests_before, "after": digests_after, "stable": True,
                "rule": "package source digests taken before verification and after the step loop; "
                        "any change aborts before outputs are written",
            },
            "environment": environment_record(),
            "note": "an uncommitted implementation is identified by package_source_digest; "
                    "git.scoped_status lists the uncommitted paths",
        },
        "domain": {"ny": ny, "nx": nx, "dx_m": g.dx_m, "dy_m": g.dy_m, "cell_area_m2": cell_area,
                   "area_m2": cell_area * ny * nx, "active_cells": ny * nx,
                   "boundaries": "irrelevant: no lateral water exchange"},
        "time": {"start_s": 0.0, "end_s": schedule.end_s, "rainfall_end_s": schedule.end_s,
                 "legacy_stormlength_s": verified.report["legacy_options"]["settings"]["stormlength_s"],
                 "max_dt_s": float(max_dt_s), "n_steps": n_steps, "n_constant_rate_segments": len(plan),
                 "dt_min_s": min(dts), "dt_max_s": max(dts),
                 "policy": "every rainfall knot is a step boundary; each segment split into equal dt <= max_dt_s"},
        "parameters": parameter_record,
        "budget": {
            "units": "volume m3 (= depth sum over cells x cell area)",
            "surface_initial_m3": sums["surface_initial"] * cell_area,
            "soil_initial_m3": sums["soil_initial"] * cell_area,
            "rain_m3": sums["rain"] * cell_area,
            "rain_mean_depth_m": sums["rain"] / (ny * nx),
            "rain_expected_m3": rain_expected * cell_area,
            "intake_m3": sums["intake"] * cell_area,
            "saturation_return_m3": sums["saturation_return"] * cell_area,
            "net_infiltration_m3": (sums["intake"] - sums["saturation_return"]) * cell_area,
            "drainage_m3": sums["drainage"] * cell_area,
            "surface_final_m3": sums["surface_final"] * cell_area,
            "soil_final_m3": sums["soil_final"] * cell_area,
            "water_residual_m3": water_residual * cell_area,
            "surface_residual_m3": surface_residual * cell_area,
            "soil_residual_m3": soil_residual * cell_area,
            "rainfall_integral_residual_m3": (sums["rain"] - rain_expected) * cell_area,
            "tolerance_m3": tolerance * cell_area,
            "rainfall_tolerance_m3": rain_tolerance * cell_area,
            "tolerance_rule": "16 eps (n_steps + n_cells) (initial + rain + intake + drainage + return) depth sums",
            "max_abs_cell_cumulative_residual_m": cell_residual_max,
            "identity": "surface_final + soil_final + drainage = surface_initial + soil_initial + rain",
            "drainage_fate": "leaves the column; not routed or stored (legacy linear drain)",
        },
        "final_state": {
            "ponded_cells": int(ponded_cells), "max_depth_m": depth_max,
            "mean_depth_m": sums["surface_final"] / (ny * nx),
            "theta_range": [theta_min, theta_max],
            "max_cell_saturation_return_m": return_max, "max_cell_drainage_m": drainage_max,
            "ponded_water": "retained in MAPLE WaterState.depth_m; rainfall end is not event end",
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
            "validation": "one batched flag read per step (scalar sync on a device); no grid transfer "
                          "in the loop; grids copied to the host once at the end",
        },
        "timings": {"setup_and_verification": setup, "step_loop": loop,
                    "note": "process CPU and wall seconds; not a performance benchmark"},
        "assumptions": [
            "independent columns: no routing, run-on or outlet; surface water is storage, not discharge",
            "deterministic XML mean Ksat (no stochastic realization)",
            "legacy model-2 K uses each cell's own rain rate (legacy tested r2(i, 2))",
            "capacity uses the pre-step ponded depth h in (psi + h)",
            ("capacity uses retained soil water S (legacy cum_inf, incl. antecedent water) where "
            "Smith-Parlange uses cumulative infiltration; a MAHLERAN-inspired choice kept as is, "
            "which makes the capillary term negligible for Plot 1 (scientific limitation)"),
            "drainage leaves the column and is not tracked further",
            "no evapotranspiration, dry reset, plants, splash or sediment exchange",
        ],
        "limitations": [
            "not a MAHLERAN Fortran benchmark; no Fortran executable was run",
            "no GPU result unless backend is cupy on an available device; no performance claim",
            "no production restart; final grids are written once for inspection",
        ],
    }

    output_dir.mkdir(parents=True)
    final_path = output_dir / FINAL_NAME
    with final_path.open("xb") as handle:
        np.savez(handle, depth_m=final_depth, soil_water_m=final_soil,
                 **{f"cumulative_{k}_m": v for k, v in final_totals.items()})
    summary["outputs"] = {FINAL_NAME: hashlib.sha256(final_path.read_bytes()).hexdigest()}
    summary["timings"]["reporting"] = clock.lap()
    _write_new_json(output_dir / SUMMARY_NAME, summary)
    return ColumnRun(summary=summary, verified=verified, water=water, soil_water_m=final_soil)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m maple_syrup.column_experiment",
        description="MAPLE-SYRUP Phase 3b: Plot 1 no-routing infiltration columns over the rainfall window.",
    )
    parser.add_argument("--case-dir", required=True, help="Verified Phase 2 case, e.g. outputs/plot1")
    parser.add_argument("--max-dt-s", required=True, type=float, help="Largest step (s); knots always split.")
    parser.add_argument("--output-dir", required=True, help="NEW directory for the summary and final grids.")
    parser.add_argument("--backend", default="numpy", choices=("numpy", "cupy"))
    parser.add_argument("--mahleran-root", help="Override the recipe's MAHLERAN root (read-only).")
    parser.add_argument("--expected-maple-root", help="Refuse any other MAPLE source root.")
    parser.add_argument("--allow-maple-source-change", action="store_true",
                        help="Run even if MAPLE's source digest differs from the import's (recorded).")
    args = parser.parse_args(argv)
    try:
        run = run_plot1_columns(
            args.case_dir, args.output_dir, max_dt_s=args.max_dt_s, backend=args.backend,
            mahleran_root=args.mahleran_root, expected_maple_root=args.expected_maple_root,
            allow_maple_source_change=args.allow_maple_source_change,
        )
    except MapleDependencyError as exc:
        print(f"MAPLE dependency check failed: {exc}", file=sys.stderr)
        return 2
    except (Plot1ImportError, ColumnExperimentError, InfiltrationError, RainfallError, ValueError) as exc:
        print(f"plot1 column diagnostic failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(_json_safe({k: run.summary[k] for k in ("status", "time", "budget", "final_state")}),
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
