"""Phase 7: matched fixed-terrain wet benchmark of Plot 1 on the actual MAPLE bed.

    python -m maple_syrup.benchmark_experiment --case-dir outputs/plot1 \\
        --applied-rainfall outputs/phase7/mahleran_reference_audit/applied_rainfall.csv \\
        --reference-run outputs/phase7/mahleran_deterministic_ksat_run \\
        --output-dir outputs/phase7/syrup_matched_dt1 --max-dt-s 1 --implementation numba

What this is
------------
The SAME accepted Phase 5 physical step (`sediment_event.evolve_sediment_event`:
Phase 3 column, Phase 4b method-5 routing, wet MAHLERAN laws, conservative
lateral transport, actual MAPLE pickup / refill / deposition / export) driven
over a FIXED window with the hydraulic geometry frozen, so that the run can be
compared with the whole-program MAHLERAN reference executed with direct dry-cell
splash disabled, `update_topography=n` and deterministic conductivity
(docs/phase7/mahleran_reference_run.md, initialization_audit.md).

Frozen hydraulic geometry: `SedimentEventControl(commit=False,
force_final_commit=False)`. MAPLE's commit triggers are never evaluated, so the
committed topography, the routing graph (aspect, slope, receivers, outlets,
conveyance), the transport network and the physics grid stay the objects built
at t = 0; every MAPLE water call still moves real mass (active layer, voxel
refill, availability, mobile pool, export) and the pending ledger accumulates
for the whole window. The run asserts this freeze at the end (object identity
and array equality) and runs MAPLE's own validators on the final state,
including the ledger-to-physical reconciliation against the unchanged committed
inventory. Net bed mass change is reported separately from the frozen
elevation; it is what the actual MAPLE inventory did, not a terrain update.

Forcing: the legacy program applies each rainfall record one step late and the
first record one extra second (Set_rain_xml.f90 138-160 with the call after
routing, MAHLERAN_storm_xml.f90 133). The benchmark therefore reads the
per-step APPLIED rainfall logged by the reference run (`applied_rainfall.csv`)
and builds a piecewise-constant schedule from it, binding the CSV hash beside
the case's original rainfall provenance, which is left untouched. Conductivity
is the deterministic 0.00025 mm/s already used by `plot1_parameters` and is
asserted, so the run matches the reference's `deterministic` distribution.

What this is not: no event completion, terminal deposition, dry reset, restart
or checkpoint (the final state is wet and explicitly diagnostic; restart is
refused, not emulated), no wind, no splash, no GPU (backend numpy only), no
change to the normal evolving-terrain runners.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import math
import os
import re
import resource
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from maple_syrup.case_import import (
    CLASS_IDS,
    Plot1ImportError,
    VerifiedPlot1Case,
    _refuse_output,
    _sha256_file,
    _write_new_json,
    verify_plot1_case,
)
from maple_syrup.characteristic_transport import (
    DEFAULT_COURANT_MAX as CHARACTERISTIC_COURANT_MAX,
)
from maple_syrup.characteristic_transport import DEFAULT_N_BINS, MAX_N_BINS
from maple_syrup.column_experiment import _Clock, _source_digests, _syrup_provenance
from maple_syrup.conservation import volume_roundoff_bound_m3
from maple_syrup.dependency import MapleDependencyError
from maple_syrup.infiltration import InfiltrationError
from maple_syrup.provenance import environment_record
from maple_syrup.rainfall import RainfallError, RainfallProvenance, RainfallSchedule
from maple_syrup.routing import IMPLEMENTATIONS, RoutingError, RoutingGraphError
from maple_syrup.sediment_bed import ADAPTER_NAME, BedIntegrationError
from maple_syrup.sediment_event import (
    TRANSPORT_IMPLEMENTATIONS,
    TRANSPORT_SCHEMES,
    SedimentEventControl,
    SedimentEventError,
    SedimentEventResult,
    SedimentEventState,
    evolve_sediment_event,
    morphology_summary,
    water_channel_index,
)
from maple_syrup.sediment_experiment import (
    RESOLVED_CONVENTIONS,
    _ensure_absent,
    _jsonable,
    phase_summary,
    prepare_verified_sediment_case,
)
from maple_syrup.sediment_physics import SedimentPhysicsError
from maple_syrup.sediment_transport import TransportError
from maple_syrup.storm import HYDROGRAPH_COLUMNS, StormControl, StormError

__all__ = [
    "APPLIED_RAINFALL_HEADER",
    "FINAL_NAME",
    "FORCING_NAME",
    "HYDROGRAPH_CSV",
    "HYDROGRAPH_NPZ",
    "SUMMARY_NAME",
    "BenchmarkRun",
    "frozen_control",
    "frozen_geometry_report",
    "load_applied_rainfall",
    "main",
    "run_frozen_event",
    "run_plot1_matched_benchmark",
    "validate_frozen_end_state",
]

SUMMARY_SCHEMA = "maple_syrup.plot1_matched_benchmark.v1"
SUMMARY_NAME = "benchmark_summary.json"
FINAL_NAME = "final_state.npz"
FORCING_NAME = "forcing.npz"
HYDROGRAPH_CSV = "hydrograph.csv"
HYDROGRAPH_NPZ = "hydrograph.npz"
APPLIED_RAINFALL_HEADER = ("start_s", "end_s", "logged_applied_rain_mm_h")
# Deterministic Plot 1 conductivity: XML final_infiltration_rate_mean type_1 (mm/s).
EXPECTED_KSAT_MM_S = 0.00025
_EPS = float(np.finfo(np.float64).eps)
_MAX_APPLIED_ROWS = 10_000_000


@dataclass(frozen=True)
class BenchmarkRun:
    summary: dict[str, Any]
    verified: VerifiedPlot1Case
    state: SedimentEventState  # final WET state on the frozen graph; diagnostic, not a handoff
    result: SedimentEventResult
    hydrograph: np.ndarray
    sediment_hydrograph: np.ndarray


def _positive(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not (math.isfinite(value) and value > 0.0):
        raise SedimentEventError(f"{name} must be finite and > 0, got {value!r}")
    return float(value)


# --- applied rainfall override ------------------------------------------------------------------------
def load_applied_rainfall(path: str | Path) -> tuple[RainfallSchedule, dict[str, Any]]:
    """Piecewise-constant schedule from the reference run's logged per-step
    applied rainfall (`audit_mahleran.py` CSV: start_s, end_s, mm/h).

    Refused: wrong header, fewer than one row, non-finite values, a first
    interval not starting at 0 s, non-positive interval lengths, any gap or
    overlap between consecutive intervals, negative intensity. Consecutive
    intervals with an identical intensity are merged into one schedule
    piece (exact depth preserved); the schedule is zero after the last
    interval. The file hash, size, row count and the printing precision of
    the source log (0.01 mm/h) are recorded in the provenance."""
    path = Path(path)
    data = path.read_bytes()
    text = data.decode("ascii")
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines or tuple(h.strip() for h in lines[0].split(",")) != APPLIED_RAINFALL_HEADER:
        raise RainfallError(f"{path}: expected header {','.join(APPLIED_RAINFALL_HEADER)}")
    if len(lines) - 1 > _MAX_APPLIED_ROWS:
        raise RainfallError(f"{path}: more than {_MAX_APPLIED_ROWS} rows; refusing an unbounded forcing file")
    try:
        rows = np.array([[float(v) for v in line.split(",")] for line in lines[1:]], dtype=np.float64)
    except ValueError as exc:
        raise RainfallError(f"{path}: non-numeric applied rainfall row: {exc}") from None
    if rows.ndim != 2 or rows.shape[0] < 1 or rows.shape[1] != 3:
        raise RainfallError(f"{path}: need rows of exactly three numeric columns, got shape {rows.shape}")
    if not np.all(np.isfinite(rows)):
        raise RainfallError(f"{path}: applied rainfall contains non-finite values")
    start, end, intensity = rows[:, 0], rows[:, 1], rows[:, 2]
    if start[0] != 0.0:
        raise RainfallError(f"{path}: the first applied interval must start at 0 s, got {start[0]!r}")
    if np.any(end <= start):
        raise RainfallError(f"{path}: every applied interval must have end_s > start_s")
    if rows.shape[0] > 1 and np.any(start[1:] != end[:-1]):
        bad = int(np.flatnonzero(start[1:] != end[:-1])[0]) + 1
        raise RainfallError(f"{path}: applied intervals must be contiguous (gap or overlap at row {bad + 1})")
    if np.any(intensity < 0.0):
        raise RainfallError(f"{path}: negative applied intensity")
    keep = np.concatenate(([True], intensity[1:] != intensity[:-1]))
    edges = np.concatenate((start[keep], [end[-1]]))
    pieces = intensity[keep]
    sha = hashlib.sha256(data).hexdigest()
    provenance = RainfallProvenance(
        kind="mahleran_applied_rainfall_csv",
        convention="per-step APPLIED rainfall logged by the reference program at iteration start (Set_rain_xml.f90 "
                   "record switch after routing: records apply one step late, the first record one extra second); "
                   "interval [start_s, end_s] at the logged intensity; exact interval integration; equal consecutive "
                   "intensities merged; intensities carry the log's 0.01 mm/h printing precision; zero after the "
                   "last interval",
        path=str(path.resolve()), sha256=sha, size_bytes=len(data), n_records=int(rows.shape[0]),
    )
    schedule = RainfallSchedule(edges_s=edges, intensity_mm_per_h=pieces, provenance=provenance)
    record = {
        "path": str(path.resolve()), "sha256": sha, "size_bytes": len(data),
        "n_rows": int(rows.shape[0]), "n_pieces": int(pieces.size),
        "interval_lengths_s": sorted({float(v) for v in np.unique(end - start)}),
        "start_s": float(start[0]), "end_s": float(end[-1]),
        "total_depth_mm": schedule.total_depth_m() * 1.0e3,
        "max_intensity_mm_h": float(intensity.max()), "convention": provenance.convention,
    }
    return schedule, record


# --- frozen-geometry mode -----------------------------------------------------------------------------
def frozen_control(storm: StormControl | None = None, **kwargs: Any) -> SedimentEventControl:
    """`SedimentEventControl` with BOTH commit flags off (frozen hydraulic
    geometry). Any `commit` / `force_final_commit` keyword is refused."""
    for name in ("commit", "force_final_commit"):
        if name in kwargs:
            raise SedimentEventError(f"frozen_control sets {name}=False itself; do not pass it")
    return SedimentEventControl(storm=storm or StormControl(), commit=False, force_final_commit=False,
                                **kwargs).validated()


def frozen_geometry_report(state0: SedimentEventState, final: SedimentEventState,
                           result: SedimentEventResult | None = None) -> dict[str, Any]:
    """Check that NOTHING hydraulic moved: same graph object, equal aspect /
    slope / receiver / outlet / active / friction / conveyance, equal
    committed elevation, unchanged commit count and clock, network and grid
    bound to that graph, and (if a result is given) zero commits. Raises
    `SedimentEventError` on any violation; returns the check record."""
    from maple.core.backend import to_host
    from maple.core.types.topographic_commit import committed_elevation_m

    g0, g1 = state0.graph, final.graph
    checks = {
        "graph_same_object": g1 is g0,
        "network_same_object": final.network is state0.network,
        "grid_same_object": final.grid is state0.grid,
        "aspect_equal": bool(np.array_equal(g1.aspect, g0.aspect)),
        "slope_equal": bool(np.array_equal(g1.slope, g0.slope)),
        "receiver_equal": bool(np.array_equal(g1.receiver, g0.receiver)),
        "outlet_equal": bool(np.array_equal(g1.outlet, g0.outlet)),
        "active_equal": bool(np.array_equal(g1.active, g0.active)),
        "friction_equal": bool(np.array_equal(g1.friction_factor, g0.friction_factor)),
        "conveyance_equal": bool(np.array_equal(to_host(g1.conveyance), to_host(g0.conveyance))),
        "graph_input_sha256_equal": g1.input_sha256 == g0.input_sha256,
        "network_bound_to_graph": final.network.graph_input_sha256 == g1.input_sha256,
        "grid_slope_equal": bool(np.array_equal(to_host(final.grid.slope), g1.slope)),
        "committed_elevation_equal": bool(np.array_equal(
            to_host(committed_elevation_m(final.bed.committed_topography)),
            to_host(committed_elevation_m(state0.bed.committed_topography)))),
        "committed_offset_equal": bool(np.array_equal(
            to_host(final.bed.committed_topography.elevation_offset_m),
            to_host(state0.bed.committed_topography.elevation_offset_m))),
        "commit_count_unchanged": final.bed.committed_topography.commit_count
        == state0.bed.committed_topography.commit_count,
        "last_commit_time_unchanged": final.bed.committed_topography.last_commit_time_s
        == state0.bed.committed_topography.last_commit_time_s,
        "terrain_reference_same_object": final.terrain is state0.terrain,
    }
    if result is not None:
        checks["n_commits_zero"] = result.n_commits == 0 and result.n_forced_commits == 0
        checks["n_graph_changes_zero"] = result.n_graph_changes == 0 and result.rerouted_cells_total == 0
        checks["commit_log_empty"] = len(result.commit_log) == 0
    failed = sorted(name for name, ok in checks.items() if not ok)
    if failed:
        raise SedimentEventError(f"frozen hydraulic geometry violated: {failed}")
    return {"frozen": True, "checks": checks,
            "rule": "commit=False and force_final_commit=False: MAPLE commit triggers never evaluated; the graph, "
                    "network, grid and committed topography built at t = 0 are the objects returned at the end"}


def validate_frozen_end_state(final: SedimentEventState, context: Any) -> dict[str, Any]:
    """Actual MAPLE validators on the wet end state: active-layer / voxel
    partition, availability closure, ledger structure, water state and the
    ledger-to-physical reconciliation against the UNCHANGED committed
    inventory (no commit ran, so the committed baseline is the initial
    inventory and the pending ledger must equal the whole-window physical
    change per cell and class within MAPLE's own tolerance). Raises on any
    failure (MAPLE `ValueError`); returns the record."""
    from maple.coupling.sediment_ledger.validation import validate_ledger_state
    from maple.surface.active_layer.validation import check_active_layer_voxel_partition
    from maple.surface.availability.validation import check_sediment_availability_state
    from maple.surface.topographic_commit.commit import (
        _reconcile_ledger_with_physical_state,
        ledger_is_empty,
        resolve_reconciliation_reference_inventory_kg,
        validate_reconciliation_context,
    )
    from maple.water import validate_water_state

    b = final.bed
    g = context.geometry
    shape = (g.ny, g.nx, len(context.grain_classes.classes))
    check_active_layer_voxel_partition(b.active_layer, b.voxel_column, g, context.mass_resolution_kg)
    check_sediment_availability_state(b.sediment_availability, b.active_layer, context.mass_resolution_kg)
    if validate_ledger_state(b.ledger) != shape:
        raise SedimentEventError("ledger shape differs from the case geometry")
    validate_water_state(b.water, *shape)
    validate_reconciliation_context(b.ledger, b.voxel_column, b.active_layer, g, committed_state=b.committed_topography,
                                    reconciliation_baseline=None, current_time_s=final.t_s,
                                    operation="SYRUP frozen-geometry benchmark end state", ledger_shape=shape)
    reference, label = resolve_reconciliation_reference_inventory_kg(b.committed_topography, None)
    _reconcile_ledger_with_physical_state(b.ledger, reference, b.voxel_column, b.active_layer,
                                          context.mass_resolution_kg, reference_label=label, ledger_shape=shape)
    pending = np.asarray(b.ledger.pending_bed_mass_change_kg)
    return {
        "maple_validators": ["check_active_layer_voxel_partition", "check_sediment_availability_state",
                             "validate_ledger_state", "validate_water_state", "validate_reconciliation_context",
                             ("_reconcile_ledger_with_physical_state (pending ledger == physical change since the "
                              "unchanged committed inventory, per cell and class, MAPLE tolerance)")],
        "ledger_reconciled_against": label,
        "ledger_empty": bool(ledger_is_empty(b.ledger)),
        "pending_bed_mass_change_by_class_kg": pending.sum(axis=(0, 1)).tolist(),
        "pending_bed_mass_abs_kg": float(np.abs(pending).sum()),
        "n_physical_touches_max": int(np.asarray(b.ledger.n_physical_touches).max()),
        "largest_ledger_residual_kg": float(b.ledger.largest_residual_kg),
    }


def run_frozen_event(state: SedimentEventState, context: Any, column: Any, field: Any, schedule: RainfallSchedule,
                     vegetation: Any, sediment: Any, end_s: float, control: SedimentEventControl, *,
                     report_every_s: float, max_report_rows: int = 100_000,
                     require_exchange: bool = True) -> tuple[SedimentEventResult, dict[str, Any]]:
    """Evolve on frozen hydraulic geometry and prove it. `control` must have
    both commit flags off (see `frozen_control`); nothing is overridden
    silently. Returns the Phase 5 result and the freeze / MAPLE end-state
    records. With `require_exchange` the run refuses to pass when MAPLE moved
    no actual sediment (a benchmark of nothing is not a benchmark)."""
    control = control.validated()
    if control.commit or control.force_final_commit:
        raise SedimentEventError("frozen benchmark requires commit=False AND force_final_commit=False "
                                 "(use frozen_control); commits would refresh the hydraulic geometry")
    result = evolve_sediment_event(state, context, column, field, schedule, vegetation, sediment, end_s, control,
                                   report_every_s=report_every_s, max_report_rows=max_report_rows)
    freeze = frozen_geometry_report(state, result.state, result)
    end_state = validate_frozen_end_state(result.state, context)
    pickup = float(np.asarray(result.by_class["actual_pickup"]).sum())
    deposition = float(np.asarray(result.by_class["deposition_actual"]).sum())
    exchange = {"actual_pickup_kg": pickup, "actual_deposition_kg": deposition,
                "export_kg": float(np.asarray(result.by_class["export_actual"]).sum()),
                "nonzero": pickup > 0.0 and deposition > 0.0}
    if require_exchange and not exchange["nonzero"]:
        raise SedimentEventError("frozen benchmark produced no actual MAPLE pickup and deposition; the window "
                                 f"is not a sediment benchmark (pickup {pickup} kg, deposition {deposition} kg)")
    return result, {"frozen_geometry": freeze, "maple_end_state": end_state, "exchange": exchange}


# --- reference-run binding -------------------------------------------------------------------------------
def _read_asc(path: Path) -> np.ndarray:
    return np.loadtxt(path, skiprows=6)


def reference_run_record(reference_dir: Path, expected_ksat_mm_s: float) -> dict[str, Any]:
    """Bind the whole-program MAHLERAN run this benchmark is matched to:
    execution manifest, recomputed output hashes, the deterministic
    conductivity map (interior uniform at the expected value) and the XML
    distribution switch. Read-only."""
    execution = json.loads((reference_dir / "execution.json").read_text())
    if execution.get("status") != "program_completed_pending_output_audit" or execution.get("returncode") != 0 \
            or not execution.get("completion_marker"):
        raise SedimentEventError(f"reference run {reference_dir} did not complete")
    outputs = {}
    for name, entry in execution["outputs"].items():
        if name.startswith("Output/"):
            actual = _sha256_file(reference_dir / name)
            if actual != entry["sha256"]:
                raise SedimentEventError(f"reference output {name} no longer matches its execution manifest")
            outputs[name] = actual
    ksat = _read_asc(reference_dir / "Output" / "ksat_001.asc")[1:-1, 1:-1]
    if ksat.shape != (60, 20) or not np.all(ksat == expected_ksat_mm_s):
        raise SedimentEventError(f"reference conductivity map is not uniformly {expected_ksat_mm_s} mm/s")
    xml = (reference_dir / "mahleran_input.xml").read_text(encoding="latin-1")
    match = re.search(r'<finalInfiltrationRateDistribution value="([^"]*)"', xml)
    if match is None or match.group(1) != "deterministic":
        raise SedimentEventError("reference XML does not select the deterministic conductivity distribution")
    return {
        "reference_dir": str(reference_dir), "executable_sha256": execution["executable_sha256"],
        "build_manifest": execution.get("build_manifest"), "wall_s": execution.get("wall_s"),
        "output_sha256": outputs, "xml_sha256": _sha256_file(reference_dir / "mahleran_input.xml"),
        "conductivity_mm_s": expected_ksat_mm_s, "conductivity_distribution": "deterministic",
        "configuration": "runtype=event, update_topography=n, method 5, direct dry-cell splash disabled by "
                         "benchmarks/phase7/no_splash.patch (see docs/phase7/mahleran_reference_run.md)",
    }


# --- Plot 1 runner -----------------------------------------------------------------------------------------
def run_plot1_matched_benchmark(
    case_dir: str | Path,
    output_dir: str | Path,
    *,
    applied_rainfall_csv: str | Path,
    reference_run_dir: str | Path | None = None,
    max_dt_s: float = 1.0,
    end_s: float | None = None,
    implementation: str = "numba",
    backend: str = "numpy",
    report_every_s: float = 1.0,
    min_dt_s: float = 1.0 / 1024.0,
    max_retries: int = 10,
    max_steps: int = 10_000_000,
    max_report_rows: int = 100_000,
    sediment_courant_max: float = CHARACTERISTIC_COURANT_MAX,
    max_transport_substeps: int = 64,
    transport_scheme: str = "characteristic",
    phase_bins: int = DEFAULT_N_BINS,
    transport_implementation: str = "auto",
    expected_ksat_mm_s: float = EXPECTED_KSAT_MM_S,
    mahleran_root: str | Path | None = None,
    expected_maple_root: str | Path | None = None,
    allow_maple_source_change: bool = False,
) -> BenchmarkRun:
    from maple.core.backend import (
        read_transfer_counters,
        synchronize,
        to_host,
        to_host_tree,
    )
    from maple.core.parameters import config_to_dict
    from maple.core.types.topographic_commit import committed_elevation_m
    from maple.surface.voxels._numerics import summation_error_bound_kg
    from maple.water import validate_water_state

    from maple_syrup import routing_numba

    max_dt = _positive(max_dt_s, "max_dt_s")
    cadence = _positive(report_every_s, "report_every_s")
    if end_s is not None:
        _positive(end_s, "end_s")
    _positive(expected_ksat_mm_s, "expected_ksat_mm_s")
    storm_control = StormControl(max_dt_s=max_dt_s, min_dt_s=min_dt_s, max_retries=max_retries, max_steps=max_steps,
                                 implementation=implementation).validated()
    control = frozen_control(storm_control, sediment_courant_max=sediment_courant_max,
                             max_transport_substeps=max_transport_substeps, transport_scheme=transport_scheme,
                             phase_bins=phase_bins, transport_implementation=transport_implementation)
    if isinstance(max_report_rows, bool) or not isinstance(max_report_rows, int) or max_report_rows < 1:
        raise SedimentEventError(f"max_report_rows must be a positive int, got {max_report_rows!r}")
    if implementation not in IMPLEMENTATIONS:
        raise SedimentEventError(f"implementation must be one of {IMPLEMENTATIONS}, got {implementation!r}")
    if backend != "numpy":
        raise SedimentEventError(f"backend {backend!r} is refused: CPU benchmark only (numba hydraulics are host-only "
                                 "and no GPU event has been executed); there is no hidden fallback")
    needs_numba = implementation == "numba" or (control.characteristic
                                                and control.resolved_transport_implementation() == "numba")
    if needs_numba and not routing_numba.numba_available():
        raise SedimentEventError("implementation 'numba' requested but Numba is not importable; no fallback -- "
                                 "pass --implementation array explicitly")
    output_dir = Path(output_dir).resolve()
    _ensure_absent(output_dir)
    applied_path = Path(applied_rainfall_csv).resolve()
    if not applied_path.is_file():
        raise SedimentEventError(f"applied rainfall CSV {applied_path} does not exist")
    reference_dir = None if reference_run_dir is None else Path(reference_run_dir).resolve()
    if reference_dir is not None and not (reference_dir / "execution.json").is_file():
        raise SedimentEventError(f"reference run {reference_dir} has no execution.json")

    clock = _Clock()
    rss_start_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    syrup_provenance = _syrup_provenance()
    verified = verify_plot1_case(case_dir, mahleran_root=mahleran_root, expected_maple_root=expected_maple_root,
                                 allow_maple_source_change=allow_maple_source_change)
    dependency = verified.maple_dependency
    recipe_record = verified.report["recipe"]
    forbidden = {
        "MAPLE source": dependency.source_root,
        "MAPLE package": dependency.package_dir,
        "MAHLERAN": Path(mahleran_root or recipe_record["mahleran_root"]).resolve(),
        "recorded MAHLERAN": Path(recipe_record["mahleran_root"]).resolve(),
        "recipe": Path(recipe_record["recipe_path"]).resolve().parent,
        "case": Path(case_dir).resolve(),
        "SYRUP package": Path(__file__).resolve().parent,
        "applied rainfall": applied_path.parent,
        "reference run": reference_dir,
    }
    _refuse_output(output_dir, forbidden)
    digests_before = {
        "maple_syrup": syrup_provenance["package_source_digest"]["digest_sha256"],
        "maple": verified.maple_provenance["package_source_digest"]["digest_sha256"],
    }
    # Forcing override and its provenance are bound BEFORE any expensive setup.
    applied_schedule, applied_record = load_applied_rainfall(applied_path)
    reference = None if reference_dir is None else reference_run_record(reference_dir, expected_ksat_mm_s)

    prepared = prepare_verified_sediment_case(verified, backend=backend, mahleran_root=mahleran_root, end_s=end_s,
                                              control=control)
    area, case, column, context = prepared["area"], prepared["case"], prepared["column"], prepared["context"]
    end, field, g, graph, host = prepared["end"], prepared["field"], prepared["g"], prepared["graph"], prepared["host"]
    n_classes, nx, ny = prepared["n_classes"], prepared["nx"], prepared["ny"]
    parameter_record, resolved, original_schedule = prepared["parameter_record"], prepared["resolved"], prepared["schedule"]
    sediment, sediment_record, settings = prepared["sediment"], prepared["sediment_record"], prepared["settings"]
    soil0, state0, vegetation = prepared["soil0"], prepared["state0"], prepared["vegetation"]
    xml_path, xml_sha_before, xp = prepared["xml_path"], prepared["xml_sha_before"], prepared["xp"]
    if applied_schedule.end_s < end:
        raise SedimentEventError(f"applied rainfall ends at {applied_schedule.end_s} s before the window end {end} s; "
                                 "the legacy log must cover the whole compared window")
    # Deterministic conductivity: the ONLY value this benchmark is matched to.
    ksat = np.asarray(host["ksat_m_per_s"], dtype=np.float64)
    if not np.all(ksat == expected_ksat_mm_s * 1.0e-3):
        raise SedimentEventError(f"case conductivity is not uniformly {expected_ksat_mm_s} mm/s; the matched "
                                 "reference uses the deterministic XML mean")
    initial_elevation = to_host(committed_elevation_m(state0.bed.committed_topography)).copy()
    initial_active = to_host(state0.bed.active_layer.mass_kg).copy()
    initial_available = to_host(state0.bed.sediment_availability.available_mass_kg).copy()
    initial_bed_cells = to_host(state0.bed.voxel_column.mass_kg.sum(axis=2) + state0.bed.active_layer.mass_kg).copy()

    override = {
        "case_water_coupling": _jsonable(dataclasses.asdict(case.config.water_coupling)),
        "run_depth_update_rule": context.depth_update_rule,
        "adapter_name": ADAPTER_NAME,
        "detachment_integration": "rate_times_dt",
        "commit_path": "none: commit=False and force_final_commit=False (frozen hydraulic geometry); MAPLE commit "
                       "triggers never evaluated; pending ledger accumulates for the whole window",
        "forcing": "reference run's logged applied rainfall replaces the case schedule for this run only; the case "
                   "binding, staged rainfall file and its hash are unchanged",
    }
    resolved_config = _jsonable({
        "schema": SUMMARY_SCHEMA,
        "maple_config": config_to_dict(case.config),
        "maple_case_identity_sha256": verified.binding["maple_case_identity_sha256"],
        "syrup": {
            "override": override,
            "sediment_parameters": sediment_record["parameters"],
            "conventions": RESOLVED_CONVENTIONS,
            "storm_control": dataclasses.asdict(storm_control),
            "transport": control.transport_record(),
            "sediment_courant_max": control.sediment_courant_max,
            "max_transport_substeps": control.max_transport_substeps,
            "commit": control.commit, "force_final_commit": control.force_final_commit,
            "end_s": end, "report_every_s": cadence, "implementation": implementation, "backend": backend,
            "column_parameters": parameter_record,
            "conductivity_mm_s": expected_ksat_mm_s,
            "graph_input_sha256": graph.input_sha256,
            "original_rainfall_sha256": original_schedule.provenance.sha256,
            "applied_rainfall_sha256": applied_record["sha256"],
        },
    })
    resolved_config_sha256 = hashlib.sha256(json.dumps(resolved_config, sort_keys=True).encode("utf-8")).hexdigest()
    setup = clock.lap()

    # --- frozen-geometry event ---------------------------------------------------------------------------
    counters_before = read_transfer_counters()
    result, checks = run_frozen_event(state0, context, column, field, applied_schedule, vegetation, sediment, end,
                                      control, report_every_s=cadence, max_report_rows=max_report_rows)
    synchronize(xp)
    loop = clock.lap()
    transfers = dataclasses.asdict(read_transfer_counters().delta(counters_before))
    rss_end_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    # --- host reduction ------------------------------------------------------------------------------------
    final = result.state
    storm_final, bed_final = final.storm, final.bed
    names = ["surface_initial", "soil_initial", "rain", "intake", "saturation_return", "drainage",
             "surface_final", "soil_final"]
    grids = [state0.storm.depth_m, soil0, result.cumulative_rain_m, result.cumulative_intake_m,
             result.cumulative_saturation_return_m, result.cumulative_drainage_m, storm_final.depth_m,
             storm_final.soil_water_m]
    scalars = to_host(xp.stack([xp.asarray(a.sum(), dtype=np.float64) for a in grids] + [
        xp.asarray(v, dtype=np.float64) for v in (
            result.cumulative_export_m3, result.peak_depth_m.max(), result.peak_velocity_m_s.max(),
            storm_final.depth_m.max(), (storm_final.depth_m > 0.0).sum(),
            result.peak_outlet_discharge_m3_s, result.time_of_peak_outlet_s,
            result.max_routing_cell_balance_residual_m, result.max_constitutive_residual_m,
            result.peak_mobile_kg, result.time_of_peak_mobile_s, result.peak_export_rate_kg_s,
            result.time_of_peak_export_s, result.max_sediment_courant, result.max_decay_exponent,
            result.max_transport_cell_residual_kg, result.numerical_residual_scalar_abs_kg,
            result.max_courant_old, result.max_courant_new,
        )
    ]))
    sums = dict(zip(names, scalars[: len(names)].tolist()))
    (export_m3, peak_depth, peak_velocity, final_max_depth, ponded_cells, true_peak_q, true_peak_t, max_balance,
     max_constitutive, peak_mobile, peak_mobile_t, peak_export_rate, peak_export_t, max_sed_courant, max_decay,
     max_cell_residual, residual_scalar, cr_old, cr_new) = scalars[len(names):].tolist()
    by_class = {k: to_host(v).astype(np.float64).copy() for k, v in result.by_class.items()}
    regime = {k: int(to_host(v)) for k, v in result.regime_cell_steps.items()}
    hydrograph = to_host(result.hydrograph).copy()
    sediment_hydrograph = to_host(result.sediment_hydrograph).copy()
    closure = result.closure()
    host_bed = to_host_tree(bed_final)
    final_elevation = to_host(committed_elevation_m(bed_final.committed_topography)).copy()
    grids_out = {
        "voxel_mass_kg": np.asarray(host_bed.voxel_column.mass_kg),
        "active_mass_kg": np.asarray(host_bed.active_layer.mass_kg),
        "available_mass_kg": np.asarray(host_bed.sediment_availability.available_mass_kg),
        "bound_mass_kg": np.asarray(host_bed.sediment_availability.bound_mass_kg),
        "initial_active_mass_kg": initial_active,
        "initial_available_mass_kg": initial_available,
        "initial_bed_by_cell_class_kg": initial_bed_cells,
        "committed_elevation_m": final_elevation,
        "initial_committed_elevation_m": initial_elevation,
        "pending_bed_mass_change_kg": np.asarray(host_bed.ledger.pending_bed_mass_change_kg),
        "depth_m": np.asarray(host_bed.water.depth_m),
        "mobile_mass_kg": np.asarray(host_bed.water.mobile_mass_by_cell_class_kg),
        "soil_water_m": to_host(storm_final.soil_water_m).copy(),
        "discharge_m2_s": to_host(storm_final.discharge_m2_s).copy(),
        "velocity_m_s": to_host(result.last_velocity_m_s).copy(),
        "sediment_velocity_m_s": to_host(final.sediment_velocity_m_s).copy(),
        "cumulative_pickup_kg": to_host(result.cumulative_pickup_kg).copy(),
        "cumulative_deposition_kg": to_host(result.cumulative_deposition_kg).copy(),
        "cumulative_export_request_kg": to_host(result.cumulative_export_request_kg).copy(),
        "peak_depth_m": to_host(result.peak_depth_m).copy(),
        "peak_velocity_m_s": to_host(result.peak_velocity_m_s).copy(),
        "cumulative_rain_m": to_host(result.cumulative_rain_m).copy(),
        "cumulative_intake_m": to_host(result.cumulative_intake_m).copy(),
        "cumulative_return_m": to_host(result.cumulative_saturation_return_m).copy(),
        "cumulative_drainage_m": to_host(result.cumulative_drainage_m).copy(),
        "aspect": np.asarray(graph.aspect), "slope": np.asarray(graph.slope), "receiver": np.asarray(graph.receiver),
        "outlet": np.asarray(graph.outlet), "conveyance": np.asarray(to_host(graph.conveyance)).reshape(graph.shape),
    }
    grids_out["water_exchange_net_kg"] = grids_out["cumulative_deposition_kg"] - grids_out["cumulative_pickup_kg"]
    grids_out["bed_change_kg"] = to_host(result.bed_change_by_cell_class_kg()).copy()
    if final.phase is not None:
        grids_out["phase_fraction"] = to_host(final.phase.fraction).copy()
        grids_out["phase_position_m"] = to_host(final.phase.position_m).copy()
    for name, value in grids_out.items():
        if value.dtype.kind == "f" and not np.all(np.isfinite(value)):
            raise SedimentEventError(f"non-finite final grid {name}; no output written")
    if not np.all(np.isfinite(hydrograph)) or not np.all(np.isfinite(sediment_hydrograph)):
        raise SedimentEventError("non-finite hydrograph; no output written")
    for name in ("depth_m", "mobile_mass_kg", "soil_water_m", "discharge_m2_s", "active_mass_kg", "voxel_mass_kg"):
        if np.any(grids_out[name] < 0.0):
            raise SedimentEventError(f"negative final inventory in {name}; no output written")
    validate_water_state(host_bed.water, ny, nx, n_classes)
    phase_record = {**control.transport_record(), **phase_summary(result)}

    # --- water budget with MAPLE's summation coefficient in m3 ---------------------------------------------
    n_steps, n_cells = result.n_accepted_steps, ny * nx
    volumes = {k: v * area for k, v in sums.items()}
    water_residual = (volumes["surface_final"] + volumes["soil_final"] + volumes["drainage"] + export_m3
                      - volumes["surface_initial"] - volumes["soil_initial"] - volumes["rain"])
    water_scale = max(volumes["surface_final"], volumes["soil_final"], volumes["drainage"], export_m3,
                      volumes["surface_initial"], volumes["soil_initial"], volumes["rain"])
    water_tolerance = volume_roundoff_bound_m3(4 * n_cells * max(n_steps, 1) + 7, water_scale)
    rain_expected = applied_schedule.depth_m(0.0, end) * float(host["rainfall_scale"].sum()) * area
    rain_tolerance = volume_roundoff_bound_m3(n_cells * max(n_steps, 1) + 1, max(rain_expected, volumes["rain"]))
    if not abs(water_residual) <= water_tolerance:
        raise SedimentEventError(f"water budget residual {water_residual} m3 exceeds {water_tolerance} m3")
    if not abs(volumes["rain"] - rain_expected) <= rain_tolerance:
        raise SedimentEventError("applied rainfall did not integrate to the override schedule; no output written")
    rows = {name: hydrograph[:, i] for i, name in enumerate(HYDROGRAPH_COLUMNS)}
    row_residual = (rows["surface_storage_m3"] + rows["soil_storage_m3"] + rows["cumulative_drainage_m3"]
                    + rows["cumulative_export_m3"]) - (volumes["surface_initial"] + volumes["soil_initial"]
                                                       + rows["cumulative_rain_m3"])
    if not np.all(np.abs(row_residual) <= water_tolerance):
        raise SedimentEventError("a hydrograph row violates the water balance; no output written")

    # --- sediment closure, MAPLE policy ----------------------------------------------------------------------
    if not closure["closed"] or not closure["request_reconciled"]:
        raise SedimentEventError(f"per-class sediment closure failed: residual {closure['residual_kg']} kg, "
                                 f"tolerance {closure['tolerance_kg']} kg; no output written")
    transfer_scale = max(float(np.max(np.abs(by_class[k]))) for k in
                         ("deposition_requested", "deposition_actual", "export_requested", "export_actual"))
    unmet_tolerance = np.full(n_classes, summation_error_bound_kg(
        4 * n_cells * max(result.n_maple_water_calls, 1), max(transfer_scale, 1.0)))
    if np.any(np.abs(by_class["deposition_unmet"]) > unmet_tolerance) or np.any(
            np.abs(by_class["export_unmet"]) > unmet_tolerance):
        raise SedimentEventError("MAPLE refused deposition or export beyond the accumulated identity bound")
    if np.any(np.abs(by_class["transport_budget_residual"]) > by_class["transport_budget_tolerance"]):
        raise SedimentEventError("accumulated transport budget residual exceeds its declared bound")

    # --- source stability --------------------------------------------------------------------------------------
    if _sha256_file(xml_path) != xml_sha_before:
        raise SedimentEventError("root MAHLERAN XML changed during the run; no output written")
    if _sha256_file(applied_path) != applied_record["sha256"]:
        raise SedimentEventError("applied rainfall CSV changed during the run; no output written")
    digests_after = _source_digests(Path(syrup_provenance["package_dir"]), dependency.package_dir)
    changed = [name for name in digests_before if digests_after[name] != digests_before[name]]
    if changed:
        raise SedimentEventError(f"source changed during the run ({', '.join(changed)}); no output written")

    # --- summary -------------------------------------------------------------------------------------------------
    sediment_rows = {name: sediment_hydrograph[:, i] for i, name in enumerate(result.sediment_columns)}
    morphology = morphology_summary(grids_out["bed_change_kg"], grids_out["water_exchange_net_kg"],
                                    bulk_density_kg_m3=g.bulk_density_kg_m3, cell_area_m2=area)
    original_depth = original_schedule.depth_m(0.0, end)
    applied_depth = applied_schedule.depth_m(0.0, end)
    water_channel = water_channel_index()
    class_ids = list(CLASS_IDS)
    summary = {
        "schema": SUMMARY_SCHEMA,
        "status": (f"fixed_window_complete_wet_diagnostic: t = 0 .. {end} s on FROZEN hydraulic geometry; actual "
                   "MAPLE mass exchange accumulated in the pending ledger; final surface water "
                   f"{volumes['surface_final']} m3, mobile sediment {sum(closure['final_mobile_kg'])} kg RETAINED "
                   "(no completion, terminal deposition, dry reset, restart or wind)"),
        "mode": {
            "frozen_hydraulic_geometry": True,
            "commit": control.commit, "force_final_commit": control.force_final_commit,
            "what_evolves": "MAPLE active layer, voxel refill, availability, mobile pool, export, pending ledger, "
                            "soil water, surface water, sediment velocity memory",
            "what_is_frozen": "committed topography, routing graph (aspect, slope, receivers, outlets, conveyance), "
                              "transport network, physics grid",
            "checks": checks["frozen_geometry"],
            "maple_end_state": checks["maple_end_state"],
            "restart": {"supported": False, "note": "no checkpoint or MAPLE snapshot is written; the final state is a "
                                                    "WET diagnostic on frozen geometry and must not be resumed or "
                                                    "handed to a wind event"},
        },
        "case": {"case_dir": str(verified.case_dir),
                 "maple_case_identity_sha256": verified.binding["maple_case_identity_sha256"],
                 "maple_artifact_sha256": verified.binding["maple_artifact_sha256"],
                 "verification": verified.checks},
        "forcing": {
            "original_case_rainfall": {"path": str(verified.rainfall_path), "sha256": original_schedule.provenance.sha256,
                                       "convention": original_schedule.provenance.convention,
                                       "depth_over_window_mm": original_depth * 1.0e3, "binding": "unchanged"},
            "applied_rainfall_override": {**applied_record, "depth_over_window_mm": applied_depth * 1.0e3,
                                          "used_for": "this run's forcing (case binding untouched)"},
            "depth_difference_mm": (applied_depth - original_depth) * 1.0e3,
            "interpretation": "the legacy program applies records one step late and the first record one extra "
                              "second; matching that log is deliberate and recorded, not an SYRUP forcing change",
            "rain_integral_m3": volumes["rain"], "rain_expected_m3": rain_expected,
            "rain_tolerance_m3": rain_tolerance,
        },
        "conductivity": {"mm_s": expected_ksat_mm_s, "m_s": expected_ksat_mm_s * 1.0e-3,
                         "source": "XML final_infiltration_rate_mean type_1 used deterministically "
                                   "(column_experiment.plot1_parameters); asserted uniform on every cell",
                         "reference_distribution": "deterministic (checked when --reference-run is given)"},
        "reference_run": reference,
        "sources": {
            "mahleran_xml": {"path": str(xml_path), "sha256": xml_sha_before, "stable": True},
            "mahleran_git": verified.report["mahleran"]["git"],
            "syrup_report_sha256": verified.binding["syrup_report_sha256"],
            "syrup_fields_sha256": verified.binding["syrup_fields_sha256"],
        },
        "provenance": {
            "maple_syrup": syrup_provenance, "maple": verified.maple_provenance,
            "maple_matches_import_binding": not verified.checks["maple_source_changed_since_import"],
            "source_stability": {"before": digests_before, "after": digests_after, "stable": True},
            "environment": environment_record(), "numba": routing_numba.numba_versions(),
            "implementation": implementation,
        },
        "resolved_config": resolved_config, "resolved_config_sha256": resolved_config_sha256,
        "domain": {"ny": ny, "nx": nx, "dx_m": g.dx_m, "cell_area_m2": area, "active_cells": graph.n_active,
                   "n_outlets": int(graph.outlet.sum()), "graph_input_sha256": graph.input_sha256,
                   "class_ids": class_ids, "n_voxels": int(grids_out["voxel_mass_kg"].shape[2]),
                   "bulk_density_kg_m3": g.bulk_density_kg_m3,
                   "orientation": "MAPLE row 0 = south; MAHLERAN ASCII rows are north-first with a one-cell ring "
                                  "(crop [1:-1, 1:-1] then reverse rows to compare)"},
        "time": {"start_s": 0.0, "end_s": end, "rainfall_end_s": applied_schedule.end_s,
                 "legacy_stormlength_s": float(settings["stormlength_s"]), "max_dt_s": max_dt,
                 "min_dt_s": storm_control.min_dt_s, "n_accepted_steps": n_steps,
                 "n_rejected_attempts": result.n_rejected_attempts, "rejections_recorded": list(result.rejections),
                 "n_transport_substep_rejections": result.n_transport_rejections,
                 "max_transport_substeps_used": result.max_transport_substeps_used,
                 "dt_min_accepted_s": result.min_accepted_dt_s, "dt_max_accepted_s": result.max_accepted_dt_s,
                 "n_boundaries": int(result.boundaries.size), "report_every_s": cadence,
                 "n_maple_water_calls": result.n_maple_water_calls},
        "parameters": {"column": parameter_record, "sediment": sediment_record},
        "routing": {"method": "MAHLERAN method 5 (Crank-Nicolson, bisection on [0, R], coherent old inflow)",
                    "implementation": implementation, "courant_max": storm_control.courant_max,
                    "max_courant_old": cr_old, "max_courant_new": cr_new,
                    "max_cell_balance_residual_m": max_balance, "max_constitutive_residual_m": max_constitutive},
        "budget": {
            "units": "m3", **{f"{k}_m3": v for k, v in volumes.items()}, "export_m3": export_m3,
            "water_residual_m3": water_residual, "tolerance_m3": water_tolerance,
            "tolerance_rule": "MAPLE summation coefficient (4 n eps) with n = 4 n_cells n_steps + 7 on the largest "
                              "genuine volume operand",
            "identity": "surface_final + soil_final + drainage + export = surface_initial + soil_initial + rain",
            "export_definition": "conservative face-volume ledger of the routing step (NOT a sum of endpoint outlet "
                                 "discharge); the hydrograph also carries the instantaneous outlet discharge",
        },
        "sediment": {
            "units": "kg", "class_ids": class_ids,
            "by_class": {k: v.tolist() for k, v in by_class.items()},
            "totals": {k: float(v.sum()) for k, v in by_class.items()},
            "closure": closure, "unmet_tolerance_by_class_kg": unmet_tolerance.tolist(),
            "numerical_residual_scalar_abs_kg": residual_scalar,
            "max_transport_cell_balance_residual_kg": max_cell_residual,
            "max_sediment_courant": max_sed_courant, "max_decay_exponent_v_r_dt": max_decay,
            "regime_cell_class_steps": regime,
            "transport": phase_record,
            "peak_mobile_kg": peak_mobile, "time_of_peak_mobile_s": peak_mobile_t,
            "final_mobile_kg": sum(closure["final_mobile_kg"]),
            "peak_export_rate_kg_s": peak_export_rate, "time_of_peak_export_s": peak_export_t,
            "sampled_peak_export_rate_kg_s": float(sediment_rows["export_rate_kg_s"].max()),
            "export_rate_note": "export_rate columns are the last accepted step's actual export / dt at each row "
                                "(a step-integrated flux); MAHLERAN's sedtr/seddisch are instantaneous outlet fluxes",
            "net_bed_mass_change_by_class_kg": closure["net_bed_change_kg"],
            "morphology_from_actual_inventory": morphology,
            "elevation_note": "bed mass changed while the hydraulic elevation stayed frozen; the "
                              "elevation-equivalent change is a diagnostic (bulk density 1250 kg/m3), not a terrain "
                              "update; MAPLE's bulk density makes it 2.12 x the legacy z_change per solid mass",
            "ledger_water_channel_index": water_channel,
            "pending_ledger_by_class_kg": checks["maple_end_state"]["pending_bed_mass_change_by_class_kg"],
        },
        "final_state": {"ponded_cells": int(ponded_cells), "max_depth_m": final_max_depth,
                        "peak_depth_m": peak_depth, "peak_velocity_m_s": peak_velocity,
                        "final_outlet_discharge_m3_s": float(rows["outlet_discharge_m3_s"][-1]),
                        "peak_outlet_discharge_m3_s": true_peak_q, "time_of_peak_outlet_discharge_s": true_peak_t,
                        "sampled_peak_outlet_discharge_m3_s": float(rows["outlet_discharge_m3_s"].max()),
                        "sampled_peak_time_s": float(rows["t_s"][int(np.argmax(rows["outlet_discharge_m3_s"]))]),
                        "residual": "wet: surface water, soil water and mobile sediment retained (diagnostic)"},
        "backend": {"backend": resolved.backend.value, "device_id": resolved.device_id,
                    "fingerprint": json.loads(json.dumps(resolved.fingerprint, default=str)),
                    "loop_transfer_counters": transfers,
                    "gpu": "not exercised; unverified; non-numpy backends refused"},
        "performance": {
            "setup_and_verification": setup,
            "first_accepted_step": {"wall_s": result.first_step_wall_s, "cpu_s": result.first_step_cpu_s,
                                    "note": "includes lazy import and JIT compilation for the numba implementation"},
            "remaining_steps": {"wall_s": result.remaining_wall_s, "cpu_s": result.remaining_cpu_s,
                                "n_steps": max(n_steps - 1, 0),
                                "mean_wall_s_per_step": (result.remaining_wall_s / max(n_steps - 1, 1))},
            "commits_wall_s": result.commit_wall_s,
            "step_loop": loop,
            "peak_rss_kib": {"process_start": int(rss_start_kib), "process_end": int(rss_end_kib),
                             "note": "ru_maxrss of this process; includes imports, verification and the case"},
            "note": "one observational run; not a controlled benchmark and not comparable with the whole-program "
                    "MAHLERAN wall time",
        },
        "hydrograph": {"water_columns": list(HYDROGRAPH_COLUMNS), "sediment_columns": list(result.sediment_columns),
                       "n_rows": int(hydrograph.shape[0]), "files": [HYDROGRAPH_CSV, HYDROGRAPH_NPZ],
                       "sediment_prefix_in_files": "sed_"},
        "assumptions": [
            ("wet MAHLERAN laws with the resolved legacy_literal conventions (docs/phase5/physics.md); direct "
             "dry-cell splash absent in both models"),
            ("hydraulic elevation and routing frozen for the whole window; sediment holdings evolve conservatively "
             "through actual MAPLE"),
            "forcing = reference program's logged applied rainfall; conductivity deterministic 0.00025 mm/s",
            "soil-water parameter thickness fixed; no pore water carried by sediment",
        ],
        "limitations": [
            "wet fixed-window diagnostic: no event completion, dry reset, restart, wind or GPU",
            ("MAHLERAN stock outputs are four-significant-digit endpoint samples; comparisons are output-aware, not "
             "conservation ledgers"),
            "sediment magnitude agreement with the reference is measured by the comparator, never assumed",
            "the pending ledger is left uncommitted by design; this state is not a MAPLE handoff",
        ],
    }

    # --- outputs: staging then rename ------------------------------------------------------------------------------
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.parent / f".{output_dir.name}.partial-{os.getpid()}"
    _ensure_absent(staging)
    staging.mkdir()
    try:
        with (staging / FINAL_NAME).open("xb") as handle:
            np.savez(handle, **grids_out)
        with (staging / FORCING_NAME).open("xb") as handle:
            np.savez(handle, edges_s=applied_schedule.edges_s, intensity_mm_per_h=applied_schedule.intensity_mm_per_h,
                     original_edges_s=original_schedule.edges_s,
                     original_intensity_mm_per_h=original_schedule.intensity_mm_per_h)
        interval = {name: np.diff(rows[name], prepend=0.0) for name in (
            "cumulative_rain_m3", "cumulative_intake_m3", "cumulative_saturation_return_m3",
            "cumulative_drainage_m3", "cumulative_export_m3")}
        sed_out = {f"sed_{k}": v for k, v in sediment_rows.items()}
        with (staging / HYDROGRAPH_NPZ).open("xb") as handle:
            np.savez(handle, **rows, **{f"interval_{k[len('cumulative_'):]}": v for k, v in interval.items()},
                     row_water_residual_m3=row_residual, **sed_out)
        with (staging / HYDROGRAPH_CSV).open("x", newline="", encoding="ascii") as handle:
            writer = csv.writer(handle)
            header = list(HYDROGRAPH_COLUMNS) + [f"interval_{k[len('cumulative_'):]}" for k in interval] + [
                "row_water_residual_m3"] + list(sed_out)
            writer.writerow(header)
            for i in range(hydrograph.shape[0]):
                writer.writerow([repr(float(v)) for v in hydrograph[i]]
                                + [repr(float(interval[k][i])) for k in interval] + [repr(float(row_residual[i]))]
                                + [repr(float(sed_out[k][i])) for k in sed_out])
        summary["outputs"] = {name: hashlib.sha256((staging / name).read_bytes()).hexdigest()
                              for name in (FINAL_NAME, FORCING_NAME, HYDROGRAPH_NPZ, HYDROGRAPH_CSV)}
        summary["performance"]["reporting"] = clock.lap()
        _write_new_json(staging / SUMMARY_NAME, summary)
        _ensure_absent(output_dir)
        staging.rename(output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return BenchmarkRun(summary=summary, verified=verified, state=final, result=result, hydrograph=hydrograph,
                        sediment_hydrograph=sediment_hydrograph)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m maple_syrup.benchmark_experiment",
        description="MAPLE-SYRUP Phase 7: matched fixed-terrain wet benchmark of Plot 1 (frozen hydraulic geometry, "
                    "actual MAPLE sediment exchange, reference applied rainfall, deterministic conductivity). "
                    "No completion, dry reset, restart, wind or GPU.")
    parser.add_argument("--case-dir", required=True, help="Verified Phase 2 case, e.g. outputs/plot1")
    parser.add_argument("--output-dir", required=True, help="NEW directory for summary, grids and hydrographs.")
    parser.add_argument("--applied-rainfall", required=True,
                        help="CSV of the reference run's logged applied rainfall (audit_mahleran.py).")
    parser.add_argument("--reference-run", default=None,
                        help="Deterministic-conductivity MAHLERAN run directory to bind (execution.json, Output/).")
    parser.add_argument("--max-dt-s", type=float, default=1.0, help="Largest step (s): 1, 0.5 or 0.25 for the study.")
    parser.add_argument("--end-s", type=float, default=None, help="Window end (s); default the legacy 5400 s.")
    parser.add_argument("--implementation", default="numba", choices=IMPLEMENTATIONS,
                        help="Hydraulic sweep: numba (default, CPU compiled; missing Numba is an error) or array.")
    parser.add_argument("--backend", default="numpy", choices=("numpy", "cupy"),
                        help="Only numpy is accepted; cupy is refused explicitly (no GPU claim).")
    parser.add_argument("--report-every-s", type=float, default=1.0, help="Hydrograph cadence (s); 1 s for matching.")
    parser.add_argument("--min-dt-s", type=float, default=1.0 / 1024.0)
    parser.add_argument("--max-retries", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=10_000_000)
    parser.add_argument("--max-report-rows", type=int, default=100_000)
    parser.add_argument("--sediment-courant-max", type=float, default=CHARACTERISTIC_COURANT_MAX,
                        help="Sediment Courant cap per substep (<= 0.5 for characteristic, <= 1 for upwind).")
    parser.add_argument("--max-transport-substeps", type=int, default=64)
    parser.add_argument("--transport-scheme", default="characteristic", choices=TRANSPORT_SCHEMES,
                        help="Lateral transport: characteristic (Phase 7b phase bins, default) or the Phase 5 upwind "
                             "operator kept as an explicit comparison.")
    parser.add_argument("--phase-bins", type=int, default=DEFAULT_N_BINS,
                        help=f"Position bins per cell/class for the characteristic scheme (1..{MAX_N_BINS}; "
                             f"{DEFAULT_N_BINS} is a candidate default, not an accepted convergence).")
    parser.add_argument("--transport-implementation", default="auto", choices=TRANSPORT_IMPLEMENTATIONS,
                        help="Characteristic kernel: auto (follow --implementation), array or numba; no fallback.")
    parser.add_argument("--expected-ksat-mm-s", type=float, default=EXPECTED_KSAT_MM_S)
    parser.add_argument("--mahleran-root", help="Override the recipe's MAHLERAN root (read-only).")
    parser.add_argument("--expected-maple-root", help="Refuse any other MAPLE source root.")
    parser.add_argument("--allow-maple-source-change", action="store_true",
                        help="Run even if MAPLE's source digest differs from the import's (recorded).")
    args = parser.parse_args(argv)
    try:
        run = run_plot1_matched_benchmark(
            args.case_dir, args.output_dir, applied_rainfall_csv=args.applied_rainfall,
            reference_run_dir=args.reference_run, max_dt_s=args.max_dt_s, end_s=args.end_s,
            implementation=args.implementation, backend=args.backend, report_every_s=args.report_every_s,
            min_dt_s=args.min_dt_s, max_retries=args.max_retries, max_steps=args.max_steps,
            max_report_rows=args.max_report_rows, sediment_courant_max=args.sediment_courant_max,
            max_transport_substeps=args.max_transport_substeps, transport_scheme=args.transport_scheme,
            phase_bins=args.phase_bins, transport_implementation=args.transport_implementation,
            expected_ksat_mm_s=args.expected_ksat_mm_s,
            mahleran_root=args.mahleran_root, expected_maple_root=args.expected_maple_root,
            allow_maple_source_change=args.allow_maple_source_change)
    except MapleDependencyError as exc:
        print(f"MAPLE dependency check failed: {exc}", file=sys.stderr)
        return 2
    except (Plot1ImportError, SedimentEventError, StormError, BedIntegrationError, SedimentPhysicsError,
            TransportError, RoutingError, RoutingGraphError, InfiltrationError, RainfallError, OSError,
            ValueError) as exc:
        print(f"plot1 matched benchmark failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    printed = {k: run.summary[k] for k in ("status", "time", "budget", "final_state", "performance")}
    printed["mode"] = {k: run.summary["mode"][k] for k in ("frozen_hydraulic_geometry", "commit", "force_final_commit")}
    printed["sediment"] = {k: run.summary["sediment"][k] for k in (
        "totals", "closure", "peak_export_rate_kg_s", "time_of_peak_export_s", "net_bed_mass_change_by_class_kg",
        "regime_cell_class_steps")}
    print(json.dumps(_jsonable(printed), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
