"""EXPERIMENTAL RFID_2014 water-only benchmark case: audit the MAHLERAN inputs, build an ACTUAL MAPLE case, and expose the
inputs the shared SYRUP hydrology solvers need. Task rfid_timing. Water only: no sediment transport, no splash, no
evolving terrain, no wind, no evapotranspiration, no dry reset and no restart claim.

What it does (and reuses). MAPLE does all bed work: the recipe becomes a MAPLE `case.yaml` that `maple` compiles and reloads
(`compile_case`, `load_compiled_case`), exactly like `case_import` does for Plot 1, and `case_import.check_compiled_plot1`
(elevation, 0.1 m voxels, 0.002 m active layer, mass/composition, validators) is applied to the RFID audit. Nothing of the
Plot 1 module or its validation is relaxed; only helpers are imported from it. Hydraulic graph building is
`routing.build_routing_graph` with its two OPT-IN policies (masked nodata, pit storage); the strict defaults are untouched.

Geometry (north-first ESRI ASCII 106 x 60, 0.1 m; MAPLE row 0 = south): the one-cell ring and the 104 x 58 interior. The
interior holds 5697 ACTIVE cells (rainfall-scaling map not nodata) and 335 inactive cells whose DEM and scaling are nodata
(-9999). INACTIVE cells hold no hydrological water, never receive rain, are never a receiver (the legacy `topog_attrib.for`
skip of nodata neighbours) and their faces are closed. The audit requires inactive == DEM-nodata exactly.

Placeholder terrain. MAPLE's bed needs a finite elevation in every cell, so the 335 inactive interior cells get the single
constant `min(active elevation)` in the MAPLE bed ONLY (`elevation_bed_placeholder_m`). It is invented, documented, never
used by any hydraulic code (the full original DEM with its nodata sentinel is kept in the sidecar and is what the graph uses).

Strict D4 sinks. 26 active cells have no lower neighbour (no flats). They are kept as terminal STORAGE cells
(`routing.PIT_STORAGE`): water routed into them stays there in the legacy-style/explicit solvers (no overtopping, no
fill/carve). The local-inertial candidate can move water out of a pit through its water-surface gradient: DIFFERENT PHYSICS
by design, not an identity with the others.

Boundary. The only D4 receiver on the ring with a valid elevation is north-first (104, 23) -> (105, 23) (MAPLE: its mirror).
The native rainfall-scaling there is POSITIVE (0.9688) so the native legacy outlet accounting does not flag it as an
export. This benchmark defines every ring cell as an EXPORT receiver (the matched legacy-routine input sets the ring rmask
negative): an explicit, documented control change, identical for all methods. The edge-rule slope of that outlet comes
from the one graph and is passed unchanged to every contender.

Hydrology (documented, sourced values; see cases/rfid/recipe.yaml and docs/rfid/README.md): infiltration model 1
(`fixed_ksat`, Smith-Parlange capacity, linear drainage); K is the exact value the native type-1 setup computed; suction,
drainage, thickness are the INTENDED XML values, NOT the native setup bug (psi overwritten by 0.05, drainage 0, FP32
thickness); forcing is the native applied rate captured from the executed native application (a constant rate, one-second
switch lag included), not the parser's interval-ending reading.

Nothing here was run by its author (file-only tools); Codex records results.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import tempfile
import traceback
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

import maple_syrup
from maple_syrup.case_import import (
    CLASS_IDS,
    N_CLASSES,
    PHASE2_REQUIRED_MAPLE_API,
    Plot1Audit,
    Plot1ImportError,
    Plot1Recipe,
    VerifiedPlot1Case,
    _git_optional_locks_disabled,
    _json_safe,
    _positive,
    _refuse_output,
    _require,
    _require_same_state,
    _sha256_bytes,
    _sha256_file,
    _stage_verbatim,
    _write_new_json,
    _xml_record,
    _xml_value,
    check_compiled_plot1,
    normalize_closure_roundoff,
    parse_mahleran_xml,
    plan_bed,
)
from maple_syrup.dependency import (
    MapleDependencyError,
    check_required_api,
    resolve_maple_dependency,
)
from maple_syrup.provenance import (
    capture_maple_provenance,
    capture_syrup_provenance,
    environment_record,
    source_tree_digest,
)

__all__ = [
    "RfidRecipe",
    "audit_rfid",
    "derive_applied_forcing",
    "generate_rfid_case",
    "load_rfid_recipe",
    "read_legacy_grid",
    "rfid_inputs",
    "rfid_parameters",
    "rfid_routing_graph",
    "rfid_schedule",
    "verify_rfid_case",
]

RECIPE_SCHEMA = "maple_syrup.rfid_recipe.v1"
REPORT_SCHEMA = "maple_syrup.rfid_import_report.v1"
BINDING_SCHEMA = "maple_syrup.rfid_binding.v1"
FORCING_SCHEMA = "maple_syrup.rfid_applied_forcing.v1"
STAGED_SOURCE_DIR = "source/mahleran_input_rfid"
DERIVED_CODES_PATH = "source/syrup_derived/composition_codes.npy"
DERIVED_DEM_PATH = "source/syrup_derived/elevation_bed_placeholder.npy"
SIDECAR_DIR = "syrup"
REPORT_NAME = "rfid_import_report.json"
FIELDS_NAME = "rfid_fields.npz"
BINDING_NAME = "rfid_binding.json"
FAILURE_NAME = "FAILED.json"
FORCING_NAME = "forcing/applied_forcing.json"
LEGACY_HEADER_LINES = 6
FORCING_KINDS = ("native_applied_capture", "legacy_file")
_EPS = float(np.finfo(np.float64).eps)


# --------------------------------------------------------------------------------------------------------------------
# Legacy raster reader (one value per line is legal ESRI ASCII; case_import.stage_legacy_ascii demands ncols per line)
# --------------------------------------------------------------------------------------------------------------------
def read_legacy_grid(path: str | Path) -> tuple[dict[str, float], np.ndarray]:
    """Header dict and the `(nrows, ncols)` float64 body in FILE (north-first) order. Strict: six `key value` header lines
    (the records Fortran `read_spatial_data` consumes), exactly `nrows * ncols` finite numeric tokens afterwards, ASCII only.
    Anything else (short, long, non-numeric, non-finite) raises `Plot1ImportError`."""
    path = Path(path)
    if not path.is_file():
        raise Plot1ImportError(f"legacy raster not found: {path}")
    try:
        text = path.read_bytes().decode("ascii")
    except UnicodeDecodeError as exc:
        raise Plot1ImportError(f"{path.name}: not ASCII") from exc
    lines = text.splitlines()
    if len(lines) < LEGACY_HEADER_LINES:
        raise Plot1ImportError(f"{path.name}: fewer than {LEGACY_HEADER_LINES} header lines")
    header: dict[str, float] = {}
    for number, line in enumerate(lines[:LEGACY_HEADER_LINES], start=1):
        parts = line.split()
        if len(parts) != 2:
            raise Plot1ImportError(f"{path.name}: header line {number} is not 'key value'")
        key = parts[0].lower()
        if key in header:
            raise Plot1ImportError(f"{path.name}: duplicate header key {key!r}")
        try:
            header[key] = float(parts[1])
        except ValueError as exc:
            raise Plot1ImportError(f"{path.name}: header {key!r} is not numeric") from exc
    try:
        ncols, nrows = int(header["ncols"]), int(header["nrows"])
    except KeyError as exc:
        raise Plot1ImportError(f"{path.name}: header lacks ncols/nrows") from exc
    tokens = " ".join(lines[LEGACY_HEADER_LINES:]).split()
    if len(tokens) != nrows * ncols:
        raise Plot1ImportError(f"{path.name}: {len(tokens)} body values, header declares {nrows} x {ncols}")
    try:
        body = np.array(tokens, dtype=np.float64)
    except ValueError as exc:
        raise Plot1ImportError(f"{path.name}: non-numeric body value") from exc
    if not np.all(np.isfinite(body)):
        raise Plot1ImportError(f"{path.name}: non-finite body value")
    return header, body.reshape(nrows, ncols)


# --------------------------------------------------------------------------------------------------------------------
# Recipe
# --------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class RfidRecipe:
    """Validated authored recipe. `base` is a `Plot1Recipe` built from it so MAPLE-side helpers (`plan_bed`,
    `check_compiled_plot1`, `_audit_geometry`) are reused unchanged; `raw` keeps the validated authored sections."""

    base: Plot1Recipe
    raw: dict[str, Any]
    recipe_path: Path | None
    recipe_sha256: str | None

    @property
    def nodata_value(self) -> float:
        return float(self.raw["legacy_grid"]["nodata_value"])

    @property
    def interior_shape(self) -> tuple[int, int]:
        return self.base.interior_shape

    def as_record(self) -> dict[str, Any]:
        return {"recipe_path": None if self.recipe_path is None else str(self.recipe_path),
                "recipe_sha256": self.recipe_sha256, "authored": self.raw, "plot1_view": self.base.as_record()}


def _section(raw: dict[str, Any], key: str, keys: set[str]) -> dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict) or set(value) != keys:
        raise Plot1ImportError(f"recipe: section {key!r} must be a mapping with exactly the keys {sorted(keys)}")
    return value


def load_rfid_recipe(path: str | Path, *, mahleran_root: str | Path | None = None) -> RfidRecipe:
    """Read and strictly validate the authored RFID recipe (no unknown keys, finite positive numbers)."""
    import yaml

    path = Path(path)
    payload = path.read_bytes()
    raw = yaml.safe_load(payload.decode("utf-8"))
    expected = {"schema", "case_name", "mahleran", "legacy_grid", "grain_maps", "bed", "maple", "placeholder",
                "hydrology", "forcing", "expected_topology"}
    if not isinstance(raw, dict) or set(raw) != expected:
        raise Plot1ImportError(f"recipe must be a mapping with exactly the keys {sorted(expected)}")
    if raw["schema"] != RECIPE_SCHEMA:
        raise Plot1ImportError(f"recipe schema must be {RECIPE_SCHEMA!r}, got {raw['schema']!r}")
    if not isinstance(raw["case_name"], str) or not raw["case_name"]:
        raise Plot1ImportError("recipe: case_name must be a non-empty string")
    mah = _section(raw, "mahleran", {"root", "xml", "input_folder", "expected_version"})
    grid = _section(raw, "legacy_grid", {"nrows", "ncols", "cellsize_m", "nodata_value"})
    grains = _section(raw, "grain_maps", {"closure_roundoff_tolerance"})
    bed = _section(raw, "bed", {"voxel_dz_m", "active_layer_thickness_m", "bulk_density_kg_m3", "minimum_fill_depth_m",
                                "datum_offset_rounding_m", "headroom_m"})
    maple = _section(raw, "maple", {"seed", "sediment_available_fraction", "boundary"})
    place = _section(raw, "placeholder", {"rule"})
    hyd = _section(raw, "hydrology", {"ksat_mm_per_s", "ksat_formula_rtol", "suction_mm", "drainage_parameter",
                                      "soil_thickness_m", "friction_factor", "theta_sat", "theta0", "model"})
    forcing = _section(raw, "forcing", {"kind", "capture_default_path", "capture_expected_sha256"})
    _section(raw, "expected_topology", {"n_active", "n_pit_storage", "n_outlets", "outlet_north_first_0based",
                                        "ring_receiver_north_first_0based"})  # key set validated; values used by the audit
    nrows, ncols = grid["nrows"], grid["ncols"]
    if not all(isinstance(v, int) and not isinstance(v, bool) and v >= 3 for v in (nrows, ncols)):
        raise Plot1ImportError("recipe: legacy_grid.nrows/ncols must be integers >= 3")
    if place["rule"] != "min_active_elevation":
        raise Plot1ImportError("recipe: placeholder.rule must be 'min_active_elevation'")
    if forcing["kind"] not in FORCING_KINDS:
        raise Plot1ImportError(f"recipe: forcing.kind must be one of {FORCING_KINDS}")
    if hyd["model"] != "fixed_ksat":
        raise Plot1ImportError("recipe: hydrology.model must be 'fixed_ksat' (native infiltration_model 1)")
    for key in ("ksat_mm_per_s", "ksat_formula_rtol", "suction_mm", "drainage_parameter", "soil_thickness_m",
                "friction_factor", "theta_sat", "theta0"):
        _positive(hyd[key], f"hydrology.{key}")
    if maple["boundary"] != "prescribed_zero_inflow":
        raise Plot1ImportError("recipe: maple.boundary must be 'prescribed_zero_inflow'")
    fraction = maple["sediment_available_fraction"]
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not 0.0 <= fraction <= 1.0:
        raise Plot1ImportError("recipe: maple.sediment_available_fraction must lie in [0, 1]")
    if not isinstance(maple["seed"], int) or isinstance(maple["seed"], bool):
        raise Plot1ImportError("recipe: maple.seed must be an integer")
    tolerance = _positive(grains["closure_roundoff_tolerance"], "grain_maps.closure_roundoff_tolerance")
    if tolerance > 1.0e-4:
        raise Plot1ImportError("recipe: closure_roundoff_tolerance above 1e-4 would hide material nonclosure")
    nodata = grid["nodata_value"]
    if isinstance(nodata, bool) or not isinstance(nodata, (int, float)) or nodata >= 0:
        raise Plot1ImportError("recipe: legacy_grid.nodata_value must be a negative number")
    base = Plot1Recipe(
        recipe_path=path.resolve(), recipe_sha256=_sha256_bytes(payload), case_name=raw["case_name"],
        mahleran_root=Path(mahleran_root if mahleran_root is not None else mah["root"]).resolve(),
        xml_name=str(mah["xml"]), expected_version=str(mah["expected_version"]),
        expected_input_folder=str(mah["input_folder"]), legacy_nrows=nrows, legacy_ncols=ncols,
        cellsize_m=_positive(grid["cellsize_m"], "legacy_grid.cellsize_m"),
        corrected_maps=tuple(f"phi_{k}" for k in range(1, 7)), closure_tolerance=tolerance,
        pavement_rescaling="none",
        voxel_dz_m=_positive(bed["voxel_dz_m"], "bed.voxel_dz_m"),
        active_layer_thickness_m=_positive(bed["active_layer_thickness_m"], "bed.active_layer_thickness_m"),
        bulk_density_kg_m3=_positive(bed["bulk_density_kg_m3"], "bed.bulk_density_kg_m3"),
        minimum_fill_depth_m=_positive(bed["minimum_fill_depth_m"], "bed.minimum_fill_depth_m"),
        datum_offset_rounding_m=_positive(bed["datum_offset_rounding_m"], "bed.datum_offset_rounding_m"),
        headroom_m=_positive(bed["headroom_m"], "bed.headroom_m"),
        seed=int(maple["seed"]), available_fraction=float(fraction),
    )
    if base.active_layer_thickness_m >= base.minimum_fill_depth_m:
        raise Plot1ImportError("recipe: active_layer_thickness_m must be below minimum_fill_depth_m")
    return RfidRecipe(base=base, raw=raw, recipe_path=path.resolve(), recipe_sha256=_sha256_bytes(payload))


# --------------------------------------------------------------------------------------------------------------------
# Applied-forcing derivation (native capture -> exact piecewise-constant schedule)
# --------------------------------------------------------------------------------------------------------------------
def derive_applied_forcing(capture_text: str, *, capture_sha256: str, max_pieces: int = 16) -> dict[str, Any]:
    """Compress the per-step applied rate of the executed native application (`syrup_hydro_steps.txt`, columns `iter t_s
    rval_applied_mm_s ...`, iterations 1..n at t = iter * dt) into constant-rate pieces on (t_{k-1}, t_k]. Exact: every
    step of a piece has the identical value; a value that changes more than `max_pieces` times is refused. Returns a JSON
    record `{edges_s, intensity_mm_per_h, rate_mm_s, ...}` (right-continuous: the rate a step STARTING at an edge sees is
    the following piece's, the convention of `rainfall.RainfallSchedule.rate_after_m_per_s`)."""
    lines = [ln for ln in capture_text.splitlines() if ln.strip()]
    if len(lines) < 4 or lines[0].split() != ["SYRUP_HYDRO_CAPTURE_V1", "steps"] or lines[-1].split()[0] != "SYRUP_HYDRO_CAPTURE_COMPLETE":
        raise Plot1ImportError("applied-forcing capture is not a complete SYRUP_HYDRO_CAPTURE_V1 steps file")
    columns = lines[1].split()
    if columns[:4] != ["columns", "iter", "t_s", "rval_applied_mm_s"]:
        raise Plot1ImportError("applied-forcing capture has unexpected columns")
    rows = []
    for line in lines[2:-1]:
        tokens = line.split()
        try:
            rows.append((int(tokens[0]), float(tokens[1]), float(tokens[2])))
        except (ValueError, IndexError) as exc:
            raise Plot1ImportError("applied-forcing capture holds a malformed row") from exc
    if not rows or [r[0] for r in rows] != list(range(1, len(rows) + 1)):
        raise Plot1ImportError("applied-forcing capture iterations are not 1..n")
    t = np.array([r[1] for r in rows])
    rate = np.array([r[2] for r in rows])
    if not (np.all(np.isfinite(rate)) and np.all(rate >= 0.0)):
        raise Plot1ImportError("applied-forcing capture holds a negative or non-finite rate")
    dt = float(t[0])
    if not (dt > 0.0 and np.array_equal(t, dt * np.arange(1, len(rows) + 1))):
        raise Plot1ImportError("applied-forcing capture times are not iter * dt")
    change = np.flatnonzero(rate[1:] != rate[:-1]) + 1
    starts = np.concatenate(([0], change))
    if starts.size > max_pieces:
        raise Plot1ImportError(f"applied forcing changes {starts.size} times; refusing to compress (max {max_pieces})")
    ends = np.concatenate((change, [len(rows)]))
    edges = [0.0] + [float(t[e - 1]) for e in ends]
    values = [float(rate[s]) for s in starts]
    if not any(v > 0.0 for v in values):
        raise Plot1ImportError("applied forcing is identically zero")
    return {
        "schema": FORCING_SCHEMA, "kind": "native_applied_capture", "capture_sha256": capture_sha256,
        "dt_s": dt, "n_steps": len(rows), "edges_s": edges, "rate_mm_s": values,
        "intensity_mm_per_h": [v * 3600.0 for v in values],
        "total_depth_mm": float(sum(v * (edges[i + 1] - edges[i]) for i, v in enumerate(values))),
        "convention": "rate applied in step k (interval (t_{k-1}, t_k]) as read before infilt; native one-second "
                      "switching lag included; right-continuous schedule edges",
    }


def _schedule_from_record(record: dict[str, Any], sha256: str):
    from maple_syrup.rainfall import RainfallProvenance, RainfallSchedule

    return RainfallSchedule(
        edges_s=record["edges_s"], intensity_mm_per_h=record["intensity_mm_per_h"],
        provenance=RainfallProvenance(kind="native_applied_capture", convention=record["convention"], sha256=sha256,
                                      n_records=len(record["rate_mm_s"])))


# --------------------------------------------------------------------------------------------------------------------
# Audit
# --------------------------------------------------------------------------------------------------------------------
def _to_maple(nf: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(nf[::-1])


def _single_value(values: np.ndarray, label: str, expected: float | None = None) -> float:
    unique = np.unique(values)
    if unique.size != 1:
        raise Plot1ImportError(f"{label}: this fixture needs ONE value over the active cells, found {unique.size}")
    value = float(unique[0])
    if expected is not None and value != expected:
        raise Plot1ImportError(f"{label}: {value!r} != the recipe's {expected!r}")
    return value


def _ring_mask(shape: tuple[int, int]) -> np.ndarray:
    ring = np.ones(shape, dtype=np.bool_)
    ring[1:-1, 1:-1] = False
    return ring


def audit_rfid(recipe: RfidRecipe, case_dir: str | Path, *, forcing_record: dict[str, Any] | None = None) -> Plot1Audit:
    """Stage, read and audit RFID_2014 into `case_dir` (created empty by the caller). Reads only; writes only staged
    verbatim copies under source/. `forcing_record` (from `derive_applied_forcing`) is stored in the report; for the
    `legacy_file` kind it may be None. Returns a `Plot1Audit` whose `report`/`fields` carry the RFID content, so MAPLE's
    `check_compiled_plot1` is reused unchanged."""
    base, raw = recipe.base, recipe.raw
    case_dir = Path(case_dir)
    root = base.mahleran_root
    xml_path = root / base.xml_name
    if not xml_path.is_file():
        raise Plot1ImportError(f"MAHLERAN XML not found: {xml_path}")
    xml = parse_mahleran_xml(xml_path)

    def value(tag):
        return _xml_value(xml, tag)

    def children(tag):
        return dict(_xml_record(xml, tag)["children"])

    if value("version") != base.expected_version:
        raise Plot1ImportError(f"XML version {value('version')!r} != {base.expected_version!r}")
    input_dir = root / base.expected_input_folder
    if not input_dir.is_dir():
        raise Plot1ImportError(f"input folder not found: {input_dir}")
    settings = {
        "runtype": value("runtype"), "model_type": int(value("model_type")), "rain_type": int(value("rain_type")),
        "infiltration_model": int(value("infiltration_model")),
        "infiltration_parameter_type": int(value("infiltration-parameter_type")),
        "flow_direction": int(value("flow_direction")),
        "native_flow_routing_solution_method": int(value("flow-routing_solution_method")),
        "friction_factor_type": int(value("friction_factor_type")), "time_step_s": float(value("time_step")),
        "stormlength_s": float(value("stormlength")), "mean_rainfall_mm_h": float(value("mean_rainfall")),
        "update_topography": value("update_topography"), "particle_density_g_cm3": float(value("particle_density")),
        "friction_factor_mean_type1": float(children("friction_factor_mean")["type_1"]),
        "wetting_front_suction_mean_type1_mm": float(children("wetting_front_suction_mean")["type_1"]),
        "drainage_parameter_mean_type1": float(children("drainage_parameter_mean")["type_1"]),
        "soil_thickness_type1_m": float(children("soil_thickness")["type_1"]),
        "initial_soil_moisture_mean_type1": float(children("initial_soil_moisture_mean")["type_1"]),
        "number_of_surface_types": int(value("number_of_surface_types")),
        "use_initial_soil_moisture_map": value("use_initial_soil_moisture_map"),
        "use_saturated_soil_moisture_map": value("use_saturated_soil_moisture_map"),
        "use_map_phi": value("use_map_phi"),
    }
    for key, want in (("runtype", "event"), ("flow_direction", 4), ("infiltration_model", 1),
                      ("infiltration_parameter_type", 1), ("rain_type", 2), ("friction_factor_type", 1),
                      ("number_of_surface_types", 1), ("update_topography", "n")):
        if settings[key] != want:
            raise Plot1ImportError(f"RFID fixture expects {key} = {want!r}, the XML has {settings[key]!r}")
    hyd = raw["hydrology"]
    for xml_key, recipe_key in (("friction_factor_mean_type1", "friction_factor"),
                                ("wetting_front_suction_mean_type1_mm", "suction_mm"),
                                ("drainage_parameter_mean_type1", "drainage_parameter"),
                                ("soil_thickness_type1_m", "soil_thickness_m")):
        if settings[xml_key] != float(hyd[recipe_key]):
            raise Plot1ImportError(f"recipe hydrology.{recipe_key} {hyd[recipe_key]!r} != the XML's {settings[xml_key]!r}")

    particle_refs = children("particle_size_map")
    phi_files = [particle_refs.get(f"phi_{k}", "") for k in range(1, 7)]
    if len(set(phi_files)) != N_CLASSES or not all(phi_files):
        raise Plot1ImportError(f"the XML must name six distinct grain maps, got {phi_files}")
    names = {"dem": value("dem"), "scaling": value("rainfall-scaling_map"), "pavement": value("pavement_map"),
             "surface": value("surface-type_map"), "vegetation": value("vegetation-cover_map"),
             "theta_sat": value("saturated_soil-moisture_map"), "theta0": value("initial_soil-moisture_map")}
    rain_name = value("rainfall_data")
    staged: dict[str, dict[str, Any]] = {}
    for name in dict.fromkeys([*names.values(), *phi_files]):
        staged[name] = _stage_verbatim(input_dir / name, case_dir / STAGED_SOURCE_DIR / name)
        staged[name]["relpath"] = f"{STAGED_SOURCE_DIR}/{name}"
    rainfall = _stage_verbatim(input_dir / rain_name, case_dir / SIDECAR_DIR / "rainfall" / rain_name)
    calib_present = (input_dir / "calib.dat").exists()

    grids: dict[str, np.ndarray] = {}
    headers: dict[str, dict[str, float]] = {}
    for name in staged:
        headers[name], grids[name] = read_legacy_grid(input_dir / name)
    ref = headers[names["dem"]]
    for name, header in headers.items():
        for key in ("ncols", "nrows", "cellsize", "xllcorner", "yllcorner", "nodata_value"):
            if header.get(key) != ref.get(key):
                raise Plot1ImportError(f"{name}: header {key}={header.get(key)!r} differs from the DEM's {ref.get(key)!r}")
    if (int(ref["nrows"]), int(ref["ncols"])) != (base.legacy_nrows, base.legacy_ncols):
        raise Plot1ImportError("DEM header shape differs from the recipe")
    if ref["cellsize"] != base.cellsize_m or ref["nodata_value"] != recipe.nodata_value:
        raise Plot1ImportError("DEM cellsize or nodata differs from the recipe")
    nodata = recipe.nodata_value

    dem_sf = _to_maple(grids[names["dem"]])
    scaling_sf = _to_maple(grids[names["scaling"]])
    interior = (slice(1, -1), slice(1, -1))
    active = scaling_sf[interior] >= 0.0
    dem_nodata_interior = dem_sf[interior] == nodata
    if not np.array_equal(~active, dem_nodata_interior):
        raise Plot1ImportError("interior inactive cells (scaling nodata) differ from the interior DEM-nodata cells: "
                               f"{int((~active).sum())} vs {int(dem_nodata_interior.sum())}; unsupported")
    if int(active.sum()) != raw["expected_topology"]["n_active"]:
        raise Plot1ImportError(f"{int(active.sum())} active cells, the recipe expects {raw['expected_topology']['n_active']}")
    scale_active = scaling_sf[interior][active]
    if not np.all(scale_active > 0.0):
        raise Plot1ImportError("an active cell has a non-positive rainfall scale")

    def interior_values(key):
        grid = _to_maple(grids[names[key]])[interior]
        return grid, grid[active]

    pave, pave_active = interior_values("pavement")
    _veg, veg_active = interior_values("vegetation")
    _stype, stype_active = interior_values("surface")
    ssm, ssm_active = interior_values("theta_sat")
    ism, ism_active = interior_values("theta0")
    if np.any(pave_active != 0.0):
        raise Plot1ImportError("pavement cover is not zero on active cells; pavement rescaling is unsupported here")
    _single_value(stype_active, "surface type", 1.0)
    theta_sat = _single_value(ssm_active, "saturated moisture", float(hyd["theta_sat"]))
    theta0 = _single_value(ism_active, "initial moisture", float(hyd["theta0"]))
    _single_value(veg_active, "vegetation cover")
    if theta0 > theta_sat:
        raise Plot1ImportError("initial moisture exceeds saturation")

    # --- hydrology constants (sourced; native-setup bugs are NOT reproduced) ---
    pave_scaled = 0.0001 * float(pave_active.max(initial=0.0))
    k_formula = 0.00585 + 0.000166667 * settings["mean_rainfall_mm_h"] - pave_scaled
    k_recipe = float(hyd["ksat_mm_per_s"])
    if abs(k_formula - k_recipe) > float(hyd["ksat_formula_rtol"]) * abs(k_recipe):
        raise Plot1ImportError(f"recipe K {k_recipe!r} mm/s disagrees with the native type-1 formula {k_formula!r}")

    # --- grains: uniform composition, MAPLE composition table ---
    phi_stack = np.stack([_to_maple(grids[f])[interior] for f in phi_files], axis=-1)
    fill = phi_stack[active][0]
    if not np.all(phi_stack[active] == fill):
        raise Plot1ImportError("grain composition is not uniform over the active cells; this fixture supports one composition")
    phi_stack[~active] = fill  # inactive cells (nodata maps) take the single composition so the MAPLE bed can exist there
    normalized, norm_stats = normalize_closure_roundoff(phi_stack, base.closure_tolerance)
    final = normalized
    table, inverse = np.unique(final.reshape(-1, N_CLASSES), axis=0, return_inverse=True)
    codes = (np.asarray(inverse).reshape(final.shape[:-1]) + 1).astype(np.int32)
    if table.shape[0] != 1 or not np.array_equal(table[codes - 1], final):
        raise Plot1ImportError("composition lookup table does not reproduce the per-cell fractions")

    # --- terrain: placeholder bed elevation, original DEM kept ---
    dem_interior = dem_sf[interior]
    placeholder = float(dem_interior[active].min())
    bed_dem = np.where(active, dem_interior, placeholder)
    bed = plan_bed(bed_dem, base)
    particle_density = float(settings["particle_density_g_cm3"]) * 1000.0

    # --- hydraulic topology with the opt-in policies (the strict builder is untouched) ---
    from maple_syrup.routing import build_routing_graph

    friction = np.full(active.shape, float(hyd["friction_factor"]))
    graph = build_routing_graph(dem_sf, _ring_mask(dem_sf.shape), friction, base.cellsize_m, active_mask=active,
                                nodata_value=nodata, allow_masked_nodata=True, allow_pit_storage=True)
    topo = raw["expected_topology"]
    summary = graph.summary()
    n_pits = int(np.sum(graph.pit_storage))
    outlets = [(int(r), int(c)) for r, c in zip(*np.nonzero(graph.outlet), strict=True)]
    ny = base.interior_shape[0]
    outlets_nf = [(ny - r, c + 1) for r, c in outlets]  # MAPLE interior (r, c) -> full north-first 0-based (row, col)
    if (n_pits != topo["n_pit_storage"] or len(outlets) != topo["n_outlets"]
            or [list(o) for o in outlets_nf] != [list(topo["outlet_north_first_0based"])]):
        raise Plot1ImportError(f"topology differs from the recipe: pits {n_pits}, outlets (north-first, 0-based) {outlets_nf}")
    ring_receiver = list(topo["ring_receiver_north_first_0based"])
    native_ring_scale = float(grids[names["scaling"]][ring_receiver[0], ring_receiver[1]])
    levels = [b - a for a, b in zip(graph.level_bounds[:-1], graph.level_bounds[1:], strict=True)]

    fields = {
        "legacy_full_elevation_m": dem_sf,
        "legacy_full_rainfall_scaling_native": scaling_sf,
        "active": np.ascontiguousarray(active),
        "inactive_interior": np.ascontiguousarray(~active),
        "elevation_bed_placeholder_m": np.ascontiguousarray(bed_dem),
        "elevation_maple_m": bed.elevation_m,
        "rainfall_scaling": np.ascontiguousarray(np.where(active, scaling_sf[interior], 0.0)),
        "saturated_soil_moisture_raw": np.ascontiguousarray(ssm),
        "initial_soil_moisture_raw": np.ascontiguousarray(ism),
        "pavement_percent_raw": np.ascontiguousarray(pave),
        "phi_final": np.ascontiguousarray(final),
        "composition_codes": codes,
        "composition_table": table,
        "expected_mass_by_cell_class_kg": (bed.elevation_m * base.bulk_density_kg_m3 * base.cellsize_m ** 2)[..., None] * final,
        "graph_aspect": np.ascontiguousarray(graph.aspect),
        "graph_slope": np.ascontiguousarray(graph.slope),
        "graph_pit_storage": np.ascontiguousarray(graph.pit_storage),
        "graph_outlet": np.ascontiguousarray(graph.outlet),
    }
    report = {
        "schema": REPORT_SCHEMA, "maple_syrup_version": maple_syrup.__version__,
        "recipe": recipe.as_record(),
        "mahleran": {"root": str(root), "xml_path": str(xml_path.resolve()), "xml_sha256": _sha256_file(xml_path),
                     "version": value("version"), "input_folder_resolved": str(input_dir.resolve()),
                     "native_calib_dat_present": calib_present},
        "staged_sources": staged,
        "rainfall": {**rainfall, "xml_key": "rainfall_data"},
        "legacy_options": {"settings": settings},
        "grid": {"legacy_shape": [base.legacy_nrows, base.legacy_ncols], "maple_shape": list(base.interior_shape),
                 "cellsize_m": base.cellsize_m, "nodata_value": nodata,
                 "orientation": "files are north-first; every array here is MAPLE south-first (row 0 = south)",
                 "n_active": int(active.sum()), "n_inactive_interior": int((~active).sum()),
                 "ring_nodata_cells": int(np.sum(dem_sf[_ring_mask(dem_sf.shape)] == nodata))},
        "placeholder_terrain": {"rule": raw["placeholder"]["rule"], "value_m": placeholder,
                                "cells": int((~active).sum()),
                                "scope": "MAPLE bed cells with no known terrain ONLY; excluded from every hydraulic computation; "
                                         "the original DEM with its nodata is kept in the sidecar",
                                "maple_mask_limitation": (
                                    "The placeholder elevation is supplied as an already-filled prepared array, so the compiled "
                                    "MAPLE masks do NOT distinguish these cells: all interior cells are labelled empirical_core "
                                    "and gap_filled is empty, although the inactive cells (their count is `cells`) are invented. "
                                    "The authoritative record is `inactive_interior` in the sidecar. The compiled case is a "
                                    "water-only benchmark bed and is NOT qualified for a production wind handoff.")},
        "hydrology": {
            "model": "fixed_ksat (native infiltration_model 1; Smith-Parlange capacity, linear drainage, saturation excess)",
            "ksat_mm_per_s": k_recipe, "ksat_native_formula_mm_per_s": k_formula,
            "ksat_source": "exact value captured from the executed native application (native_static_audit.json); native setup "
                           "inf_type 1 ignores the XML conductivity (storm_setting 420-426)",
            "suction_mm": float(hyd["suction_mm"]), "drainage_parameter": float(hyd["drainage_parameter"]),
            "theta_sat": theta_sat, "theta0": theta0, "soil_thickness_m": float(hyd["soil_thickness_m"]),
            "friction_factor": float(hyd["friction_factor"]), "pavement": 0.0,
            "departures_from_native_setup": [
                ("native setup overwrote the suction with 0.05 mm and left the drainage parameter at 0 "
                 "(storm_setting 524-529 passes psi as the drainage target when inf_type != 2); NOT reproduced: the XML "
                 "values 23.6 mm and 0.05 are used"),
                "native soil thickness is the FP32 rounding of 0.21 m (0.209999993); the canonical 0.21 m is used",
                "native flow routing is method 2 (Newton-Crank-Nicolson); the SYRUP routing is method 5 (bisection)"],
        },
        "terrain": {"summary": summary, "n_pit_storage": n_pits,
                    "pits_maple_rc": [[int(r), int(c)] for r, c in zip(*np.nonzero(graph.pit_storage), strict=True)],
                    "outlets_maple_rc": [list(o) for o in outlets], "outlets_north_first_0based": [list(o) for o in outlets_nf],
                    "native_ring_scaling_at_outlet_receiver": native_ring_scale,
                    "boundary_control_change": "every ring cell is an export receiver (matched legacy input: ring rmask < 0); the "
                                               "native positive ring scaling there did not flag the export",
                    "level_widths": levels, "n_levels": len(levels), "graph_input_sha256": graph.input_sha256,
                    "graph_policy": graph.policy},
        "grain_maps": {"files": phi_files, "composition": [float(v) for v in table[0]], "roundoff_normalization": norm_stats,
                       "pavement_rescaling": "not applicable (pavement 0)", "class_ids": list(CLASS_IDS)},
        "bed": {**bed.as_record(), "bulk_density_kg_m3": base.bulk_density_kg_m3,
                "active_layer_thickness_m": base.active_layer_thickness_m, "particle_density_kg_m3": particle_density,
                "vertical_composition": "homogeneous per cell: one composition fills every column"},
        "forcing": forcing_record if forcing_record is not None else {"kind": "legacy_file", "file": rain_name},
        "not_done": ["no sediment transport / splash / erosion / wind", "no evolving terrain", "no restart",
                     "native routing method 2 is only a separate matched-Fortran contender"],
    }
    return Plot1Audit(recipe=base, case_dir=case_dir, report=report, fields=fields,
                      elevation_source_m=bed_dem, final_fractions=final, particle_density_kg_m3=particle_density, bed=bed,
                      composition_codes=codes, composition_table=table, dem_staged_relpath=DERIVED_DEM_PATH)


# --------------------------------------------------------------------------------------------------------------------
# MAPLE case package
# --------------------------------------------------------------------------------------------------------------------
def _case_config(audit: Plot1Audit) -> dict[str, Any]:
    from maple.core.parameters.water_coupling import MAHLERAN_1_2_1_CLASS_DIAMETERS_M

    recipe, bed = audit.recipe, audit.bed
    ny, nx = recipe.interior_shape

    def boundary():
        return {"kind": "prescribed", "inflow_flux_kg_m_s": {c: 0.0 for c in CLASS_IDS}}

    def prepared(path, kind, units):
        return {"path": path, "format": "npy", "row_order": "south_to_north", "value_kind": kind, "value_units": units}

    profiles = {
        str(code + 1): {"intervals": [{"top_depth_m": 0.0, "bottom_depth_m": float(bed.vertical_extent_m),
                                       "fractions": {c: float(v) for c, v in zip(CLASS_IDS, row, strict=True)}}]}
        for code, row in enumerate(audit.composition_table)
    }
    return {
        "run_name": recipe.case_name, "seed": recipe.seed,
        "geometry": {"nx": nx, "ny": ny, "dx_m": recipe.cellsize_m, "dy_m": recipe.cellsize_m,
                     "boundary_x": boundary(), "boundary_y": boundary(), "voxel_dz_m": recipe.voxel_dz_m,
                     "bulk_density_kg_m3": recipe.bulk_density_kg_m3,
                     "active_layer_thickness_m": recipe.active_layer_thickness_m},
        "grain_classes": [{"class_id": c, "diameter_m": float(d), "particle_density_kg_m3": audit.particle_density_kg_m3,
                           "is_aggregate": False}
                          for c, d in zip(CLASS_IDS, MAHLERAN_1_2_1_CLASS_DIAMETERS_M, strict=True)],
        "topographic_wind": {"enabled": False},
        "topography": {"base": "imported", "nz": bed.nz, "perturbation": {"relief_m": 0.0, "correlation_length_m": 1.0}},
        "sediment_availability": {"global_available_fraction": recipe.available_fraction},
        "import": {
            "domain": {"label": "empirical_transformed"},
            "elevation": {"source": prepared(DERIVED_DEM_PATH, "length", "m"),
                          "transforms": [{"name": "datum_offset", "reference": "none", "offset_m": bed.datum_offset_m}]},
            "sediment": {"mode": "categorical_map",
                         "categorical_map": {"categories": {"source": prepared(DERIVED_CODES_PATH, "dimensionless",
                                                                                "dimensionless")},
                                             "profiles": profiles}},
        },
    }


_RFID_HEADER = (
    "# GENERATED by maple_syrup.rfid_case (EXPERIMENTAL RFID_2014 water-only benchmark) -- do not edit.\n"
    "# Bed elevation of the INACTIVE interior cells is a documented placeholder; see syrup/rfid_import_report.json.\n"
)


def _write_case_package(audit: Plot1Audit) -> dict[str, Any]:
    import yaml

    for relpath, array in ((DERIVED_CODES_PATH, audit.composition_codes), (DERIVED_DEM_PATH, audit.elevation_source_m)):
        path = audit.case_dir / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            np.save(handle, np.ascontiguousarray(array), allow_pickle=False)
    text = _RFID_HEADER + yaml.safe_dump(_case_config(audit), sort_keys=False)
    case_yaml = audit.case_dir / "case.yaml"
    with case_yaml.open("x", encoding="utf-8") as handle:
        handle.write(text)
    return {"case_yaml": str(case_yaml), "case_yaml_sha256": _sha256_bytes(text.encode("utf-8")),
            "composition_codes_sha256": _sha256_file(audit.case_dir / DERIVED_CODES_PATH),
            "elevation_placeholder_sha256": _sha256_file(audit.case_dir / DERIVED_DEM_PATH)}


def _write_sidecar(audit: Plot1Audit, forcing_record: dict[str, Any] | None) -> dict[str, str]:
    directory = audit.case_dir / SIDECAR_DIR
    directory.mkdir(parents=True, exist_ok=True)
    fields_path = directory / FIELDS_NAME
    with fields_path.open("xb") as handle:
        np.savez(handle, **audit.fields)
    out = {"fields_sha256": _sha256_file(fields_path)}
    if forcing_record is not None:
        out["forcing_sha256"] = _write_new_json(directory / FORCING_NAME, forcing_record)
    report = dict(audit.report)
    report["sidecar_fields"] = {"path": f"{SIDECAR_DIR}/{FIELDS_NAME}", "sha256": out["fields_sha256"],
                                "arrays": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in audit.fields.items()}}
    out["report_sha256"] = _write_new_json(directory / REPORT_NAME, report)
    return out


def generate_rfid_case(recipe: RfidRecipe, output_dir: str | Path, *, applied_forcing_capture: str | Path | None = None,
                       compile_case: bool = True, expected_maple_root: str | Path | None = None) -> dict[str, Any]:
    """Audit RFID_2014, write a MAPLE case package into the NEW `output_dir`, compile and reload it with MAPLE, check it
    with `check_compiled_plot1` and bind the SYRUP sidecar. A failure after the directory exists leaves
    `syrup/FAILED.json` and no binding. Only a directory with `syrup/rfid_binding.json` is a completed import."""
    dependency = resolve_maple_dependency(expected_maple_root)
    check_required_api(PHASE2_REQUIRED_MAPLE_API)
    maple_provenance = capture_maple_provenance(dependency)
    forcing_record = None
    kind = recipe.raw["forcing"]["kind"]
    if kind == "native_applied_capture":
        capture = Path(applied_forcing_capture or recipe.raw["forcing"]["capture_default_path"])
        if not capture.is_file():
            raise Plot1ImportError(f"applied-forcing capture not found: {capture}")
        payload = capture.read_bytes()
        digest = _sha256_bytes(payload)
        want = recipe.raw["forcing"]["capture_expected_sha256"]
        if want is not None and digest != want:
            raise Plot1ImportError(f"applied-forcing capture hash {digest} != the recipe's {want}")
        forcing_record = derive_applied_forcing(payload.decode("ascii"), capture_sha256=digest)
        forcing_record["capture_path"] = str(capture.resolve())

    output_dir = Path(output_dir).resolve()
    _refuse_output(output_dir, {"MAPLE": dependency.source_root, "MAHLERAN": recipe.base.mahleran_root,
                                "recipe": recipe.recipe_path.parent if recipe.recipe_path else None})
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    try:
        audit = audit_rfid(recipe, output_dir, forcing_record=forcing_record)
        package = _write_case_package(audit)
        sidecar = _write_sidecar(audit, forcing_record)
        result: dict[str, Any] = {"status": "case_written_not_compiled", "case_dir": str(output_dir), **package,
                                  "sidecar": sidecar}
        if not compile_case:
            return result
        from maple.case_tools.compilers.case_compiler import (
            compile_case as maple_compile,
        )
        from maple.case_tools.compilers.case_compiler import load_compiled_case

        with _git_optional_locks_disabled():
            compiled = maple_compile(output_dir)
        compiled_checks = check_compiled_plot1(compiled, audit)
        loaded = load_compiled_case(output_dir)
        loaded_checks = check_compiled_plot1(loaded, audit)
        _require_same_state(compiled, loaded)
        digest_after = source_tree_digest(dependency.package_dir).digest_sha256
        record = compiled.provenance_record
        bound_files = {
            "case_yaml_sha256": _sha256_file(output_dir / "case.yaml"),
            "provenance_yaml_sha256": _sha256_file(output_dir / "provenance.yaml"),
            "rfid_report_sha256": _sha256_file(output_dir / SIDECAR_DIR / REPORT_NAME),
            "rfid_fields_sha256": _sha256_file(output_dir / SIDECAR_DIR / FIELDS_NAME),
            "composition_codes_sha256": _sha256_file(output_dir / DERIVED_CODES_PATH),
            "elevation_placeholder_sha256": _sha256_file(output_dir / DERIVED_DEM_PATH),
        }
        if forcing_record is not None:
            bound_files["applied_forcing_sha256"] = _sha256_file(output_dir / SIDECAR_DIR / FORCING_NAME)
        binding = {
            "schema": BINDING_SCHEMA, "status": "ok", "case_dir": str(output_dir),
            "maple_case_identity_sha256": record["case_identity_sha256"], "maple_code_version": record["code_version"],
            "maple_artifact_sha256": record["artifact_sha256"], "maple_source_files": record["source"]["files"],
            **bound_files, "recipe_sha256": recipe.recipe_sha256, "mahleran_xml_sha256": audit.report["mahleran"]["xml_sha256"],
            "staged_sources_sha256": {k: v["sha256"] for k, v in audit.report["staged_sources"].items()},
            "rainfall_sha256": audit.report["rainfall"]["sha256"],
            "grid": {"ny": audit.recipe.interior_shape[0], "nx": audit.recipe.interior_shape[1], "nz": audit.bed.nz,
                     "class_ids": list(CLASS_IDS)},
            "checks": {"compiled": compiled_checks, "reloaded": loaded_checks, "compiled_equals_reloaded": True},
            "maple": maple_provenance, "maple_source_digest_after": digest_after,
            "maple_source_stable": digest_after == maple_provenance["package_source_digest"]["digest_sha256"],
            "maple_syrup": capture_syrup_provenance(), "environment": environment_record(),
            "not_done": ["no wind run", "no sediment/erosion", "no GPU claim", "no restart"],
        }
        _require(binding["maple_source_stable"], "MAPLE source changed during the import")
        binding_sha = _write_new_json(output_dir / SIDECAR_DIR / BINDING_NAME, binding)
        return {**result, "status": "ok", "binding_sha256": binding_sha,
                "maple_case_identity_sha256": record["case_identity_sha256"], "nz": audit.bed.nz,
                "datum_offset_m": audit.bed.datum_offset_m, "checks": compiled_checks}
    except BaseException as exc:
        with contextlib.suppress(Exception):
            _write_new_json(output_dir / SIDECAR_DIR / FAILURE_NAME, {
                "status": "failed", "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()})
        raise


def verify_rfid_case(case_dir: str | Path, *, mahleran_root: str | Path | None = None,
                     expected_maple_root: str | Path | None = None,
                     allow_maple_source_change: bool = False) -> VerifiedPlot1Case:
    """Re-verify a completed RFID import before any benchmark uses it (reads only). Every bound file must still hash to the
    binding; MAPLE reloads the case (`load_compiled_case` re-hashes its processed artifacts); the imported MAPLE source
    digest must equal the bound one (unless `allow_maple_source_change`, reported); the audit is re-run from the recipe and
    the MAHLERAN inputs in a temporary directory and every sidecar array and key report entry must agree exactly; and
    `check_compiled_plot1` passes on the loaded case. Raises `Plot1ImportError` on any disagreement."""
    from maple.case_tools.compilers.case_compiler import load_compiled_case

    case_dir = Path(case_dir).resolve()
    sidecar = case_dir / SIDECAR_DIR
    binding_path = sidecar / BINDING_NAME
    _require(binding_path.is_file(), f"{binding_path} missing: not a completed RFID import")
    _require(not (sidecar / FAILURE_NAME).exists(), f"{sidecar / FAILURE_NAME} exists: the import failed")
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    _require(binding.get("schema") == BINDING_SCHEMA and binding.get("status") == "ok", "not a completed RFID import")
    paths = {"case_yaml_sha256": case_dir / "case.yaml", "provenance_yaml_sha256": case_dir / "provenance.yaml",
             "rfid_report_sha256": sidecar / REPORT_NAME, "rfid_fields_sha256": sidecar / FIELDS_NAME,
             "composition_codes_sha256": case_dir / DERIVED_CODES_PATH,
             "elevation_placeholder_sha256": case_dir / DERIVED_DEM_PATH}
    if "applied_forcing_sha256" in binding:
        paths["applied_forcing_sha256"] = sidecar / FORCING_NAME
    for key, path in paths.items():
        _require(path.is_file() and _sha256_file(path) == binding[key], f"{path.name} does not match binding {key}")
    report = json.loads((sidecar / REPORT_NAME).read_text(encoding="utf-8"))
    with np.load(sidecar / FIELDS_NAME, allow_pickle=False) as data:
        fields = {name: data[name] for name in data.files}

    dependency = resolve_maple_dependency(expected_maple_root)
    check_required_api(PHASE2_REQUIRED_MAPLE_API)
    maple_record = capture_maple_provenance(dependency)
    now = maple_record["package_source_digest"]["digest_sha256"]
    bound = binding["maple"]["package_source_digest"]["digest_sha256"]
    _require(allow_maple_source_change or now == bound, f"MAPLE source digest {now} differs from the bound {bound}")

    case = load_compiled_case(case_dir)
    record = case.provenance_record
    _require(record["case_identity_sha256"] == binding["maple_case_identity_sha256"], "case identity differs")
    _require(record["artifact_sha256"] == binding["maple_artifact_sha256"], "artifact hash differs")
    for entry in binding["maple_source_files"]:
        path = case_dir / entry["authored_path"]
        _require(path.is_file() and _sha256_file(path) == entry["sha256"], f"{entry['authored_path']} modified")

    recipe_record = report["recipe"]
    recipe = load_rfid_recipe(recipe_record["recipe_path"], mahleran_root=mahleran_root or report["mahleran"]["root"])
    _require(recipe.recipe_sha256 == recipe_record["recipe_sha256"] == binding["recipe_sha256"], "recipe changed since the import")
    forcing_record = None
    if "applied_forcing_sha256" in binding:
        forcing_record = json.loads((sidecar / FORCING_NAME).read_text(encoding="utf-8"))
        capture = Path(forcing_record.get("capture_path", ""))
        if capture.is_file():  # re-derive when the original capture is still there; otherwise the bound copy is the authority
            _require(_sha256_file(capture) == forcing_record["capture_sha256"], "the applied-forcing capture changed")
            again = derive_applied_forcing(capture.read_text(encoding="ascii"), capture_sha256=forcing_record["capture_sha256"])
            _require(all(forcing_record[k] == again[k] for k in again), "re-derived applied forcing differs from the bound one")
    with tempfile.TemporaryDirectory(prefix="rfid_verify_") as tmp:
        fresh = audit_rfid(recipe, Path(tmp) / "case", forcing_record=forcing_record)
    _require(fresh.report["mahleran"]["xml_sha256"] == report["mahleran"]["xml_sha256"] == binding["mahleran_xml_sha256"],
             "MAHLERAN XML changed")
    for name, rec in report["staged_sources"].items():
        new = fresh.report["staged_sources"][name]
        _require(new["sha256"] == rec["sha256"] == binding["staged_sources_sha256"][name], f"MAHLERAN source {name} changed")
        path = case_dir / rec["relpath"]
        _require(path.is_file() and _sha256_file(path) == rec["sha256"], f"staged {name} modified")
    _require(set(report["staged_sources"]) == set(fresh.report["staged_sources"]), "staged source set differs")
    rain = report["rainfall"]
    _require(fresh.report["rainfall"]["sha256"] == rain["sha256"] == binding["rainfall_sha256"], "MAHLERAN rainfall file changed")
    rainfall_path = sidecar / "rainfall" / rain["file"]
    _require(rainfall_path.is_file() and _sha256_file(rainfall_path) == rain["sha256"], "staged rainfall modified")
    for key in ("hydrology", "terrain", "placeholder_terrain", "grid"):
        _require(_json_safe(fresh.report[key]) == _json_safe(report[key]), f"report section {key!r} differs from a fresh audit")
    _require(set(fields) == set(fresh.fields), "sidecar array names differ from a fresh audit")
    for name, array in fresh.fields.items():
        _require(fields[name].dtype == array.dtype and np.array_equal(fields[name], array),
                 f"sidecar array {name} differs from a fresh audit")
    compiled_checks = check_compiled_plot1(case, fresh)
    return VerifiedPlot1Case(
        case_dir=case_dir, case=case, fields=fields, report=report, binding=binding, rainfall_path=rainfall_path,
        checks={"binding_sha256": _sha256_file(binding_path), "maple_source_digest_now": now,
                "maple_source_digest_bound": bound, "maple_source_changed_since_import": now != bound,
                "fresh_audit": "recipe, XML, sources, staged copies, report sections and all sidecar arrays equal",
                "compiled_state": compiled_checks, "forcing": None if forcing_record is None else forcing_record["schema"]},
        maple_dependency=dependency, maple_provenance=maple_record)


# --------------------------------------------------------------------------------------------------------------------
# Solver inputs
# --------------------------------------------------------------------------------------------------------------------
def rfid_routing_graph(fields: dict[str, Any], report: dict[str, Any], *, xp=None):
    """The RFID `RoutingGraph` from the verified sidecar: the ORIGINAL full DEM with its nodata sentinel, the ring as the
    export receiver set, active interior cells, opt-in policies `allow_masked_nodata` and `allow_pit_storage`. The result is
    checked against the aspect and pit/outlet masks stored at import."""
    from maple_syrup.routing import build_routing_graph

    dem = np.asarray(fields["legacy_full_elevation_m"], dtype=np.float64)
    active = np.asarray(fields["active"], dtype=np.bool_)
    friction = np.full(active.shape, float(report["hydrology"]["friction_factor"]))
    graph = build_routing_graph(dem, _ring_mask(dem.shape), friction, float(report["grid"]["cellsize_m"]),
                                active_mask=active, nodata_value=float(report["grid"]["nodata_value"]),
                                allow_masked_nodata=True, allow_pit_storage=True, xp=xp)
    for name, got in (("graph_aspect", graph.aspect), ("graph_slope", graph.slope), ("graph_pit_storage", graph.pit_storage),
                      ("graph_outlet", graph.outlet)):
        if not np.array_equal(np.asarray(got), np.asarray(fields[name])):
            raise Plot1ImportError(f"rebuilt graph differs from the imported {name}")
    if graph.input_sha256 != report["terrain"]["graph_input_sha256"]:
        raise Plot1ImportError("rebuilt graph digest differs from the imported one")
    return graph


def rfid_parameters(report: dict[str, Any], fields: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Host FP64 `(ny, nx)` column inputs (SI) and the parameter record. Inactive cells get the active value so every
    parameter is valid there, but they hold no water (initial soil water and depth are zero there)."""
    h = report["hydrology"]
    active = np.asarray(fields["active"], dtype=np.bool_)
    shape = active.shape

    def full(v):
        return np.full(shape, float(v), dtype=np.float64)

    host = {
        "ksat_m_per_s": full(h["ksat_mm_per_s"] * 1.0e-3), "suction_m": full(h["suction_mm"] * 1.0e-3),
        "drainage_parameter": full(h["drainage_parameter"]), "theta_sat": full(h["theta_sat"]),
        "soil_thickness_m": full(h["soil_thickness_m"]), "initial_theta": full(h["theta0"]),
        "rainfall_scale": np.asarray(fields["rainfall_scaling"], dtype=np.float64), "active": active,
    }
    record = {"model": h["model"], "ksat_mm_per_s": h["ksat_mm_per_s"], "suction_mm": h["suction_mm"],
              "drainage_parameter": h["drainage_parameter"], "theta_sat": h["theta_sat"], "theta0": h["theta0"],
              "soil_thickness_m": h["soil_thickness_m"], "friction_factor": h["friction_factor"],
              "departures_from_native_setup": h["departures_from_native_setup"]}
    return host, record


def rfid_schedule(case_dir: str | Path, report: dict[str, Any]):
    """The rainfall schedule bound to the case: the stored applied-forcing record (kind `native_applied_capture`) or the
    staged legacy file. The hash is recorded in the schedule provenance."""
    from maple_syrup.rainfall import parse_legacy_rainfall_file

    case_dir = Path(case_dir)
    forcing = report["forcing"]
    if forcing.get("kind") == "native_applied_capture":
        path = case_dir / SIDECAR_DIR / FORCING_NAME
        return _schedule_from_record(json.loads(path.read_text(encoding="utf-8")), _sha256_file(path))
    schedule = parse_legacy_rainfall_file(case_dir / SIDECAR_DIR / "rainfall" / report["rainfall"]["file"])
    if schedule.provenance.sha256 != report["rainfall"]["sha256"]:
        raise Plot1ImportError("parsed rainfall bytes differ from the verified rainfall hash")
    return schedule


def rfid_inputs(verified: VerifiedPlot1Case, backend: str, *, with_geometry: bool) -> SimpleNamespace:
    """The factory `benchmarks/hydraulic_candidates/compare_plot1.build_runner(inputs_factory=...)` calls: the same namespace
    `experimental_experiment.plot1_inputs` returns (so the SAME scheduler, guards and budget code run), built for RFID.
    `with_geometry` builds the local-inertial geometry from the finite-substituted full DEM (faces touching an inactive cell
    or the ring are closed; only the legacy outlet face is open, an unchanged candidate rule)."""
    from maple.core.backend import resolve_backend, to_device

    from maple_syrup.experimental_hydrology import (
        build_local_inertial_geometry,
        open_faces_from_graph,
    )
    from maple_syrup.infiltration import column_parameters, initial_soil_water_m
    from maple_syrup.rainfall import rainfall_field

    case, fields, report = verified.case, verified.fields, verified.report
    schedule = rfid_schedule(verified.case_dir, report)
    host, parameter_record = rfid_parameters(report, fields)
    g = case.config.geometry
    ny, nx, n_classes = g.ny, g.nx, len(case.config.grain_classes.classes)
    if host["theta_sat"].shape != (ny, nx) or g.dx_m != g.dy_m:
        raise Plot1ImportError("sidecar fields do not match the MAPLE grid, or cells are not square")
    resolved = resolve_backend(backend)
    xp = resolved.xp
    graph = rfid_routing_graph(fields, report, xp=xp)
    if graph.dx_m != g.dx_m:
        raise Plot1ImportError("routing graph spacing differs from the MAPLE geometry")
    dev = {name: to_device(np.ascontiguousarray(array), xp) for name, array in host.items()}
    params = column_parameters(
        model="fixed_ksat",
        **{k: dev[k] for k in ("ksat_m_per_s", "suction_m", "drainage_parameter", "theta_sat", "soil_thickness_m")},
        active_mask=dev["active"])
    field = rainfall_field(ny, nx, scale=dev["rainfall_scale"], active_mask=dev["active"])
    depth0 = to_device(np.zeros((ny, nx), dtype=np.float64), xp)
    case_depth = np.asarray(case.water.depth_m)
    if np.any(case_depth != 0.0):
        raise Plot1ImportError("the MAPLE case starts with surface water; unsupported")
    soil0 = xp.where(dev["active"], initial_soil_water_m(params, dev["initial_theta"]), 0.0)
    geometry = None
    if with_geometry:
        full = np.array(fields["legacy_full_elevation_m"], dtype=np.float64)
        nodata = float(report["grid"]["nodata_value"])
        full[full == nodata] = float(report["placeholder_terrain"]["value_m"])  # finite; faces to these cells are closed
        geometry = build_local_inertial_geometry(full, np.asarray(graph.active), np.asarray(graph.friction_factor),
                                                 graph.dx_m, open_faces_from_graph(graph))
    settings = report["legacy_options"]["settings"]
    return SimpleNamespace(
        case=case, settings=settings, stormlength_s=float(settings["stormlength_s"]), schedule=schedule, host=host,
        parameter_record=parameter_record, grid=g, ny=ny, nx=nx, n_classes=n_classes, area=g.dx_m * g.dy_m,
        resolved=resolved, xp=xp, graph=graph, params=params, field=field, depth0=depth0, soil0=soil0, geometry=geometry)


# --------------------------------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m maple_syrup.rfid_case",
                                     description="Audit RFID_2014 and compile it as an actual MAPLE case (water-only benchmark).")
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--output-dir", required=True, help="NEW directory for the case package.")
    parser.add_argument("--applied-forcing-capture", help="syrup_hydro_steps.txt of the executed native run")
    parser.add_argument("--mahleran-root")
    parser.add_argument("--expected-maple-root")
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args(argv)
    try:
        recipe = load_rfid_recipe(args.recipe, mahleran_root=args.mahleran_root)
        summary = generate_rfid_case(recipe, args.output_dir, applied_forcing_capture=args.applied_forcing_capture,
                                     compile_case=not args.no_compile, expected_maple_root=args.expected_maple_root)
    except MapleDependencyError as exc:
        print(f"MAPLE dependency check failed: {exc}", file=sys.stderr)
        return 2
    except Plot1ImportError as exc:
        print(f"RFID import failed: {exc}", file=sys.stderr)
        return 1
    except (ValueError, FileExistsError) as exc:
        print(f"MAPLE rejected the case: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
