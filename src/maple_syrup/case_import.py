"""Phase 2: audit MAHLERAN Plot 1 and import it into an actual MAPLE case.

What this module does, and only this:

1. Stages the Plot 1 rasters named by the root ``mahleran_input.xml`` into a
   NEW case directory. A file is copied byte-for-byte, unless it carries
   non-grid bytes after its declared rows (``p1dem.asc`` does); then exactly
   the six header lines plus the declared rows are staged -- the records the
   Fortran reader (``read_spatial_data.f90``) consumes -- and the removed
   trailer is hashed and reported.
2. Reads every raster through MAPLE's own importer (``read_source``,
   ``build_imported_field``). ESRI ASCII is north-first; MAPLE row 0 is south.
   The 62 x 22 legacy grid is cropped to its 60 x 20 computed interior with
   MAPLE's ``crop`` transform; the one-cell ring is kept only as evidence.
3. Resolves which legacy maps and options the storm setup actually uses,
   records the root XML's repeated grain-map defect, applies the recipe's
   correction (six distinct supplied maps), normalizes float-storage
   roundoff only, and applies the legacy pavement rescaling.
4. Audits terrain with the legacy D4 aspect rule (``topog_attrib.for``):
   sinks, ring exits and boundary-ring evidence. Nothing is filled, routed
   or edited.
5. Writes a MAPLE ``case.yaml`` (imported DEM plus a declared constant datum
   offset; per-cell composition through MAPLE's ``categorical_map`` sediment
   import) and compiles and reloads it with MAPLE's ``compile_case`` and
   ``load_compiled_case``. No wind run is launched.
6. Checks the compiled bed against quantities computed from the source
   arrays, and binds a SYRUP sidecar (surface fields, audit masks, report)
   to the compiled case identity by SHA-256.

There is no rainfall, infiltration, routing or erosion physics here. All
work is host NumPy at case-initialization time, as in MAPLE's compiler.

    python -m maple_syrup.case_import --recipe cases/plot1/recipe.yaml --output-dir NEW_DIR
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import json
import math
import os
import sys
import traceback
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any

import numpy as np

import maple_syrup
from maple_syrup.dependency import (
    ApiRequirement,
    MapleDependencyError,
    check_required_api,
    resolve_maple_dependency,
)
from maple_syrup.provenance import (
    capture_maple_provenance,
    capture_syrup_provenance,
    environment_record,
    read_git_head,
    source_tree_digest,
)

__all__ = [
    "CLASS_IDS",
    "PHASE2_REQUIRED_MAPLE_API",
    "BedPlan",
    "Plot1Audit",
    "Plot1ImportError",
    "Plot1Recipe",
    "apply_legacy_pavement_rescaling",
    "audit_plot1",
    "check_compiled_plot1",
    "generate_plot1_case",
    "legacy_d4_audit",
    "load_recipe",
    "main",
    "normalize_closure_roundoff",
    "parse_mahleran_xml",
    "plan_bed",
    "resolve_particle_size_maps",
    "stage_legacy_ascii",
]

RECIPE_SCHEMA = "maple_syrup.plot1_recipe.v1"
REPORT_SCHEMA = "maple_syrup.plot1_import_report.v1"
BINDING_SCHEMA = "maple_syrup.plot1_binding.v1"

CLASS_IDS: tuple[str, ...] = tuple(f"phi_{k}" for k in range(1, 7))
N_CLASSES = len(CLASS_IDS)
# Fortran read_spatial_data reads exactly six header records.
LEGACY_HEADER_LINES = 6

STAGED_SOURCE_DIR = "source/mahleran_input_p1"
DERIVED_CODES_PATH = "source/syrup_derived/composition_codes.npy"
SIDECAR_DIR = "syrup"
REPORT_NAME = "plot1_import_report.json"
FIELDS_NAME = "plot1_fields.npz"
BINDING_NAME = "plot1_binding.json"
FAILURE_NAME = "FAILED.json"

# After the legacy rescaling of CLOSED fractions the six classes close again
# up to a few roundings; anything larger is a formula or input error.
_POST_RESCALE_CLOSURE_TOL = 1.0e-12
_EPS = float(np.finfo(np.float64).eps)

# Legacy D4 directions in topog_attrib.for's `sdir` order, on the NORTH-FIRST
# file grid: 1 = N (row - 1), 2 = E (col + 1), 3 = S (row + 1), 4 = W (col - 1).
_D4 = ((-1, 0), (0, 1), (1, 0), (0, -1))
_D4_NAMES = ("N", "E", "S", "W")

# Everything this module calls in MAPLE, beyond Phase 1's REQUIRED_MAPLE_API.
PHASE2_REQUIRED_MAPLE_API: tuple[ApiRequirement, ...] = (
    ApiRequirement(
        "maple.case_tools.compilers.case_compiler", "compile_case",
        parameters=("case_path", "output_path", "overwrite"),
    ),
    ApiRequirement(
        "maple.case_tools.compilers.case_compiler", "load_compiled_case",
        parameters=("case_path", "processed_path"),
    ),
    ApiRequirement(
        "maple.case_tools.importers.readers", "read_source",
        parameters=("spec", "case_dir", "label"),
    ),
    ApiRequirement(
        "maple.case_tools.importers.field", "build_imported_field",
        parameters=("spec", "geometry", "case_dir", "seed", "label", "semantics", "value_range"),
    ),
    ApiRequirement("maple.core.parameters.case_import", "resolve_imported_field"),
    ApiRequirement("maple.core.parameters.case_import", "validate_imported_field"),
    ApiRequirement("maple.core.parameters.case_initialization", "FRACTION_SUM_TOLERANCE"),
    ApiRequirement("maple.core.parameters.geometry", "GeometrySpec"),
    ApiRequirement("maple.core.boundaries", "AxisBoundary"),
    ApiRequirement("maple.core.boundaries", "BoundaryKind"),
    ApiRequirement("maple.core.parameters.water_coupling", "MAHLERAN_1_2_1_CLASS_DIAMETERS_M"),
    ApiRequirement("maple.surface.active_layer", "check_active_layer_voxel_partition"),
    ApiRequirement("maple.surface.active_layer.diagnostics", "diagnose_combined_surface_state"),
    ApiRequirement("maple.surface.voxels.diagnostics", "check_voxel_column_state"),
)


class Plot1ImportError(RuntimeError):
    """A Plot 1 source, decision or compiled-state check failed."""


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(Path(path).read_bytes())


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _write_new_json(path: Path, payload: dict[str, Any]) -> str:
    text = json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)
    return _sha256_bytes(text.encode("utf-8"))


def _legacy_index(maple_row: int, maple_col: int, legacy_nrows: int) -> dict[str, int]:
    """MAPLE interior (row 0 = south) -> legacy 1-based file indices."""
    return {
        "maple_row": int(maple_row),
        "maple_col": int(maple_col),
        "legacy_row_1based": int(legacy_nrows - 1 - maple_row),
        "legacy_col_1based": int(maple_col + 2),
    }


def _range(array: np.ndarray) -> list[float]:
    return [float(np.min(array)), float(np.max(array))]


# --------------------------------------------------------------------------
# Recipe
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Plot1Recipe:
    recipe_path: Path | None
    recipe_sha256: str | None
    case_name: str
    mahleran_root: Path
    xml_name: str
    expected_version: str
    expected_input_folder: str
    legacy_nrows: int
    legacy_ncols: int
    cellsize_m: float
    corrected_maps: tuple[str, ...]
    closure_tolerance: float
    pavement_rescaling: str
    voxel_dz_m: float
    active_layer_thickness_m: float
    bulk_density_kg_m3: float
    minimum_fill_depth_m: float
    datum_offset_rounding_m: float
    headroom_m: float
    seed: int
    available_fraction: float

    @property
    def interior_shape(self) -> tuple[int, int]:
        return (self.legacy_nrows - 2, self.legacy_ncols - 2)

    def as_record(self) -> dict[str, Any]:
        record = dataclasses.asdict(self)
        record["recipe_path"] = str(self.recipe_path) if self.recipe_path else None
        record["mahleran_root"] = str(self.mahleran_root)
        return record


def _section(raw: dict[str, Any], key: str, keys: set[str]) -> dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise Plot1ImportError(f"recipe: section {key!r} is required and must be a mapping")
    if set(value) != keys:
        raise Plot1ImportError(
            f"recipe: section {key!r} must have exactly keys {sorted(keys)}, got {sorted(value)}"
        )
    return value


def _positive(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise Plot1ImportError(f"recipe: {label} must be a finite number, got {value!r}")
    if value <= 0:
        raise Plot1ImportError(f"recipe: {label} must be > 0, got {value!r}")
    return float(value)


def load_recipe(path: str | Path, *, mahleran_root: str | Path | None = None) -> Plot1Recipe:
    """Read and strictly validate an authored Plot 1 recipe (no unknown keys)."""
    import yaml

    path = Path(path)
    payload = path.read_bytes()
    raw = yaml.safe_load(payload.decode("utf-8"))
    if not isinstance(raw, dict):
        raise Plot1ImportError(f"recipe {path} must be a mapping")
    expected = {"schema", "case_name", "mahleran", "legacy_grid", "grain_maps", "bed", "maple"}
    if set(raw) != expected:
        raise Plot1ImportError(f"recipe must have exactly keys {sorted(expected)}, got {sorted(raw)}")
    if raw["schema"] != RECIPE_SCHEMA:
        raise Plot1ImportError(f"recipe schema must be {RECIPE_SCHEMA!r}, got {raw['schema']!r}")
    case_name = raw["case_name"]
    if not isinstance(case_name, str) or not case_name:
        raise Plot1ImportError("recipe: case_name must be a non-empty string")

    mahleran = _section(raw, "mahleran", {"root", "xml", "expected_version", "expected_input_folder"})
    grid = _section(
        raw, "legacy_grid",
        {"nrows", "ncols", "cellsize_m", "interior_rows_north_first_1based", "interior_cols_1based"},
    )
    maps = _section(
        raw, "grain_maps",
        {"correction", "corrected_maps", "closure_roundoff_tolerance", "pavement_rescaling"},
    )
    bed = _section(
        raw, "bed",
        {"voxel_dz_m", "active_layer_thickness_m", "bulk_density_kg_m3", "minimum_fill_depth_m",
         "datum_offset_rounding_m", "headroom_m"},
    )
    maple = _section(raw, "maple", {"seed", "sediment_available_fraction", "boundary"})

    nrows, ncols = grid["nrows"], grid["ncols"]
    if not all(isinstance(v, int) and not isinstance(v, bool) and v >= 3 for v in (nrows, ncols)):
        raise Plot1ImportError("recipe: legacy_grid.nrows/ncols must be integers >= 3")
    # The legacy computed loops are i = 2..n_rows-1, k = 2..n_cols-1
    # (MAHLERAN_storm_setting_xml.f90 204-205; topog_attrib.for 44-45). A
    # different window would not be the legacy interior.
    if list(grid["interior_rows_north_first_1based"]) != [2, nrows - 1] or list(
        grid["interior_cols_1based"]
    ) != [2, ncols - 1]:
        raise Plot1ImportError(
            "recipe: the interior window must be the legacy computed loops "
            f"rows [2, {nrows - 1}] and cols [2, {ncols - 1}] (1-based, north-first)"
        )

    if maps["correction"] != "distinct_supplied_maps":
        raise Plot1ImportError("recipe: grain_maps.correction must be 'distinct_supplied_maps'")
    corrected = maps["corrected_maps"]
    if not isinstance(corrected, dict) or list(corrected) != list(CLASS_IDS):
        raise Plot1ImportError(f"recipe: grain_maps.corrected_maps must list exactly {list(CLASS_IDS)} in order")
    tolerance = _positive(maps["closure_roundoff_tolerance"], "grain_maps.closure_roundoff_tolerance")
    if tolerance > 1.0e-4:
        raise Plot1ImportError(
            "recipe: closure_roundoff_tolerance above 1e-4 would normalize material nonclosure, "
            "not float-storage roundoff"
        )
    if maps["pavement_rescaling"] not in ("legacy_storm_setting", "none"):
        raise Plot1ImportError("recipe: grain_maps.pavement_rescaling must be 'legacy_storm_setting' or 'none'")

    fraction = maple["sediment_available_fraction"]
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not 0.0 <= fraction <= 1.0:
        raise Plot1ImportError("recipe: maple.sediment_available_fraction must lie in [0, 1]")
    if maple["boundary"] != "prescribed_zero_inflow":
        raise Plot1ImportError("recipe: maple.boundary must be 'prescribed_zero_inflow'")
    if not isinstance(maple["seed"], int) or isinstance(maple["seed"], bool):
        raise Plot1ImportError("recipe: maple.seed must be an integer")

    recipe = Plot1Recipe(
        recipe_path=path.resolve(),
        recipe_sha256=_sha256_bytes(payload),
        case_name=case_name,
        mahleran_root=Path(mahleran_root if mahleran_root is not None else mahleran["root"]).resolve(),
        xml_name=str(mahleran["xml"]),
        expected_version=str(mahleran["expected_version"]),
        expected_input_folder=str(mahleran["expected_input_folder"]),
        legacy_nrows=nrows,
        legacy_ncols=ncols,
        cellsize_m=_positive(grid["cellsize_m"], "legacy_grid.cellsize_m"),
        corrected_maps=tuple(str(corrected[c]) for c in CLASS_IDS),
        closure_tolerance=tolerance,
        pavement_rescaling=maps["pavement_rescaling"],
        voxel_dz_m=_positive(bed["voxel_dz_m"], "bed.voxel_dz_m"),
        active_layer_thickness_m=_positive(bed["active_layer_thickness_m"], "bed.active_layer_thickness_m"),
        bulk_density_kg_m3=_positive(bed["bulk_density_kg_m3"], "bed.bulk_density_kg_m3"),
        minimum_fill_depth_m=_positive(bed["minimum_fill_depth_m"], "bed.minimum_fill_depth_m"),
        datum_offset_rounding_m=_positive(bed["datum_offset_rounding_m"], "bed.datum_offset_rounding_m"),
        headroom_m=_positive(bed["headroom_m"], "bed.headroom_m"),
        seed=int(maple["seed"]),
        available_fraction=float(fraction),
    )
    if recipe.active_layer_thickness_m >= recipe.minimum_fill_depth_m:
        raise Plot1ImportError("recipe: active_layer_thickness_m must be below minimum_fill_depth_m")
    return recipe


# --------------------------------------------------------------------------
# Staging legacy files
# --------------------------------------------------------------------------
def stage_legacy_ascii(original: str | Path, destination: str | Path) -> dict[str, Any]:
    """Copy one legacy ESRI ASCII grid into a NEW file.

    Mirrors what Fortran ``read_spatial_data`` consumes: six header records,
    then ``nrows`` records of ``ncols`` values each. Bytes after those
    records that are not whitespace (a trailer) are NOT staged -- MAPLE's
    reader rightly rejects a body that holds anything but the grid -- and are
    hashed and reported instead. A clean file is copied byte-for-byte.
    Refuses a short, wrapped or non-ASCII grid rather than guessing.
    """
    original, destination = Path(original), Path(destination)
    if not original.is_file():
        raise Plot1ImportError(f"legacy raster not found: {original}")
    payload = original.read_bytes()
    lines = payload.splitlines(keepends=True)
    if len(lines) < LEGACY_HEADER_LINES:
        raise Plot1ImportError(f"{original.name}: fewer than {LEGACY_HEADER_LINES} header lines")
    header: dict[str, str] = {}
    for number, line in enumerate(lines[:LEGACY_HEADER_LINES], start=1):
        try:
            parts = line.decode("ascii").split()
        except UnicodeDecodeError as exc:
            raise Plot1ImportError(f"{original.name}: header line {number} is not ASCII") from exc
        if len(parts) != 2:
            raise Plot1ImportError(f"{original.name}: header line {number} is not 'key value'")
        key = parts[0].lower()
        if key in header:
            raise Plot1ImportError(f"{original.name}: duplicate header key {key!r}")
        header[key] = parts[1]
    try:
        ncols, nrows = int(header["ncols"]), int(header["nrows"])
    except (KeyError, ValueError) as exc:
        raise Plot1ImportError(f"{original.name}: header lacks integer ncols/nrows") from exc
    end = LEGACY_HEADER_LINES + nrows
    if len(lines) < end:
        raise Plot1ImportError(
            f"{original.name}: declares {nrows} rows but holds only {len(lines) - LEGACY_HEADER_LINES}"
        )
    for number, line in enumerate(lines[LEGACY_HEADER_LINES:end], start=LEGACY_HEADER_LINES + 1):
        try:
            n_tokens = len(line.decode("ascii").split())
        except UnicodeDecodeError as exc:
            raise Plot1ImportError(f"{original.name}: data line {number} is not ASCII") from exc
        if n_tokens != ncols:
            raise Plot1ImportError(
                f"{original.name}: data line {number} holds {n_tokens} values, header declares "
                f"{ncols} (wrapped or short rows are not staged)"
            )
    prefix = b"".join(lines[:end])
    trailer = payload[len(prefix):]
    trailer_record: dict[str, Any] | None = None
    if trailer.strip():
        staged = prefix
        try:
            trailer.decode("ascii")
            ascii_decodable = True
        except UnicodeDecodeError:
            ascii_decodable = False
        trailer_record = {
            "length_bytes": len(trailer),
            "sha256": _sha256_bytes(trailer),
            "first_16_bytes_hex": trailer[:16].hex(),
            "ascii_decodable": ascii_decodable,
            "mentions_own_file_name": original.name.encode("ascii", "ignore") in trailer,
            "handling": (
                "not staged: bytes after the declared rows, which the Fortran reader never "
                "reads; staged file is the byte-identical header + declared rows prefix"
            ),
        }
    else:
        staged = payload
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as handle:
        handle.write(staged)
    return {
        "file": original.name,
        "original_path": str(original.resolve()),
        "original_sha256": _sha256_bytes(payload),
        "original_size_bytes": len(payload),
        "staged_path": str(destination),
        "staged_sha256": _sha256_bytes(staged),
        "staged_is_byte_identical": staged == payload,
        "staged_is_prefix_of_original": payload.startswith(staged),
        "header": header,
        "trailer": trailer_record,
    }


def _stage_verbatim(original: Path, destination: Path) -> dict[str, Any]:
    if not original.is_file():
        raise Plot1ImportError(f"legacy file not found: {original}")
    payload = original.read_bytes()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as handle:
        handle.write(payload)
    try:
        payload.decode("utf-8")
        utf8 = True
    except UnicodeDecodeError:
        utf8 = False
    return {
        "file": original.name,
        "original_path": str(original.resolve()),
        "sha256": _sha256_bytes(payload),
        "size_bytes": len(payload),
        "utf8_decodable": utf8,
        "staged_path": str(destination),
    }


# --------------------------------------------------------------------------
# Legacy XML and options
# --------------------------------------------------------------------------
def parse_mahleran_xml(path: str | Path) -> dict[str, Any]:
    """Every top-level element of a MAHLERAN input XML, duplicates kept."""
    root = ET.parse(path).getroot()
    entries: dict[str, list[dict[str, Any]]] = {}
    for element in root:
        entries.setdefault(element.tag, []).append(
            {
                "value": element.attrib.get("value"),
                "attributes": dict(element.attrib),
                "children": {child.tag: (child.text or "").strip() for child in element},
            }
        )
    return {"root_tag": root.tag, "root_attributes": dict(root.attrib), "entries": entries}


def _xml_record(xml: dict[str, Any], tag: str, *, required: bool = True) -> dict[str, Any] | None:
    records = xml["entries"].get(tag)
    if not records:
        if required:
            raise Plot1ImportError(f"mahleran XML: required element <{tag}> is missing")
        return None
    if any(r != records[0] for r in records[1:]):
        raise Plot1ImportError(f"mahleran XML: <{tag}> appears {len(records)} times with different content")
    return records[0]


def _xml_value(xml: dict[str, Any], tag: str) -> str:
    value = _xml_record(xml, tag)["value"]
    if value is None:
        raise Plot1ImportError(f"mahleran XML: <{tag}> has no value attribute")
    return value


def _fortran_logical(text: str) -> bool:
    # List-directed logical input: optional '.', then T or F decides.
    token = text.strip().lower().lstrip(".")
    if not token or token[0] not in "tf":
        raise Plot1ImportError(f"mahleran XML: {text!r} is not a Fortran logical")
    return token[0] == "t"


def _resolve_input_folder(root: Path, raw: str) -> Path:
    parts = [p for p in PureWindowsPath(raw).parts if p not in (".", "\\", "/")]
    return root.joinpath(*parts)


def resolve_particle_size_maps(
    xml_references: tuple[str, ...], corrected: tuple[str, ...]
) -> dict[str, Any]:
    """Compare the XML's six particle-size map references with the recipe's
    corrected selection. Repeated references are a defect that is REPORTED;
    they are never used as six classes. The selection itself must be six
    distinct files."""
    if len(xml_references) != N_CLASSES or len(corrected) != N_CLASSES:
        raise Plot1ImportError("particle-size map lists must have six entries")
    if len(set(corrected)) != N_CLASSES:
        repeated = sorted(name for name in set(corrected) if corrected.count(name) > 1)
        raise Plot1ImportError(
            f"selected particle-size maps repeat {repeated}; six classes need six distinct maps. "
            "Repeated maps are never normalized into a composition"
        )
    counts = {name: xml_references.count(name) for name in dict.fromkeys(xml_references)}
    return {
        "xml_references": list(xml_references),
        "xml_references_distinct": len(set(xml_references)) == N_CLASSES,
        "xml_reference_counts": counts,
        "selected_maps": list(corrected),
        "correction_applied": tuple(xml_references) != tuple(corrected),
    }


def resolve_legacy_options(xml: dict[str, Any]) -> dict[str, Any]:
    """Which XML maps and options the legacy storm setup actually uses.

    Follows MAHLERAN_storm_setting_xml.f90 (line numbers as read on
    2026-09-29). Values are raw XML text plus the resolved meaning; nothing
    here is used by physics in this phase.
    """
    def value(tag: str) -> str:
        return _xml_value(xml, tag)

    def children(tag: str) -> dict[str, str]:
        return dict(_xml_record(xml, tag)["children"])

    inf_type = int(value("infiltration-parameter_type"))
    ff_type = int(value("friction_factor_type"))
    n_types = int(value("number_of_surface_types"))
    use = {
        tag: _fortran_logical(value(tag))
        for tag in (
            "use_final_infiltration_map", "use_suction_map", "use_drainage_map",
            "use_initial_soil_moisture_map", "use_saturated_soil_moisture_map",
            "use_friction_factor_map", "use_map_phi",
        )
    }
    density_records = xml["entries"].get("particle_density", [])
    densities = sorted({r["value"] for r in density_records})
    if len(densities) != 1:
        raise Plot1ImportError(f"mahleran XML: particle_density values disagree: {densities}")

    def map_entry(tag, active, condition, evidence, consumer, units):
        return {
            "xml_key": tag,
            "file": value(tag) or None,
            "read_by_storm_setup": bool(active),
            "condition": condition,
            "evidence": evidence,
            "later_consumer": consumer,
            "units": units,
        }

    maps = [
        map_entry("dem", True, "always", "storm_setting 161-189 (m -> mm)",
                  "terrain", "m"),
        map_entry("vegetation-cover_map", True, "always", "storm_setting 226-227",
                  "raindrop_detachment.for 24-34: KE x (1 - 8.1e-3 veg)", "percent cover"),
        map_entry("surface-type_map", True, "always", "storm_setting 240-270",
                  f"per-type parameters; values clamped to [1, {n_types}], nodata -> 1",
                  "integer code"),
        map_entry("rainfall-scaling_map", True, "always", "storm_setting 275-276 (rmask)",
                  "rmask < 0 excludes cells (topog_attrib 190-197, 235-260; route_sediment_xml); "
                  "rainfall scaling traced in Phase 3", "dimensionless"),
        map_entry("pavement_map", len(value("pavement_map").strip()) > 5,
                  "len_trim(name) > 5 and file exists", "storm_setting 279-297",
                  "grain-fraction rescaling 352-369; pave x 1e-4 for infiltration 375-379",
                  "percent cover"),
        map_entry("final_infiltration_map", use["use_final_infiltration_map"] or inf_type == 4,
                  "use_final_infiltration_map or infiltration-parameter_type == 4",
                  "storm_setting 414-417", "ksat", "not established (inactive)"),
        map_entry("suction_map", use["use_suction_map"], "use_suction_map",
                  "storm_setting 469-473", "psi", "not established (inactive)"),
        map_entry("drainage-map_map", use["use_drainage_map"], "use_drainage_map",
                  "storm_setting 518-522", "drain_par", "not established (inactive)"),
        map_entry("friction_factor_map", use["use_friction_factor_map"] or ff_type == 9,
                  "use_friction_factor_map or friction_factor_type == 9",
                  "storm_setting 557-561", "ff", "not established (inactive)"),
        map_entry("initial_soil-moisture_map", use["use_initial_soil_moisture_map"],
                  "use_initial_soil_moisture_map", "storm_setting 705-709", "theta_0",
                  "m3/m3 (assumed; inactive)"),
        map_entry("saturated_soil-moisture_map", use["use_saturated_soil_moisture_map"],
                  "use_saturated_soil_moisture_map", "storm_setting 736-740", "theta_sat",
                  "m3/m3"),
    ]
    by_type = {
        tag: children(tag)
        for tag in (
            "final_infiltration_rate_mean", "final_infiltration_std_dev",
            "wetting_front_suction_mean", "wetting_front_suction_std_dev",
            "drainage_parameter_mean", "drainage_parameter_std_dev",
            "initial_soil_moisture_mean", "initial_soil_moisture_std_dev", "soil_thickness",
            "friction_factor_mean", "friction_factor_std_dev",
        )
    }
    settings = {
        "version": value("version"),
        "model_type": value("model_type"),
        "runtype": value("runtype"),
        "time_step_s": float(value("time_step")),
        "rain_type": int(value("rain_type")),
        "stormlength_s": float(value("stormlength")),
        "mean_rainfall": float(value("mean_rainfall")),
        "infiltration_model": int(value("infiltration_model")),
        "infiltration_parameter_type": inf_type,
        "flow_direction": int(value("flow_direction")),
        "flow_routing_solution_method": int(value("flow-routing_solution_method")),
        "sediment_routing_solution_method": int(value("sediment-routing_solution_method")),
        "friction_factor_type": ff_type,
        "number_of_surface_types": n_types,
        "use_flags": use,
        "distributions": {
            tag: value(tag)
            for tag in (
                "finalInfiltrationRateDistribution", "wettingFrontSuctionDistribution",
                "drainageParameterDistribution", "initialSoilMoistureDistribution",
                "saturatedSoilMoistureDistribution", "frictionFactorDistribution",
            )
        },
        "by_surface_type": by_type,
        "particle_density_raw_g_cm3": densities[0],
        "particle_density_occurrences": len(density_records),
        "active_layer_sensitivity": float(value("active_layer_sensitivity")),
        "update_topography_raw": value("update_topography"),
        "update_topography": value("update_topography").strip().lower() == "y",
        "KE_model_type": value("KE_model_type"),
    }
    resolved = [
        {"option": "final infiltration (ksat)",
         "resolved": (
             f"from surface types (infiltration-parameter_type {inf_type}), distribution "
             f"{settings['distributions']['finalInfiltrationRateDistribution']!r}, mean "
             f"{by_type['final_infiltration_rate_mean']}, std {by_type['final_infiltration_std_dev']}"
         ),
         "note": "a 'normal' draw with std > mean per surface type is stochastic and can go "
                 "negative; resolve in Phase 3 (storm_setting 429-452)."},
        {"option": "wetting-front suction (psi)",
         "resolved": f"from surface types, {by_type['wetting_front_suction_mean']}, then x psi_mod",
         "note": "psi_mod comes from calibration_xml (storm_setting 801-809); not traced here."},
        {"option": "drainage parameter", "resolved": f"from surface types, {by_type['drainage_parameter_mean']}",
         "note": "storm_setting 518-555"},
        {"option": "initial soil moisture (theta_0)",
         "resolved": f"from surface types {by_type['initial_soil_moisture_mean']} (map unused)",
         "note": "storm_setting 705-735"},
        {"option": "saturated soil moisture (theta_sat)",
         "resolved": "map" if use["use_saturated_soil_moisture_map"] else "from surface types",
         "note": "storm_setting 736-766"},
        {"option": "friction factor", "resolved": f"type {ff_type}: {by_type['friction_factor_mean']} per surface type",
         "note": "storm_setting 557-568"},
        {"option": "soil_thickness", "resolved": f"{by_type['soil_thickness']} parsed only",
         "note": "declared in parameters_from_xml.f90:67 and read by the XML reader; no storm-code "
                 "consumer found by source search. It is a soil-water depth, not a sediment inventory."},
        {"option": "particle density",
         "resolved": f"{float(densities[0]) * 1000.0} kg/m3 (XML {densities[0]} g/cm3, "
                     f"{len(density_records)} identical occurrences)", "note": ""},
        {"option": "active_layer_sensitivity",
         "resolved": settings["active_layer_sensitivity"],
         "note": "a legacy detachment coefficient; it is NOT an active-layer thickness."},
        {"option": "update_topography", "resolved": settings["update_topography"],
         "note": "initialize_values_xml.f90 138-142: only 'y'/'Y' enables it."},
        {"option": "grain composition", "resolved": "maps" if use["use_map_phi"] else "surface types",
         "note": "storm_setting 299-348; mean_particle_size is unused when use_map_phi is true."},
        {"option": "nutrients/marker/continuous sections", "resolved": "excluded",
         "note": "outside the first water experiment's scope."},
    ]
    particle_refs = children("particle_size_map")
    return {
        "maps": maps,
        "settings": settings,
        "resolved": resolved,
        "particle_size_map_references": [particle_refs.get(c, "") for c in CLASS_IDS],
    }


# --------------------------------------------------------------------------
# Composition
# --------------------------------------------------------------------------
def normalize_closure_roundoff(
    raw: np.ndarray, tolerance: float
) -> tuple[np.ndarray, dict[str, Any]]:
    """Divide each cell's fractions by their sum ONLY where that sum is within
    `tolerance` of 1 (float-storage roundoff). Any cell outside it, any
    negative or non-finite value, is rejected -- never rescaled."""
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim != 3 or raw.shape[-1] != N_CLASSES:
        raise Plot1ImportError(f"fractions must be (ny, nx, {N_CLASSES}), got {raw.shape}")
    if not np.all(np.isfinite(raw)):
        raise Plot1ImportError("grain fractions hold non-finite values")
    if np.any(raw < 0.0):
        raise Plot1ImportError(f"grain fractions hold {int(np.sum(raw < 0.0))} negative value(s)")
    sums = raw.sum(axis=-1)
    deviation = np.abs(sums - 1.0)
    bad = deviation > tolerance
    if np.any(bad):
        worst = np.unravel_index(int(np.argmax(deviation)), deviation.shape)
        raise Plot1ImportError(
            f"{int(bad.sum())} cell(s) have six-class sums departing from 1 by more than the "
            f"roundoff tolerance {tolerance}; worst sum {float(sums[worst])} at MAPLE cell "
            f"{tuple(int(i) for i in worst)}. This is material nonclosure and is rejected"
        )
    normalized = raw / sums[..., None]
    change = normalized - raw
    return normalized, {
        "tolerance": float(tolerance),
        "raw_sum_range": _range(sums),
        "max_abs_sum_deviation": float(deviation.max()),
        "max_abs_change_by_class": [float(v) for v in np.abs(change).max(axis=(0, 1))],
        "normalized_sum_max_abs_deviation": float(np.abs(normalized.sum(axis=-1) - 1.0).max()),
    }


def apply_legacy_pavement_rescaling(
    fractions: np.ndarray, pave_fraction: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    """MAHLERAN_storm_setting_xml.f90 352-369, vectorized:

        grav = phi5 + phi6;  fines = 1 - grav;  p = pave / 100
        p <= 0 or grav == 0:  unchanged
        otherwise:  phi1..4 *= (1 - p) / fines;  phi5..6 *= p / grav

    With CLOSED input this sets the gravel share to the pavement cover and
    keeps closure. Cells the legacy code would divide by zero (fines == 0
    with p > 0) and covers outside [0, 1] are rejected. Closure of the
    result is NOT enforced here, so the same function can characterize the
    root XML's unclosed as-configured input.
    """
    f = np.asarray(fractions, dtype=np.float64)
    p = np.asarray(pave_fraction, dtype=np.float64)
    if p.shape != f.shape[:-1]:
        raise Plot1ImportError(f"pavement shape {p.shape} does not match fractions {f.shape[:-1]}")
    if not np.all(np.isfinite(p)) or np.any(p < 0.0) or np.any(p > 1.0):
        raise Plot1ImportError("pavement cover must be finite and within [0, 100] percent")
    grav = f[..., 4] + f[..., 5]
    fines = 1.0 - grav
    apply = (p > 0.0) & (grav != 0.0)
    if np.any(apply & (fines == 0.0)):
        raise Plot1ImportError("legacy pavement rescaling would divide by zero fines (grav == 1, pave > 0)")
    fine_factor = np.where(apply, (1.0 - p) / np.where(apply, fines, 1.0), 1.0)
    coarse_factor = np.where(apply, p / np.where(apply, grav, 1.0), 1.0)
    out = f.copy()
    out[..., :4] = f[..., :4] * fine_factor[..., None]
    out[..., 4:] = f[..., 4:] * coarse_factor[..., None]
    return out, {
        "n_cells_rescaled": int(apply.sum()),
        "n_cells_pave_zero": int(np.sum(p <= 0.0)),
        "n_cells_pave_positive_grav_zero": int(np.sum((p > 0.0) & (grav == 0.0))),
        "gravel_share_before_range": _range(grav),
        "gravel_share_after_range": _range(out[..., 4] + out[..., 5]),
        "max_abs_change_by_class": [float(v) for v in np.abs(out - f).max(axis=(0, 1))],
        "sum_after_range": _range(out.sum(axis=-1)),
    }


# --------------------------------------------------------------------------
# Terrain audit (diagnostic only)
# --------------------------------------------------------------------------
def legacy_d4_audit(
    z_north_first_m: np.ndarray, rmask_north_first: np.ndarray, nodata_value: float
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Legacy D4 aspect (topog_attrib.for 85-117) on the full legacy grid,
    plus where each interior cell's steepest-descent path ends.

    Inputs are the whole legacy grids in FILE (north-first) order; outputs
    are interior arrays in the same order. Elevations are compared in mm as
    the legacy code does, with strict `<` in N, E, S, W order, so a tie never
    becomes a receiver and aspect 0 marks a sink or flat. This is an audit of
    the terrain the legacy code sees, not a routing solver: nothing is
    filled, rerouted or changed.
    """
    z_m = np.asarray(z_north_first_m, dtype=np.float64)
    rmask = np.asarray(rmask_north_first, dtype=np.float64)
    if z_m.shape != rmask.shape or z_m.ndim != 2 or min(z_m.shape) < 3:
        raise Plot1ImportError("D4 audit needs matching 2-D grids of at least 3 x 3")
    z = np.where(z_m > nodata_value, z_m * 1000.0, z_m)
    nr2, nc2 = z.shape
    centre = z[1:-1, 1:-1]
    zmin = centre.copy()
    aspect = np.zeros(centre.shape, dtype=np.int8)
    has_equal = np.zeros(centre.shape, dtype=bool)
    for code, (di, dk) in enumerate(_D4, start=1):  # four directions, cells vectorized
        neighbour = z[1 + di: nr2 - 1 + di, 1 + dk: nc2 - 1 + dk]
        valid = neighbour != nodata_value
        better = valid & (neighbour < zmin)
        aspect = np.where(better, np.int8(code), aspect)
        zmin = np.where(better, neighbour, zmin)
        has_equal |= valid & (neighbour == centre)

    ni, nk = centre.shape
    rows, cols = np.indices((ni, nk))
    rows, cols = rows + 1, cols + 1  # full-grid indices
    step_i = np.array([0, -1, 0, 1, 0])[aspect]
    step_k = np.array([0, 0, 1, 0, -1])[aspect]
    recv_i, recv_k = rows + step_i, cols + step_k
    on_ring = lambda i, k: (i == 0) | (i == nr2 - 1) | (k == 0) | (k == nc2 - 1)
    receiver_is_ring = (aspect > 0) & on_ring(recv_i, recv_k)

    # Node ids: interior cells first, then every ring cell. Ring cells and
    # sinks are fixed points; pointer doubling finds each path's terminal.
    node = np.full((nr2, nc2), -1, dtype=np.int64)
    node[1:-1, 1:-1] = np.arange(ni * nk).reshape(ni, nk)
    ring = on_ring(*np.indices((nr2, nc2)))
    ring_i, ring_k = np.nonzero(ring)
    node[ring_i, ring_k] = ni * nk + np.arange(ring_i.size)
    nxt = np.empty(ni * nk + ring_i.size, dtype=np.int64)
    nxt[: ni * nk] = np.where(aspect > 0, node[recv_i, recv_k], node[1:-1, 1:-1]).reshape(-1)
    nxt[ni * nk:] = np.arange(ni * nk, nxt.size)
    for _ in range(math.ceil(math.log2(max(nxt.size, 2))) + 1):
        nxt = nxt[nxt]
    if not np.array_equal(nxt[nxt], nxt):
        raise Plot1ImportError("D4 audit: descent paths did not terminate")
    terminal = nxt[: ni * nk].reshape(ni, nk)
    ends_in_ring = terminal >= ni * nk
    ring_index = np.where(ends_in_ring, terminal - ni * nk, 0)
    term_i, term_k = ring_i[ring_index], ring_k[ring_index]
    side = np.select(
        [term_i == 0, term_i == nr2 - 1, term_k == 0, term_k == nc2 - 1], [1, 3, 4, 2], 0
    ).astype(np.int8)
    drains_to_ring_side = np.where(ends_in_ring, side, 0).astype(np.int8)
    drains_to_masked_ring = ends_in_ring & (rmask[term_i, term_k] < 0.0)
    edge_side = np.where(receiver_is_ring, aspect, 0).astype(np.int8)

    sink = aspect == 0
    arrays = {
        "aspect": aspect,
        "sink": sink,
        "flat_sink": sink & has_equal,
        "strict_pit": sink & ~has_equal,
        "edge_outflow_side": edge_side,
        "drains_to_ring_side": drains_to_ring_side,
        "drains_to_masked_ring": drains_to_masked_ring,
    }
    sink_cells = []
    for i, k in zip(*np.nonzero(sink), strict=True):
        sink_cells.append(
            {
                "legacy_row_1based": int(i + 2),
                "legacy_col_1based": int(k + 2),
                "kind": "flat" if has_equal[i, k] else "strict_pit",
                "elevation_m": float(z_m[i + 1, k + 1]),
                "cells_draining_here": int(np.sum(terminal == node[i + 1, k + 1])),
            }
        )
    summary = {
        "rule": "legacy topog_attrib D4: strict '<' in N,E,S,W order on elevation in mm; "
                "aspect 0 = no strictly lower neighbour",
        "n_interior_cells": int(ni * nk),
        "n_sinks": int(sink.sum()),
        "n_flat_sinks": int((sink & has_equal).sum()),
        "n_strict_pits": int((sink & ~has_equal).sum()),
        "n_cells_ending_in_sinks": int((~ends_in_ring).sum()),
        "n_cells_ending_in_ring_by_side": {
            name: int(np.sum(drains_to_ring_side == code)) for code, name in enumerate(_D4_NAMES, 1)
        },
        "n_cells_ending_in_masked_ring": int(drains_to_masked_ring.sum()),
        "n_edge_outflow_cells_by_side": {
            name: int(np.sum(edge_side == code)) for code, name in enumerate(_D4_NAMES, 1)
        },
        "sinks": sink_cells,
    }
    return arrays, summary


def _ring_evidence(full_north_first: np.ndarray) -> dict[str, Any]:
    a = np.asarray(full_north_first, dtype=np.float64)
    pairs = {
        "N": (a[0, 1:-1], a[1, 1:-1]),
        "S": (a[-1, 1:-1], a[-2, 1:-1]),
        "W": (a[1:-1, 0], a[1:-1, 1]),
        "E": (a[1:-1, -1], a[1:-1, -2]),
    }
    return {
        side: {
            "ring_minus_adjacent_interior_range": _range(ring - inner),
            "ring_equals_adjacent_interior": bool(np.array_equal(ring, inner)),
            "ring_value_range": _range(ring),
        }
        for side, (ring, inner) in pairs.items()
    } | {"corners": [float(a[0, 0]), float(a[0, -1]), float(a[-1, 0]), float(a[-1, -1])]}


# --------------------------------------------------------------------------
# Bed plan
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class BedPlan:
    """Vertical datum and column allocation for the compiled bed.

    MAPLE elevation = legacy DEM elevation + `datum_offset_m`; the column
    base (MAPLE elevation 0) lies `datum_offset_m` below the legacy DEM
    datum. The whole column is erodible sediment of the cell's composition.
    """

    datum_offset_m: float
    nz: int
    voxel_dz_m: float
    vertical_extent_m: float
    elevation_m: np.ndarray  # (ny, nx) expected MAPLE surface = fill depth
    source_range_m: tuple[float, float]
    fill_depth_range_m: tuple[float, float]
    fill_depth_mean_m: float
    headroom_range_m: tuple[float, float]

    def as_record(self) -> dict[str, Any]:
        top_level = np.floor((self.elevation_m - 1e-12) / self.voxel_dz_m).astype(int)
        return {
            "datum_offset_m": self.datum_offset_m,
            "datum": "MAPLE elevation = legacy DEM elevation (m) + datum_offset_m; "
                     "MAPLE z = 0 is the base of the erodible column",
            "nz": self.nz,
            "voxel_dz_m": self.voxel_dz_m,
            "vertical_extent_m": self.vertical_extent_m,
            "source_dem_range_m": list(self.source_range_m),
            "fill_depth_range_m": list(self.fill_depth_range_m),
            "fill_depth_mean_m": self.fill_depth_mean_m,
            "headroom_range_m": list(self.headroom_range_m),
            "top_occupied_voxel_level_range": [int(top_level.min()), int(top_level.max())],
            "empty_levels_above_highest_surface": int(self.nz - 1 - top_level.max()),
        }


def plan_bed(z_source_m: np.ndarray, recipe: Plot1Recipe) -> BedPlan:
    """Declared constant datum offset and voxel allocation.

    The offset lifts the lowest interior DEM cell to at least
    `minimum_fill_depth_m`, rounded UP to `datum_offset_rounding_m`; DEM
    slopes are unchanged. `nz` leaves at least `headroom_m` above the highest
    surface. The elevation uses MAPLE's own `datum_offset` arithmetic
    (`values - 0.0 + offset`).
    """
    z = np.asarray(z_source_m, dtype=np.float64)
    zmin, zmax = float(z.min()), float(z.max())
    step = recipe.datum_offset_rounding_m
    offset = round(math.ceil((recipe.minimum_fill_depth_m - zmin) / step - 1e-9) * step, 12)
    elevation = z - 0.0 + offset
    if float(elevation.min()) < recipe.minimum_fill_depth_m - 1e-9:
        raise Plot1ImportError("datum offset leaves less than the minimum fill depth")
    dz = recipe.voxel_dz_m
    nz = math.ceil((float(elevation.max()) + recipe.headroom_m) / dz - 1e-9)
    extent = nz * dz
    if extent < float(elevation.max()) + recipe.headroom_m - 1e-9:
        raise Plot1ImportError("voxel allocation leaves less than the requested headroom")
    return BedPlan(
        datum_offset_m=float(offset),
        nz=nz,
        voxel_dz_m=dz,
        vertical_extent_m=float(extent),
        elevation_m=elevation,
        source_range_m=(zmin, zmax),
        fill_depth_range_m=(float(elevation.min()), float(elevation.max())),
        fill_depth_mean_m=float(elevation.mean()),
        headroom_range_m=(float(extent - elevation.max()), float(extent - elevation.min())),
    )


# --------------------------------------------------------------------------
# Audit
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Plot1Audit:
    recipe: Plot1Recipe
    case_dir: Path
    report: dict[str, Any]
    fields: dict[str, np.ndarray]  # sidecar arrays; MAPLE grid (row 0 = south)
    elevation_source_m: np.ndarray  # (ny, nx) legacy DEM metres
    final_fractions: np.ndarray  # (ny, nx, 6)
    particle_density_kg_m3: float
    bed: BedPlan
    composition_codes: np.ndarray  # (ny, nx) int32, 1-based
    composition_table: np.ndarray  # (n_codes, 6)
    dem_staged_relpath: str

    @property
    def expected_mass_by_cell_class_kg(self) -> np.ndarray:
        cell_area = self.recipe.cellsize_m * self.recipe.cellsize_m
        mass_per_m = self.recipe.bulk_density_kg_m3 * cell_area
        return (self.bed.elevation_m * mass_per_m)[..., None] * self.final_fractions


def _audit_geometry(recipe: Plot1Recipe):
    from maple.core.boundaries import AxisBoundary, BoundaryKind
    from maple.core.parameters.geometry import GeometrySpec

    ny, nx = recipe.interior_shape

    def boundary():
        return AxisBoundary(
            kind=BoundaryKind.PRESCRIBED, inflow_flux_kg_m_s={c: 0.0 for c in CLASS_IDS}
        )

    return GeometrySpec(
        nx=nx, ny=ny, dx_m=recipe.cellsize_m, dy_m=recipe.cellsize_m,
        boundary_x=boundary(), boundary_y=boundary(), voxel_dz_m=recipe.voxel_dz_m,
        bulk_density_kg_m3=recipe.bulk_density_kg_m3,
        active_layer_thickness_m=recipe.active_layer_thickness_m,
    )


def _source_dict(relpath: str, kind: str, units: str) -> dict[str, Any]:
    return {"path": relpath, "format": "ascii_grid", "value_kind": kind,
            "value_units": units, "horizontal_units": "m"}


def _crop_step(recipe: Plot1Recipe) -> dict[str, Any]:
    # MAPLE's reader has already flipped rows to south-first; a one-cell ring
    # is symmetric, so the window is [1, nrows-1) x [1, ncols-1) either way.
    return {"name": "crop", "row_start": 1, "row_stop": recipe.legacy_nrows - 1,
            "col_start": 1, "col_stop": recipe.legacy_ncols - 1}


def _read_field(recipe, case_dir, geometry, relpath, *, kind, units, semantics, value_range, label):
    """Read one legacy raster with MAPLE: the full south-first grid via
    `read_source`, and the cropped interior via `build_imported_field`."""
    from maple.case_tools.importers.field import build_imported_field
    from maple.case_tools.importers.readers import read_source
    from maple.core.parameters.case_import import (
        resolve_imported_field,
        validate_imported_field,
    )

    spec = resolve_imported_field(
        {"source": _source_dict(relpath, kind, units), "transforms": [_crop_step(recipe)]}, label
    )
    violations = validate_imported_field(spec, label)
    if violations:
        raise Plot1ImportError("; ".join(violations))
    full = read_source(spec.source, case_dir, label)
    expected = (recipe.legacy_nrows, recipe.legacy_ncols)
    if full.values.shape != expected:
        raise Plot1ImportError(f"{label}: grid shape {full.values.shape} differs from {expected}")
    interior_missing = full.nodata_mask[1:-1, 1:-1]
    if interior_missing.any():
        raise Plot1ImportError(
            f"{label}: {int(interior_missing.sum())} interior cell(s) are nodata; the fixture has "
            "no gap-fill policy and does not invent values"
        )
    result = build_imported_field(
        spec, geometry, case_dir, recipe.seed, label=label, semantics=semantics,
        value_range=value_range,
    )
    factor = {"percent": 0.01}.get(units, 1.0)
    if not np.array_equal(result.values, full.values[1:-1, 1:-1] * factor):
        raise Plot1ImportError(f"{label}: MAPLE's cropped field disagrees with its own full read")
    return full, result


def audit_plot1(recipe: Plot1Recipe, case_dir: str | Path) -> Plot1Audit:
    """Stage, read, resolve and audit Plot 1 into `case_dir` (which the
    caller created empty). Writes only staged copies under source/ and the
    rainfall copy under syrup/; the MAPLE case itself is written later."""
    case_dir = Path(case_dir)
    root = recipe.mahleran_root
    xml_path = root / recipe.xml_name
    if not xml_path.is_file():
        raise Plot1ImportError(f"MAHLERAN XML not found: {xml_path}")
    xml = parse_mahleran_xml(xml_path)
    options = resolve_legacy_options(xml)
    version = options["settings"]["version"]
    if version != recipe.expected_version:
        raise Plot1ImportError(f"MAHLERAN XML version {version!r} != expected {recipe.expected_version!r}")
    folder_raw = _xml_value(xml, "input_folder")
    input_dir = _resolve_input_folder(root, folder_raw)
    if input_dir.resolve() != (root / recipe.expected_input_folder).resolve():
        raise Plot1ImportError(f"XML input_folder {folder_raw!r} resolves to {input_dir}, not the recipe's")

    selection = resolve_particle_size_maps(
        tuple(options["particle_size_map_references"]), recipe.corrected_maps
    )
    map_files = {m["xml_key"]: m["file"] for m in options["maps"]}
    active = {m["xml_key"]: m["read_by_storm_setup"] for m in options["maps"]}
    for key in ("pavement_map", "saturated_soil-moisture_map"):
        if not active[key]:
            raise Plot1ImportError(f"this fixture expects {key} to be read by the legacy setup")
    if not options["settings"]["use_flags"]["use_map_phi"]:
        raise Plot1ImportError("this fixture expects use_map_phi = true")

    # Stage every raster the fixture reads; hash (only) inactive references.
    needed = [map_files[k] for k in (
        "dem", "vegetation-cover_map", "surface-type_map", "rainfall-scaling_map",
        "pavement_map", "saturated_soil-moisture_map",
    )]
    needed += [n for n in dict.fromkeys(selection["xml_references"] + selection["selected_maps"])]
    staged: dict[str, dict[str, Any]] = {}
    for name in dict.fromkeys(needed):
        staged[name] = stage_legacy_ascii(input_dir / name, case_dir / STAGED_SOURCE_DIR / name)
        staged[name]["relpath"] = f"{STAGED_SOURCE_DIR}/{name}"
    headers = {name: rec["header"] for name, rec in staged.items()}
    reference = headers[map_files["dem"]]
    for name, header in headers.items():
        for key in ("ncols", "nrows", "cellsize", "xllcorner", "yllcorner", "nodata_value"):
            if key not in header or float(header[key]) != float(reference[key]):
                raise Plot1ImportError(
                    f"{name}: header {key}={header.get(key)!r} differs from the DEM's {reference.get(key)!r}"
                )
    if (int(reference["nrows"]), int(reference["ncols"])) != (recipe.legacy_nrows, recipe.legacy_ncols):
        raise Plot1ImportError("DEM header shape differs from the recipe's legacy grid")
    if float(reference["cellsize"]) != recipe.cellsize_m:
        raise Plot1ImportError("DEM cellsize differs from the recipe")
    nodata = float(reference["nodata_value"])

    inactive_refs = []
    for m in options["maps"]:
        if m["read_by_storm_setup"] or not m["file"]:
            continue
        path = input_dir / m["file"]
        inactive_refs.append({
            "xml_key": m["xml_key"], "file": m["file"], "exists": path.is_file(),
            "sha256": _sha256_file(path) if path.is_file() else None,
        })
    rain_name = _xml_value(xml, "rainfall_data")
    rainfall = _stage_verbatim(input_dir / rain_name, case_dir / SIDECAR_DIR / "rainfall" / rain_name)

    geometry = _audit_geometry(recipe)

    def read(name, **kw):
        return _read_field(recipe, case_dir, geometry, staged[name]["relpath"], **kw)

    dem_full, dem = read(map_files["dem"], kind="length", units="m", semantics="continuous",
                         value_range=None, label="plot1.dem")
    if dem_full.nodata_mask.any():
        raise Plot1ImportError("the DEM holds nodata cells, including the ring; not supported")
    veg_full, veg = read(map_files["vegetation-cover_map"], kind="dimensionless", units="percent",
                         semantics="continuous", value_range=(0.0, 1.0), label="plot1.vegetation_cover")
    pave_full, pave = read(map_files["pavement_map"], kind="dimensionless", units="percent",
                           semantics="continuous", value_range=(0.0, 1.0), label="plot1.pavement_cover")
    _, stype = read(map_files["surface-type_map"], kind="dimensionless", units="dimensionless",
                             semantics="categorical", value_range=None, label="plot1.surface_type")
    rm_full, rm = read(map_files["rainfall-scaling_map"], kind="dimensionless", units="dimensionless",
                       semantics="continuous", value_range=None, label="plot1.rainfall_scaling")
    ts_full, ts = read(map_files["saturated_soil-moisture_map"], kind="dimensionless", units="fraction",
                       semantics="continuous", value_range=(0.0, 1.0), label="plot1.theta_sat")
    phi_interior: dict[str, np.ndarray] = {}
    phi_full: dict[str, np.ndarray] = {}
    for name in dict.fromkeys(selection["xml_references"] + selection["selected_maps"]):
        full, interior = read(name, kind="dimensionless", units="fraction", semantics="class_fraction",
                              value_range=(0.0, 1.0), label=f"plot1.{name}")
        phi_interior[name], phi_full[name] = interior.values, full.values

    if np.any(rm.values < 0.0):
        raise Plot1ImportError("interior rainfall-scaling mask excludes cells the fixture would keep")
    stype_raw = stype.values
    if not np.array_equal(stype_raw, np.round(stype_raw)):
        raise Plot1ImportError("surface-type map holds non-integer codes")
    n_types = options["settings"]["number_of_surface_types"]
    stype_resolved = np.clip(stype_raw, 1, n_types).astype(np.int16)

    # --- composition --------------------------------------------------------
    raw = np.stack([phi_interior[n] for n in selection["selected_maps"]], axis=-1)
    normalized, norm_stats = normalize_closure_roundoff(raw, recipe.closure_tolerance)
    if recipe.pavement_rescaling == "legacy_storm_setting":
        final, rescale_stats = apply_legacy_pavement_rescaling(normalized, pave.values)
    else:
        final, rescale_stats = normalized.copy(), {"n_cells_rescaled": 0, "note": "disabled by recipe"}
    closure = float(np.abs(final.sum(axis=-1) - 1.0).max())
    if closure > _POST_RESCALE_CLOSURE_TOL or np.any(final < 0.0):
        raise Plot1ImportError(f"resolved composition does not close (max |sum-1| {closure})")
    xml_stack = np.stack([phi_interior[n] for n in selection["xml_references"]], axis=-1)
    xml_post, _ = apply_legacy_pavement_rescaling(xml_stack, pave.values)
    xml_raw_sum, xml_post_sum = xml_stack.sum(axis=-1), xml_post.sum(axis=-1)

    table, inverse = np.unique(final.reshape(-1, N_CLASSES), axis=0, return_inverse=True)
    codes = (np.asarray(inverse).reshape(final.shape[:-1]) + 1).astype(np.int32)
    if not np.array_equal(table[codes - 1], final):
        raise Plot1ImportError("composition lookup table does not reproduce the per-cell fractions")

    # --- terrain --------------------------------------------------------------
    z_nf = dem_full.values[::-1]
    rmask_nf = np.where(rm_full.nodata_mask, nodata, rm_full.values)[::-1]
    d4, d4_summary = legacy_d4_audit(z_nf, rmask_nf, nodata)
    ring = {
        "dem_m": _ring_evidence(z_nf),
        "rainfall_scaling": _ring_evidence(rmask_nf),
        "pavement_percent": _ring_evidence(pave_full.values[::-1]),
        "vegetation_percent": _ring_evidence(veg_full.values[::-1]),
        "theta_sat": _ring_evidence(ts_full.values[::-1]),
        **{f"{n}": _ring_evidence(phi_full[n][::-1]) for n in selection["selected_maps"]},
    }
    ring_nodata = {"rainfall_scaling_nodata_cells": int(rm_full.nodata_mask.sum()),
                   "rainfall_scaling_nodata_rows_legacy_1based": sorted(
                       {int(recipe.legacy_nrows - r) for r in np.nonzero(rm_full.nodata_mask)[0]})}

    bed = plan_bed(dem.values, recipe)
    density_g_cm3 = float(options["settings"]["particle_density_raw_g_cm3"])
    particle_density = density_g_cm3 * 1000.0
    cell_area = recipe.cellsize_m ** 2
    expected_mass = (bed.elevation_m * recipe.bulk_density_kg_m3 * cell_area)[..., None] * final

    def to_maple(a):  # legacy north-first interior -> MAPLE south-first
        return np.ascontiguousarray(a[::-1])

    fields = {
        "elevation_source_m": dem.values,
        "elevation_maple_m": bed.elevation_m,
        "vegetation_cover_fraction": veg.values,
        "pavement_cover_fraction": pave.values,
        "surface_type_raw": stype_raw.astype(np.int16),
        "surface_type_resolved": stype_resolved,
        "rainfall_scaling": rm.values,
        "saturated_soil_moisture": ts.values,
        "phi_raw_selected": raw,
        "phi_normalized": normalized,
        "phi_final": final,
        "phi_xml_as_configured_raw_sum": xml_raw_sum,
        "phi_xml_as_configured_post_setup_sum": xml_post_sum,
        "composition_codes": codes,
        "composition_table": table,
        "expected_mass_by_cell_class_kg": expected_mass,
        "legacy_d4_aspect": to_maple(d4["aspect"]),
        "legacy_d4_sink": to_maple(d4["sink"]),
        "legacy_d4_flat_sink": to_maple(d4["flat_sink"]),
        "legacy_d4_strict_pit": to_maple(d4["strict_pit"]),
        "hydraulic_edge_outflow_side": to_maple(d4["edge_outflow_side"]),
        "hydraulic_candidate_outlet": to_maple(d4["edge_outflow_side"] > 0),
        "hydraulic_drains_to_ring_side": to_maple(d4["drains_to_ring_side"]),
        "hydraulic_drains_to_masked_ring": to_maple(d4["drains_to_masked_ring"]),
        "legacy_full_elevation_m": dem_full.values,
        "legacy_full_rainfall_scaling": np.where(rm_full.nodata_mask, nodata, rm_full.values),
    }
    for sink in d4_summary["sinks"]:
        r = recipe.legacy_nrows - sink["legacy_row_1based"]
        sink["maple_row"], sink["maple_col"] = int(r - 1), int(sink["legacy_col_1based"] - 2)

    worst_sum = np.unravel_index(int(np.argmax(np.abs(raw.sum(-1) - 1.0))), raw.shape[:-1])
    report = {
        "schema": REPORT_SCHEMA,
        "maple_syrup_version": maple_syrup.__version__,
        "recipe": recipe.as_record(),
        "mahleran": {
            "root": str(root),
            "xml_path": str(xml_path.resolve()),
            "xml_sha256": _sha256_file(xml_path),
            "xml_root": {"tag": xml["root_tag"], "attributes": xml["root_attributes"]},
            "version": version,
            "git": read_git_head(root),
            "input_folder_raw": folder_raw,
            "input_folder_resolved": str(input_dir.resolve()),
        },
        "staged_sources": staged,
        "inactive_referenced_maps": inactive_refs,
        "rainfall": {
            **rainfall,
            "xml_key": "rainfall_data",
            "status": "provenance only; no forcing integrator in Phase 2 (Phase 3)",
            "rain_type": options["settings"]["rain_type"],
            "stormlength_s": options["settings"]["stormlength_s"],
        },
        "legacy_options": options,
        "grid": {
            "legacy_shape": [recipe.legacy_nrows, recipe.legacy_ncols],
            "legacy_interior_1based_north_first": {
                "rows": [2, recipe.legacy_nrows - 1], "cols": [2, recipe.legacy_ncols - 1]},
            "maple_shape": list(recipe.interior_shape),
            "cellsize_m": recipe.cellsize_m,
            "orientation": "ESRI ASCII rows are north-first (MAHLERAN i = 1 is the first data line); "
                           "MAPLE row 0 is south. MAPLE's ascii_grid reader flips rows.",
            "index_map": f"MAPLE (row r, col c) = legacy 1-based (i = {recipe.legacy_nrows - 1} - r, k = c + 2)",
            "crop_transform": _crop_step(recipe),
            "headers_consistent": True,
            "header": reference,
            "ring_evidence": ring,
            "ring_nodata": ring_nodata,
        },
        "grain_maps": {
            **selection,
            "xml_defect": (
                "root XML maps phi_1..phi_6 to the same file; as configured the legacy setup "
                "would read one map six times"
            ) if not selection["xml_references_distinct"] else None,
            "decision": (
                "CORRECTED FIXTURE: use the six distinct supplied maps plot1_phi1..6.asc; "
                "normalize float-storage roundoff only; apply the legacy pavement rescaling. "
                "This is not a reproduction of the root XML."
            ),
            "selected_raw_sum_range": _range(raw.sum(-1)),
            "selected_worst_sum_cell": _legacy_index(*worst_sum, recipe.legacy_nrows),
            "roundoff_normalization": norm_stats,
            "pavement_rescaling": {"mode": recipe.pavement_rescaling, **rescale_stats},
            "final_max_abs_sum_deviation": closure,
            "xml_as_configured": {
                "raw_sum_range": _range(xml_raw_sum),
                "post_setup_sum_range": _range(xml_post_sum),
                "n_cells_post_setup_not_closed_1e-6": int(np.sum(np.abs(xml_post_sum - 1.0) > 1e-6)),
                "note": "diagnostic of the root XML's own configuration; never used as a composition",
            },
            "class_ids": list(CLASS_IDS),
            "class_order": "phi_1 (finest) .. phi_6 (coarsest), MAHLERAN order",
        },
        "surface_fields": {
            "vegetation_cover_fraction": {"range": _range(veg.values), "units": "fraction (file: percent)",
                                          "legacy_use": "raindrop KE attenuation (1 - 8.1e-3 * percent)"},
            "pavement_cover_fraction": {"range": _range(pave.values), "units": "fraction (file: percent)",
                                        "legacy_use": "grain-fraction rescaling; infiltration (pave x 1e-4)"},
            "surface_type": {"raw_unique": sorted({int(v) for v in np.unique(stype_raw)}),
                             "resolved_unique": sorted({int(v) for v in np.unique(stype_resolved)})},
            "rainfall_scaling": {"interior_unique": sorted({float(v) for v in np.unique(rm.values)})},
            "saturated_soil_moisture": {"range": _range(ts.values), "units": "m3/m3"},
            "note": "SYRUP sidecar fields for later water laws. They are NOT MAPLE vegetation, "
                    "availability, moisture or threshold-modifier state.",
        },
        "terrain": {
            "d4_audit": d4_summary,
            "outlet_candidate": "interior cells whose legacy D4 receiver is a ring cell "
                                "(hydraulic_candidate_outlet); not yet a SYRUP outlet decision",
            "interior_rmask_all_nonnegative": True,
        },
        "bed": {
            **bed.as_record(),
            "bulk_density_kg_m3": recipe.bulk_density_kg_m3,
            "active_layer_thickness_m": recipe.active_layer_thickness_m,
            "particle_density_kg_m3": particle_density,
            "vertical_composition": "homogeneous per cell: the resolved surface composition fills the "
                                    "whole column (no measured stratigraphy exists)",
            "total_mass_kg": float(expected_mass.sum()),
            "total_mass_by_class_kg": [float(v) for v in expected_mass.sum(axis=(0, 1))],
            "bulk_density_investigation": {
                "mahleran_value": "none (no bulk density or porosity in MAHLERAN source or XML)",
                "theta_sat_range": _range(ts.values),
                "theta_sat_implied_bulk_density_kg_m3": float((1.0 - float(ts.values.mean())) * particle_density),
                "decision": "1250 kg/m3 assumed (MAPLE default). theta_sat is a hydraulic parameter "
                            "and need not equal total porosity, so it was not adopted as porosity.",
            },
        },
        "availability": {
            "maple_global_available_fraction": recipe.available_fraction,
            "note": "MAPLE availability is the aeolian available/bound split. Legacy pavement and "
                    "vegetation effects stay SYRUP surface fields for later water laws.",
        },
        "composition_codes": {
            "n_codes": int(table.shape[0]),
            "path": DERIVED_CODES_PATH,
            "note": "MAPLE has no per-cell fraction-map import; each code is one exact resolved "
                    "composition, supplied through import.sediment.categorical_map",
        },
    }
    return Plot1Audit(
        recipe=recipe, case_dir=case_dir, report=report, fields=fields,
        elevation_source_m=dem.values, final_fractions=final,
        particle_density_kg_m3=particle_density, bed=bed, composition_codes=codes,
        composition_table=table, dem_staged_relpath=staged[map_files["dem"]]["relpath"],
    )


# --------------------------------------------------------------------------
# MAPLE case package
# --------------------------------------------------------------------------
def build_case_config(audit: Plot1Audit) -> dict[str, Any]:
    """The MAPLE case.yaml content (plain data)."""
    from maple.core.parameters.water_coupling import MAHLERAN_1_2_1_CLASS_DIAMETERS_M

    recipe, bed = audit.recipe, audit.bed
    ny, nx = recipe.interior_shape

    def boundary():
        return {"kind": "prescribed", "inflow_flux_kg_m_s": {c: 0.0 for c in CLASS_IDS}}

    profiles = {
        str(code + 1): {
            "intervals": [{
                "top_depth_m": 0.0,
                "bottom_depth_m": float(bed.vertical_extent_m),
                "fractions": {c: float(v) for c, v in zip(CLASS_IDS, row, strict=True)},
            }]
        }
        for code, row in enumerate(audit.composition_table)
    }
    return {
        "run_name": recipe.case_name,
        "seed": recipe.seed,
        "geometry": {
            "nx": nx, "ny": ny, "dx_m": recipe.cellsize_m, "dy_m": recipe.cellsize_m,
            "boundary_x": boundary(), "boundary_y": boundary(),
            "voxel_dz_m": recipe.voxel_dz_m,
            "bulk_density_kg_m3": recipe.bulk_density_kg_m3,
            "active_layer_thickness_m": recipe.active_layer_thickness_m,
        },
        "grain_classes": [
            {"class_id": c, "diameter_m": float(d),
             "particle_density_kg_m3": audit.particle_density_kg_m3, "is_aggregate": False}
            for c, d in zip(CLASS_IDS, MAHLERAN_1_2_1_CLASS_DIAMETERS_M, strict=True)
        ],
        # A nonperiodic plot cannot use the periodic FFT wind operator.
        "topographic_wind": {"enabled": False},
        "topography": {
            "base": "imported",
            "nz": bed.nz,
            "perturbation": {"relief_m": 0.0, "correlation_length_m": 1.0},
        },
        "sediment_availability": {"global_available_fraction": recipe.available_fraction},
        "import": {
            "domain": {"label": "empirical_transformed"},
            "elevation": {
                "source": _source_dict(audit.dem_staged_relpath, "length", "m"),
                "transforms": [
                    _crop_step(recipe),
                    {"name": "datum_offset", "reference": "none", "offset_m": bed.datum_offset_m},
                ],
            },
            "sediment": {
                "mode": "categorical_map",
                "categorical_map": {
                    "categories": {"source": {
                        "path": DERIVED_CODES_PATH, "format": "npy", "row_order": "south_to_north",
                        "value_kind": "dimensionless", "value_units": "dimensionless",
                    }},
                    "profiles": profiles,
                },
            },
        },
    }


_CASE_HEADER = (
    "# GENERATED by maple_syrup.case_import (MAPLE-SYRUP Phase 2) -- do not edit.\n"
    "# MAHLERAN Plot 1, CORRECTED fixture (six distinct grain maps, roundoff-only\n"
    "# normalization, legacy pavement rescaling). See syrup/plot1_import_report.json.\n"
)


def write_case_package(audit: Plot1Audit) -> dict[str, Any]:
    import yaml

    codes_path = audit.case_dir / DERIVED_CODES_PATH
    codes_path.parent.mkdir(parents=True, exist_ok=True)
    with codes_path.open("xb") as handle:
        np.save(handle, audit.composition_codes, allow_pickle=False)
    config = build_case_config(audit)
    text = _CASE_HEADER + yaml.safe_dump(config, sort_keys=False)
    case_yaml = audit.case_dir / "case.yaml"
    with case_yaml.open("x", encoding="utf-8") as handle:
        handle.write(text)
    return {
        "case_yaml": str(case_yaml),
        "case_yaml_sha256": _sha256_bytes(text.encode("utf-8")),
        "composition_codes_sha256": _sha256_file(codes_path),
    }


def write_sidecar(audit: Plot1Audit) -> dict[str, str]:
    directory = audit.case_dir / SIDECAR_DIR
    directory.mkdir(parents=True, exist_ok=True)
    fields_path = directory / FIELDS_NAME
    with fields_path.open("xb") as handle:
        np.savez(handle, **audit.fields)
    fields_sha = _sha256_file(fields_path)
    report = dict(audit.report)
    report["sidecar_fields"] = {
        "path": f"{SIDECAR_DIR}/{FIELDS_NAME}",
        "sha256": fields_sha,
        "orientation": "(ny, nx[, 6]) MAPLE grid, row 0 = south, FP64 unless integer/bool; "
                       "legacy_full_* are the 62 x 22 legacy grids, also row 0 = south",
        "arrays": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in audit.fields.items()},
    }
    report_sha = _write_new_json(directory / REPORT_NAME, report)
    return {"fields_sha256": fields_sha, "report_sha256": report_sha}


# --------------------------------------------------------------------------
# Independent checks of the compiled state
# --------------------------------------------------------------------------
def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Plot1ImportError(message)


def _mass_tolerance(n_terms: int, magnitude_kg: float, mass_resolution_kg: float) -> float:
    # Sub-resolution residuals MAPLE may leave per voxel and class, plus FP64
    # roundoff at the compared magnitude.
    return n_terms * mass_resolution_kg + 64.0 * n_terms * _EPS * magnitude_kg


def check_compiled_plot1(case: Any, audit: Plot1Audit) -> dict[str, Any]:
    """Compare a compiled or reloaded MAPLE case with quantities computed
    from the source arrays and the declared assumptions. Raises
    `Plot1ImportError` on any disagreement; returns the measured errors."""
    from maple.core.parameters.water_coupling import MAHLERAN_1_2_1_CLASS_DIAMETERS_M
    from maple.surface.active_layer import check_active_layer_voxel_partition
    from maple.surface.active_layer.diagnostics import diagnose_combined_surface_state
    from maple.surface.voxels.diagnostics import check_voxel_column_state

    recipe, bed = audit.recipe, audit.bed
    g = case.config.geometry
    ny, nx = recipe.interior_shape
    _require((g.ny, g.nx) == (ny, nx), f"geometry {(g.ny, g.nx)} != {(ny, nx)}")
    _require(g.dx_m == g.dy_m == recipe.cellsize_m, "cell size differs from the source")
    _require(g.voxel_dz_m == recipe.voxel_dz_m, "voxel_dz_m differs")
    _require(g.bulk_density_kg_m3 == recipe.bulk_density_kg_m3, "bulk density differs")
    _require(g.active_layer_thickness_m == recipe.active_layer_thickness_m, "active layer differs")
    classes = case.config.grain_classes.classes
    _require(tuple(c.class_id for c in classes) == CLASS_IDS, "grain class order differs")
    _require(
        tuple(c.diameter_m for c in classes) == tuple(MAHLERAN_1_2_1_CLASS_DIAMETERS_M),
        "grain diameters differ from MAPLE's MAHLERAN table",
    )
    _require(all(c.particle_density_kg_m3 == audit.particle_density_kg_m3 for c in classes),
             "particle density differs")

    voxel = np.asarray(case.voxel_column.mass_kg)
    active = np.asarray(case.active_layer.mass_kg)
    _require(voxel.shape == (ny, nx, bed.nz, N_CLASSES), f"voxel shape {voxel.shape} unexpected")
    mass_resolution = float(case.config.numerics.mass_resolution_kg)
    cell_area = recipe.cellsize_m ** 2
    mass_per_m = recipe.bulk_density_kg_m3 * cell_area

    # Terrain: elevation, zero perturbation, preserved DEM slopes.
    elevation = np.asarray(case.topography_result.elevation_m)
    elevation_error = float(np.abs(elevation - bed.elevation_m).max())
    _require(elevation_error <= 8 * _EPS * bed.vertical_extent_m,
             f"compiled elevation differs from source DEM + offset by {elevation_error} m")
    _require(not np.any(np.asarray(case.topography_result.perturbation_m)), "perturbation not zero")
    slope_error = max(
        float(np.abs(np.diff(elevation, axis=a) - np.diff(audit.elevation_source_m, axis=a)).max())
        for a in (0, 1)
    )
    _require(slope_error <= 16 * _EPS * bed.vertical_extent_m, f"DEM slopes changed by {slope_error} m")

    # Per-cell, per-class bed inventory.
    expected = audit.expected_mass_by_cell_class_kg
    bed_mass = voxel.sum(axis=2) + active
    max_cell = float(expected.sum(axis=-1).max())
    # Deposit and active-layer extraction may each leave a sub-resolution
    # residual per voxel and class.
    tol_cell = _mass_tolerance(2 * bed.nz + 4, max_cell, mass_resolution)
    inventory_error = float(np.abs(bed_mass - expected).max())
    _require(inventory_error <= tol_cell,
             f"bed inventory differs from area*density*depth*fraction by {inventory_error} kg (tol {tol_cell})")
    class_error = float(np.abs(bed_mass.sum(axis=(0, 1)) - expected.sum(axis=(0, 1))).max())
    _require(class_error <= tol_cell * ny * nx, f"per-class domain inventory differs by {class_error} kg")
    maple_expected_error = float(
        np.abs(np.asarray(case.expected_total_mass_by_class_kg) - expected.sum(axis=(0, 1))).max()
    )
    _require(maple_expected_error <= tol_cell * ny * nx,
             f"MAPLE's own expected class totals differ from the source-derived ones by {maple_expected_error} kg")

    # Voxel packing: bottom-up full voxels below the active layer.
    voxel_depth = bed.elevation_m - recipe.active_layer_thickness_m
    levels = np.arange(bed.nz, dtype=np.float64) * bed.voxel_dz_m
    expected_levels = np.clip(voxel_depth[..., None] - levels, 0.0, bed.voxel_dz_m) * mass_per_m
    level_error = float(np.abs(voxel.sum(axis=3) - expected_levels).max())
    tol_level = _mass_tolerance(N_CLASSES + 2, bed.voxel_dz_m * mass_per_m, mass_resolution) + tol_cell
    _require(level_error <= tol_level, f"voxel level masses differ from bottom-up packing by {level_error} kg")
    top_used = int(np.max(np.nonzero(voxel.sum(axis=(0, 1, 3)) > 0.0)[0]))
    top_expected = int(np.floor((float(voxel_depth.max()) - 1e-12) / bed.voxel_dz_m))
    _require(top_used == top_expected,
             f"highest occupied voxel level {top_used} != {top_expected} expected from the fill depth")
    _require(bed.vertical_extent_m - float(bed.elevation_m.max()) >= recipe.headroom_m - 1e-9,
             "allocated column leaves less than the declared headroom")

    # Active layer: target thickness and the cell's resolved composition.
    target = recipe.active_layer_thickness_m * mass_per_m
    active_total = active.sum(axis=-1)
    active_error = float(np.abs(active_total - target).max())
    _require(active_error <= _mass_tolerance(bed.nz + 2, target, mass_resolution),
             f"active-layer mass differs from its target by {active_error} kg")
    composition_error = float(np.abs(active / active_total[..., None] - audit.final_fractions).max())
    _require(composition_error <= 1e-9, f"active-layer composition differs by {composition_error}")

    # MAPLE's own validators and surface diagnostic.
    check_voxel_column_state(case.voxel_column, g, mass_resolution)
    check_active_layer_voxel_partition(case.active_layer, case.voxel_column, g, mass_resolution)
    surface = np.asarray(
        diagnose_combined_surface_state(case.voxel_column, case.active_layer, g, mass_resolution)
        .combined_elevation_m
    )
    surface_error = float(np.abs(surface - bed.elevation_m).max())
    _require(surface_error * mass_per_m <= tol_cell + tol_level,
             f"MAPLE-diagnosed surface differs from the declared one by {surface_error} m")

    # Nothing moving, nothing pending.
    water = case.water
    _require(not np.any(np.asarray(water.depth_m)), "initial water depth is not zero")
    _require(not np.any(np.asarray(water.mobile_mass_by_cell_class_kg)), "initial mobile mass is not zero")
    _require(case.initial_mobile_state is None, "an initial mobile seed exists")
    ledger = case.sediment_ledger
    for field in dataclasses.fields(ledger):
        value = np.asarray(getattr(ledger, field.name))
        if value.dtype.kind in "biuf":
            _require(not np.any(value), f"ledger field {field.name} is not zero")

    availability = case.sediment_availability
    available_error = float(
        np.abs(np.asarray(availability.available_mass_kg) - recipe.available_fraction * active).max()
    )
    _require(available_error <= 64 * _EPS * target, f"available mass differs by {available_error} kg")
    _require(
        float(np.abs(np.asarray(availability.bound_mass_kg) - (1.0 - recipe.available_fraction) * active).max())
        <= 64 * _EPS * target,
        "bound mass differs from the declared availability",
    )
    _require(case.vegetation.source == "none", "MAPLE vegetation should be absent")
    masks = case.masks
    if masks:
        _require(bool(np.all(masks.get("empirical_core", True))), "not every cell is empirical core")
        for name in ("synthetic_buffer", "excluded", "gap_filled"):
            _require(not np.any(masks.get(name, False)), f"{name} cells exist")

    return {
        "elevation_max_abs_m": elevation_error,
        "slope_max_abs_m": slope_error,
        "inventory_max_abs_kg": inventory_error,
        "inventory_tolerance_kg": tol_cell,
        "class_total_max_abs_kg": class_error,
        "maple_expected_total_max_abs_kg": maple_expected_error,
        "voxel_level_max_abs_kg": level_error,
        "voxel_level_tolerance_kg": tol_level,
        "top_used_voxel_level": top_used,
        "active_layer_max_abs_kg": active_error,
        "active_composition_max_abs": composition_error,
        "maple_surface_max_abs_m": surface_error,
        "available_mass_max_abs_kg": available_error,
        "total_bed_mass_by_class_kg": [float(v) for v in bed_mass.sum(axis=(0, 1))],
        "maple_validators": "check_voxel_column_state, check_active_layer_voxel_partition passed",
        "water_mobile_ledger": "all zero",
    }


def _require_same_state(compiled: Any, loaded: Any) -> None:
    pairs = {
        "voxel": (compiled.voxel_column.mass_kg, loaded.voxel_column.mass_kg),
        "active": (compiled.active_layer.mass_kg, loaded.active_layer.mass_kg),
        "available": (compiled.sediment_availability.available_mass_kg,
                      loaded.sediment_availability.available_mass_kg),
        "elevation": (compiled.topography_result.elevation_m, loaded.topography_result.elevation_m),
        "water_depth": (compiled.water.depth_m, loaded.water.depth_m),
    }
    for name, (a, b) in pairs.items():
        _require(np.array_equal(np.asarray(a), np.asarray(b)), f"reloaded {name} differs from the compiled one")
    _require(
        compiled.provenance_record["case_identity_sha256"] == loaded.provenance_record["case_identity_sha256"],
        "reloaded case identity differs",
    )


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
@contextlib.contextmanager
def _git_optional_locks_disabled():
    """MAPLE's compile_case runs `git status` in its own checkout; keep git
    from refreshing that checkout's index while we compile."""
    previous = os.environ.get("GIT_OPTIONAL_LOCKS")
    os.environ["GIT_OPTIONAL_LOCKS"] = "0"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("GIT_OPTIONAL_LOCKS", None)
        else:
            os.environ["GIT_OPTIONAL_LOCKS"] = previous


def _refuse_output(output_dir: Path, forbidden: dict[str, Path | None]) -> None:
    if output_dir.exists():
        raise Plot1ImportError(f"refusing to write into existing path {output_dir}")
    for label, root in forbidden.items():
        if root is not None and output_dir.is_relative_to(root.resolve()):
            raise Plot1ImportError(f"refusing to write inside the {label} tree {root}")


def generate_plot1_case(
    recipe: Plot1Recipe,
    output_dir: str | Path,
    *,
    compile_case: bool = True,
    expected_maple_root: str | Path | None = None,
) -> dict[str, Any]:
    """Audit Plot 1, write a MAPLE case package into the NEW `output_dir`,
    compile and reload it with MAPLE, check it, and bind the SYRUP sidecar.

    On failure after the directory exists, `syrup/FAILED.json` records the
    error and no binding file is written: only a directory holding
    `syrup/plot1_binding.json` is a completed import.
    """
    dependency = resolve_maple_dependency(expected_maple_root)
    check_required_api(PHASE2_REQUIRED_MAPLE_API)
    maple_provenance = capture_maple_provenance(dependency)

    output_dir = Path(output_dir).resolve()
    _refuse_output(output_dir, {
        "MAPLE": dependency.source_root, "MAHLERAN": recipe.mahleran_root,
        "recipe": recipe.recipe_path.parent if recipe.recipe_path else None,
    })
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    try:
        audit = audit_plot1(recipe, output_dir)
        package = write_case_package(audit)
        sidecar = write_sidecar(audit)
        result: dict[str, Any] = {
            "status": "case_written_not_compiled",
            "case_dir": str(output_dir),
            **package,
            "sidecar": sidecar,
        }
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
        binding = {
            "schema": BINDING_SCHEMA,
            "status": "ok",
            "case_dir": str(output_dir),
            "maple_case_identity_sha256": record["case_identity_sha256"],
            "maple_code_version": record["code_version"],
            "maple_case_schema_version": record.get("case_schema_version"),
            "provenance_yaml_sha256": _sha256_file(output_dir / "provenance.yaml"),
            "case_yaml_sha256": _sha256_file(output_dir / "case.yaml"),
            "maple_artifact_sha256": record["artifact_sha256"],
            "maple_source_files": record["source"]["files"],
            "syrup_report_sha256": _sha256_file(output_dir / SIDECAR_DIR / REPORT_NAME),
            "syrup_fields_sha256": _sha256_file(output_dir / SIDECAR_DIR / FIELDS_NAME),
            "grid": {"ny": audit.recipe.interior_shape[0], "nx": audit.recipe.interior_shape[1],
                     "nz": audit.bed.nz, "class_ids": list(CLASS_IDS)},
            "checks": {"compiled": compiled_checks, "reloaded": loaded_checks,
                       "compiled_equals_reloaded": True},
            "maple": maple_provenance,
            "maple_source_digest_after": digest_after,
            "maple_source_stable": digest_after == maple_provenance["package_source_digest"]["digest_sha256"],
            "maple_syrup": capture_syrup_provenance(),
            "environment": environment_record(),
            "not_done": [
                "no wind run launched", "no water physics", "no GPU placement or GPU claim",
                "hydraulic outlet/pit policy not decided (audit masks only)",
            ],
        }
        _require(binding["maple_source_stable"], "MAPLE source changed during the import")
        binding_sha = _write_new_json(output_dir / SIDECAR_DIR / BINDING_NAME, binding)
        return {
            **result,
            "status": "ok",
            "binding_sha256": binding_sha,
            "maple_case_identity_sha256": record["case_identity_sha256"],
            "nz": audit.bed.nz,
            "datum_offset_m": audit.bed.datum_offset_m,
            "n_composition_codes": int(audit.composition_table.shape[0]),
            "checks": compiled_checks,
        }
    except BaseException as exc:
        with contextlib.suppress(Exception):
            _write_new_json(output_dir / SIDECAR_DIR / FAILURE_NAME, {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            })
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m maple_syrup.case_import",
        description="MAPLE-SYRUP Phase 2: audit MAHLERAN Plot 1 and compile it as a MAPLE case.",
    )
    parser.add_argument("--recipe", required=True, help="Authored recipe, e.g. cases/plot1/recipe.yaml")
    parser.add_argument("--output-dir", required=True, help="NEW directory for the case package.")
    parser.add_argument("--mahleran-root", help="Override the recipe's MAHLERAN root (read-only).")
    parser.add_argument("--expected-maple-root", help="Refuse any other MAPLE source root.")
    parser.add_argument("--no-compile", action="store_true",
                        help="Write the audited case package but do not call MAPLE's compiler.")
    args = parser.parse_args(argv)
    try:
        recipe = load_recipe(args.recipe, mahleran_root=args.mahleran_root)
        summary = generate_plot1_case(
            recipe, args.output_dir, compile_case=not args.no_compile,
            expected_maple_root=args.expected_maple_root,
        )
    except MapleDependencyError as exc:
        print(f"MAPLE dependency check failed: {exc}", file=sys.stderr)
        return 2
    except Plot1ImportError as exc:
        print(f"plot1 import failed: {exc}", file=sys.stderr)
        return 1
    except (ValueError, FileExistsError) as exc:
        # MAPLE's schema, compiler and loader report rejections as ValueError;
        # the traceback is kept in <output-dir>/syrup/FAILED.json.
        print(f"MAPLE rejected the case: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
