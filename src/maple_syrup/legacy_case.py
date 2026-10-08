"""Case adapters for the native-walk legacy benchmark (task gpu_sediment, A1): Plot 1, RFID_2014 and Chastre.

Every adapter starts from an already VERIFIED imported MAPLE case (`verify_plot1_case`, `verify_rfid_case`,
`verify_chastre_case`: hashes, MAPLE reload, fresh audit) and returns a `LegacyCase`: the water inputs the prepared hydrology
needs, the wet-law parameters read from the ACTUAL source XML, the frozen composition and the prescribed vegetation. Nothing
here mutates MAPLE state or a case directory, and no dense voxel bed is ever built for Chastre (tile 0 is loaded once,
read-only, to validate the composition; the 44 tiles are guarded by the verification's file hashes).

Provenance rules:

* Sediment parameters come from `parse_mahleran_xml` of the XML whose SHA-256 equals the case report's, with the six class
  diameters equal to the legacy `shared_data.f90` radii doubled in the MAPLE class order and one particle density. The
  result is COMPARED with the transcribed Plot 1 constants and the differing keys are recorded; it is never replaced by them.
  The XML `Raindrop_detachment_d` values are recorded as UNUSED (`spq` comes from `shared_data.f90`, like the accepted code).
* Vegetation (RFID/Chastre) is read from the case's staged copy of the XML-named `vegetation-cover_map`, re-hashed against the
  report, required uniform (percent in [0, 100]) over the active cells, and stored as a fraction. Chastre uses its source RFID
  case (its binding pins the RFID report hash, checked here).
* Composition: the report composition vector is compared with the ACTUAL loaded MAPLE active layer (RFID: every cell; Chastre:
  every cell of tile 0). Holdings given to `prepare_legacy_physics` are that actual mass (RFID) or tile 0's cell (0,0) mass vector
  broadcast over the grid (Chastre), after the per-cell fraction check.
"""
from __future__ import annotations

import gc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from maple_syrup.case_import import (
    CLASS_IDS,
    _sha256_file,
    _xml_record,
    _xml_value,
    parse_mahleran_xml,
)
from maple_syrup.sediment_physics import (
    KE_MODELS,
    LEGACY_CLASS_RADII_M,
    PhysicsGrid,
    SedimentPhysicsParameters,
    physics_grid_from_graph,
    plot1_sediment_parameters,
    sediment_physics_parameters,
)

__all__ = ["LegacyCase", "LegacyCaseError", "legacy_case_for", "sediment_parameters_from_xml", "uniform_vegetation_fraction"]

_KE_MODEL_BY_XML = {"1": KE_MODELS[0], "2": KE_MODELS[1]}
_RESOLVED = {"ke_vegetation_form": "legacy_literal", "distance_convention": "legacy_literal",
             "dstar_convention": "legacy_sigma_minus_one", "bagnold_depth_units": "legacy_mm",
             "raindrop_composition_scaling": "legacy_none"}
_FRACTION_ATOL = 1.0e-12


class LegacyCaseError(ValueError):
    pass


def _tree_files(directory: Path) -> list[Path]:
    """Every regular file under `directory` (sidecars are small; tile directories are never passed here)."""
    return sorted(p for p in directory.rglob("*") if p.is_file()) if directory.is_dir() else []


@dataclass
class LegacyCase:
    kind: str
    case_dir: Path
    verified: Any
    graph: Any
    column: Any
    field: Any
    schedule: Any
    schedule_record: dict[str, Any]
    initial_storm: Any
    sediment: SedimentPhysicsParameters
    grid: PhysicsGrid
    vegetation: np.ndarray  # (ny, nx) fraction
    holdings_kg: np.ndarray  # (ny, nx, nc) frozen composition carrier (only its fractions are used)
    end_s: float
    default_bisection_iterations: int
    record: dict[str, Any]  # JSON-safe provenance of everything above
    pin_paths: tuple[Path, ...] = ()  # small input artifacts (XML, forcing, vegetation, sidecars) the driver re-hashes after the run
    rainfall_scale: np.ndarray | None = None  # host (ny, nx) case-verified rainfall scaling (0 on inactive cells)
    cell_area_m2: float = 0.0

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.graph.shape)

    @property
    def n_classes(self) -> int:
        return int(self.sediment.n_classes)


def sediment_parameters_from_xml(xml_path: str | Path, expected_sha256: str, grain_classes: Any):
    """`(SedimentPhysicsParameters, record)` from the actual XML. Refuses a different file hash, a class order or diameter
    different from the legacy radii doubled, several particle densities, an unsupported KE model or a time step other than the
    legacy 1 s. Records every difference from the transcribed Plot 1 constants (never substitutes them)."""
    xml_path = Path(xml_path)
    sha = _sha256_file(xml_path)
    if sha != expected_sha256:
        raise LegacyCaseError(f"{xml_path} hashes to {sha}, the verified case report binds {expected_sha256}")
    xml = parse_mahleran_xml(xml_path)

    def per_class(tag: str) -> tuple[float, ...]:
        children = _xml_record(xml, tag)["children"]
        missing = [c for c in CLASS_IDS if c not in children]
        if missing:
            raise LegacyCaseError(f"mahleran XML: <{tag}> lacks {missing}")
        return tuple(float(children[c]) for c in CLASS_IDS)

    classes = grain_classes.classes
    if tuple(c.class_id for c in classes) != CLASS_IDS:
        raise LegacyCaseError(f"MAPLE grain classes {[c.class_id for c in classes]} are not {list(CLASS_IDS)} in order")
    diameters = tuple(float(c.diameter_m) for c in classes)
    if diameters != tuple(2.0 * r for r in LEGACY_CLASS_RADII_M):
        raise LegacyCaseError("MAPLE class diameters differ from the legacy shared_data.f90 radii doubled")
    density_g_cm3 = float(_xml_value(xml, "particle_density"))
    if {float(c.particle_density_kg_m3) for c in classes} != {density_g_cm3 * 1000.0}:
        raise LegacyCaseError("MAPLE particle densities differ from the XML particle_density")
    ke_raw = _xml_value(xml, "KE_model_type").strip()
    if ke_raw not in _KE_MODEL_BY_XML:
        raise LegacyCaseError(f"unsupported KE_model_type {ke_raw!r}")
    time_step = float(_xml_value(xml, "time_step"))
    if time_step != 1.0:
        raise LegacyCaseError(f"XML time_step {time_step} s: the legacy replay is fixed at the Fortran dt = 1 s")
    selector = _xml_record(xml, "sediment-routing_solution_method", required=False)
    kwargs = {
        "diameter_m": diameters,
        "raindrop_a": per_class("Raindrop_detachment_a_parameter_size"),
        "raindrop_b": per_class("Raindrop_detachment_b_parameter_size"),
        "raindrop_c": per_class("Raindrop_detachment_c_parameter_size"),
        "raindrop_max_depth_mm": per_class("Raindrop_detachment_max_parameter_size"),
        "particle_density_g_cm3": density_g_cm3,
        "active_layer_sensitivity_mm": float(_xml_value(xml, "active_layer_sensitivity")),
        "ke_model": _KE_MODEL_BY_XML[ke_raw],
        "reference_interval_s": time_step,
        **_RESOLVED,
    }
    params = sediment_parameters_from_kwargs(kwargs)
    reference = plot1_sediment_parameters(**_RESOLVED).summary()
    summary = params.summary()
    differing = sorted(k for k in summary if summary[k] != reference.get(k))
    record = {
        "xml_path": str(xml_path.resolve()), "xml_sha256": sha, "ke_model_type_xml": ke_raw,
        "sediment_routing_selector_in_xml": None if selector is None else selector["value"],
        "sediment_routing_used_by_this_replay": "Crank-Nicolson (selector 2)",
        "raindrop_d_parameter_xml_unused": per_class("Raindrop_detachment_d_parameter_size"),
        "differs_from_transcribed_plot1_constants": differing,
        "parameters": summary,
    }
    return params, record


def sediment_parameters_from_kwargs(kwargs: dict[str, Any]) -> SedimentPhysicsParameters:
    return sediment_physics_parameters(xp=np, **kwargs)


def uniform_vegetation_fraction(rfid_case_dir: str | Path, rfid_report: dict[str, Any], source_active: np.ndarray,
                                target_shape: tuple[int, int] | None = None):
    """Vegetation fraction of an RFID-family case from the SOURCE RFID case's bound, re-hashed staged raster.

    `source_active` is the SOURCE case's active mask (its interior shape, e.g. 104 x 58 for RFID): the raster is read and
    validated only there (hash, uniform over the source active cells, [0, 100] percent). The verified uniform value is then
    broadcast to `target_shape` (default: the source shape; Chastre passes its own 1393 x 1604 grid). Nothing is resampled
    and no new vegetation map is invented; a non-uniform source is refused. Refuses a missing or modified file."""
    active = np.asarray(source_active, dtype=bool)
    target = tuple(active.shape) if target_shape is None else tuple(int(v) for v in target_shape)
    from maple_syrup.rfid_case import read_legacy_grid

    case_dir = Path(rfid_case_dir)
    xml = parse_mahleran_xml(rfid_report["mahleran"]["xml_path"])
    if _sha256_file(Path(rfid_report["mahleran"]["xml_path"])) != rfid_report["mahleran"]["xml_sha256"]:
        raise LegacyCaseError("the source XML differs from the report hash")
    name = _xml_value(xml, "vegetation-cover_map")
    staged = rfid_report["staged_sources"].get(name)
    if staged is None:
        raise LegacyCaseError(f"the vegetation map {name!r} is not among the bound staged sources")
    path = case_dir / staged["relpath"]
    if not path.is_file() or _sha256_file(path) != staged["sha256"]:
        raise LegacyCaseError(f"staged vegetation map {path} is missing or modified")
    _header, body = read_legacy_grid(path)
    interior = body[::-1][1:-1, 1:-1]  # file north-first -> MAPLE south-first, then the computed interior
    if interior.shape != active.shape:
        raise LegacyCaseError(f"vegetation interior {interior.shape} != grid {active.shape}")
    values = np.unique(interior[active])
    if values.size != 1:
        raise LegacyCaseError(f"vegetation cover is not uniform over the active cells ({values.size} distinct values)")
    percent = float(values[0])
    if not (0.0 <= percent <= 100.0):
        raise LegacyCaseError(f"vegetation cover {percent} percent is outside [0, 100]")
    record = {"file": name, "path": str(path), "sha256": staged["sha256"], "value_percent": percent,
              "n_source_active": int(active.sum()), "source_shape": list(active.shape), "target_shape": list(target),
              "broadcast_to_target": target != tuple(active.shape),
              "inactive_cells_filled_with_the_active_value": True}
    return np.full(target, percent / 100.0, dtype=np.float64), record


def _check_composition(mass: np.ndarray, composition: list[float], where: str) -> dict[str, Any]:
    mass = np.asarray(mass, dtype=np.float64)
    total = mass.sum(axis=-1)
    if mass.shape[-1] != len(composition) or not np.all(total > 0.0):
        raise LegacyCaseError(f"{where}: active layer has an empty cell or the wrong class count")
    fractions = mass / total[..., None]
    err = float(np.abs(fractions - np.asarray(composition)).max())
    if err > _FRACTION_ATOL:
        raise LegacyCaseError(f"{where}: active-layer fractions differ from the bound composition by {err:.3e}")
    return {"checked_cells": int(total.size), "max_abs_fraction_difference": err, "atol": _FRACTION_ATOL}


def _holdings(kind: str, loaded: Any, shape: tuple[int, int], composition: list[float], where: str):
    mass = np.asarray(loaded.active_layer.mass_kg, dtype=np.float64)
    check = _check_composition(mass, composition, where)
    if kind == "rfid":
        if mass.shape[:2] != shape:
            raise LegacyCaseError("RFID active layer does not match the grid")
        return np.array(mass, copy=True), check
    row = np.array(mass[0, 0, :], copy=True)
    return np.ascontiguousarray(np.broadcast_to(row, (*shape, row.size))), check


def _rfid_family(kind: str, verified: Any, args: dict[str, Any]) -> LegacyCase:
    from maple_syrup.rfid_case import rfid_inputs
    from maple_syrup.storm import initial_state

    inputs = rfid_inputs(verified, "numpy", with_geometry=False)
    if kind == "rfid":
        rfid_dir, rfid_report = verified.case_dir, verified.report
    else:
        pins = verified.binding["source_rfid"]
        rfid_dir = Path(pins["case_dir"])
        report_path = rfid_dir / "syrup" / "rfid_import_report.json"
        if _sha256_file(report_path) != pins["report_sha256"]:
            raise LegacyCaseError("the source RFID report differs from the hash bound by the Chastre case")
        rfid_report = json.loads(report_path.read_text(encoding="utf-8"))
    mah = rfid_report["mahleran"]
    params, xml_record = sediment_parameters_from_xml(mah["xml_path"], mah["xml_sha256"], verified.case.config.grain_classes)
    composition = list(rfid_report["grain_maps"]["composition"])
    if kind == "rfid":
        loaded, where = verified.case, "RFID active layer"
    else:
        from maple.case_tools.compilers.case_compiler import load_compiled_case

        tile0 = verified.case.tiles[0]
        loaded, where = load_compiled_case(Path(verified.case_dir) / tile0["dir"]), "Chastre tile 0 active layer"
    holdings, comp_check = _holdings(kind, loaded, tuple(inputs.graph.shape), composition, where)
    del loaded
    gc.collect()
    if kind == "rfid":
        source_active = np.asarray(verified.fields["active"], dtype=bool)
    else:  # the SOURCE RFID mask (104 x 58), from the sidecar the Chastre binding pins; not the Chastre mask
        fields_path = rfid_dir / "syrup" / "rfid_fields.npz"
        if _sha256_file(fields_path) != pins["fields_sha256"]:
            raise LegacyCaseError("the source RFID fields differ from the hash bound by the Chastre case")
        with np.load(fields_path, allow_pickle=False) as data:
            source_active = np.asarray(data["active"], dtype=bool)
    veg, veg_record = uniform_vegetation_fraction(rfid_dir, rfid_report, source_active, tuple(inputs.graph.shape))
    graph = inputs.graph
    pins_list = [Path(mah["xml_path"]), Path(veg_record["path"]), Path(verified.rainfall_path), *_tree_files(Path(verified.case_dir) / "syrup")]
    if kind == "chastre":
        pins_list += _tree_files(rfid_dir / "syrup")
    storm0 = initial_state(graph, inputs.depth0, inputs.soil0)
    record = {"xml": xml_record, "composition": {"vector": composition, **comp_check, "source": where},
              "vegetation": veg_record, "stormlength_s": inputs.stormlength_s,
              "hydrology_parameters": inputs.parameter_record,
              "rainfall_forcing": str(getattr(inputs.schedule, "provenance", None))}
    return LegacyCase(
        kind=kind, case_dir=Path(verified.case_dir), verified=verified, graph=graph, column=inputs.params, field=inputs.field,
        schedule=inputs.schedule, schedule_record={"kind": "case forcing (bound applied capture or staged legacy file)"},
        initial_storm=storm0, sediment=params, grid=physics_grid_from_graph(graph), vegetation=veg, holdings_kg=holdings,
        end_s=float(inputs.stormlength_s), default_bisection_iterations=64, record=record,
        pin_paths=tuple(dict.fromkeys(pins_list)),
        rainfall_scale=np.asarray(inputs.host["rainfall_scale"], dtype=np.float64), cell_area_m2=float(inputs.area))


def _plot1(verified: Any, args: dict[str, Any]) -> LegacyCase:
    from maple_syrup.benchmark_experiment import load_applied_rainfall
    from maple_syrup.sediment_experiment import prepare_verified_sediment_case

    prepared = prepare_verified_sediment_case(verified, end_s=args.get("end_s"))
    schedule, sched_record = prepared["schedule"], {"kind": "parsed legacy rainfall (interval-ending reading)"}
    if args.get("applied_rainfall"):
        schedule, sched_record = load_applied_rainfall(args["applied_rainfall"])
    mah = verified.report["mahleran"]
    params, xml_record = sediment_parameters_from_xml(prepared["xml_path"], mah["xml_sha256"], prepared["case"].config.grain_classes)
    if params.summary() != prepared["sediment"].summary():
        raise LegacyCaseError("generic XML parameters differ from the accepted Plot 1 parameter path")
    record = {"xml": xml_record, "composition": "actual MAPLE active layer of the verified Plot 1 case (per cell)",
              "vegetation": "Plot 1 sidecar vegetation_cover_fraction", "schedule": sched_record}
    return LegacyCase(
        kind="plot1", case_dir=Path(verified.case_dir), verified=verified, graph=prepared["graph"], column=prepared["column"],
        field=prepared["field"], schedule=schedule, schedule_record=dict(sched_record),
        initial_storm=prepared["state0"].storm, sediment=prepared["sediment"], grid=prepared["state0"].grid,
        vegetation=np.asarray(prepared["vegetation"], dtype=np.float64),
        holdings_kg=np.asarray(prepared["state0"].bed.active_layer.mass_kg, dtype=np.float64),
        end_s=float(prepared["end"]), default_bisection_iterations=40, record=record,
        rainfall_scale=np.asarray(prepared["host"]["rainfall_scale"], dtype=np.float64), cell_area_m2=float(prepared["area"]),
        pin_paths=tuple(dict.fromkeys([Path(prepared["xml_path"]), Path(verified.rainfall_path),
                                       *_tree_files(Path(verified.case_dir) / "syrup"),
                                       *([Path(args["applied_rainfall"])] if args.get("applied_rainfall") else [])])))


def legacy_case_for(kind: str, case_dir: str | Path, *, allow_maple_source_change: bool = False,
                    hash_only_tile_verify: bool = False, end_s: float | None = None,
                    applied_rainfall: str | Path | None = None) -> LegacyCase:
    """Verify the imported case and build its `LegacyCase`. `kind` in plot1 / rfid / chastre."""
    args = {"end_s": end_s, "applied_rainfall": applied_rainfall}
    if kind == "plot1":
        from maple_syrup.case_import import verify_plot1_case

        return _plot1(verify_plot1_case(case_dir, allow_maple_source_change=allow_maple_source_change), args)
    if kind == "rfid":
        from maple_syrup.rfid_case import verify_rfid_case

        return _rfid_family(kind, verify_rfid_case(case_dir, allow_maple_source_change=allow_maple_source_change), args)
    if kind == "chastre":
        from maple_syrup.chastre_case import verify_chastre_case

        verified = verify_chastre_case(case_dir, allow_maple_source_change=allow_maple_source_change,
                                       reload_tiles=not hash_only_tile_verify)
        return _rfid_family(kind, verified, args)
    raise LegacyCaseError(f"unknown case kind {kind!r}; use plot1, rfid or chastre")
