"""EXPERIMENTAL Chastre 1 m terrain with RFID_2014 settings: a WATER-ONLY, memory-bounded timing case (task chastre_timing).

What this is. The native 1 m Chastre DTM (`DTM_Chastre_PD4way.asc`, full interior, no crop/downsample/fill/carve/jitter) carries
the RFID_2014 hydrology UNCHANGED (captured applied forcing, fixed-Ksat Smith-Parlange columns, f = 40, theta, soil depth,
grains, bed settings), with the RFID rainfall-scaling map resized to the Chastre interior by nearest-pixel-centre mapping after
its masked source gaps are filled with the nearest valid source value. It exists to time the SYRUP water solvers (Numba CPU,
CUDA) on a large real terrain. It is NOT a Chastre flood prediction, rainfall is not Chastre rainfall, nothing was calibrated.

Terrain and routing. Active cells are exactly the DTM cells that are not nodata; the one-cell outer ring is nodata (required).
`routing.build_routing_graph` is called with its opt-in policies `allow_masked_nodata`, `allow_pit_storage` and
`allow_flat_storage`: the legacy strict `<` search (`topog_attrib.for` 94-117; nodata neighbours skipped) leaves a cell with
aspect 0 / slope 0 when no neighbour is strictly lower. Such cells (strict pits AND flats) are terminal STORAGE cells, conveyance 0,
no overtopping (iroute 6 is excluded), nothing filled/carved/perturbed/tie-routed. Nodata faces are closed and the ring is nodata,
so there are ZERO outlets: all water ends as infiltration, soil drainage, or surface storage (pits and in-transit water).

Bed. The sediment bed is REAL MAPLE state, compiled by `maple.case_tools.compilers.case_compiler.compile_case` in row-band tiles
that share ONE global datum offset, nz, voxel dz, active layer, bulk density, grain classes and composition (the bed of every
cell is column-local: the tiles equal the rows of the dense case, proved bitwise on small grids in tests/chastre). A dense
(1393, 1604, 306, 6) FP64 voxel array (~32.8 GB) is never built. Every band is instantiated, including bands of inactive
cells, which carry the RFID placeholder (`min active elevation`, invented, bed only). The persisted tile artifacts are hashed
as a manifest. The hydrology never receives bed arrays; `ChastreCase` is a deliberately small ADAPTER (geometry, class table,
zero water depth, tile manifest): NOT a MAPLE `CompiledCase`, not wind-ready, not evolving-bed-ready, and it has no voxel /
active-layer / availability attributes. The runtime bed guard (`RunGuard(bed_digest=chastre_bed_digest)`) re-hashes the persisted
tile files, so it reports artifact immutability, not an in-memory bed digest.

Nothing here was run by its author (file-only tools); Codex records results.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import gc
import hashlib
import json
import math
import os
import re
import resource
import shutil
import sys
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

import maple_syrup
from maple_syrup.case_import import (
    PHASE2_REQUIRED_MAPLE_API,
    BedPlan,
    Plot1Audit,
    Plot1ImportError,
    Plot1Recipe,
    VerifiedPlot1Case,
    _audit_geometry,
    _git_optional_locks_disabled,
    _json_safe,
    _refuse_output,
    _require,
    _require_same_state,
    _sha256_bytes,
    _stage_verbatim,
    _write_new_json,
    check_compiled_plot1,
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
from maple_syrup.rfid_case import (
    _ring_mask,
    _to_maple,
    _write_case_package,
    load_rfid_recipe,
    read_legacy_grid,
    verify_rfid_case,
)

__all__ = [
    "BedDefinition",
    "ChastreAudit",
    "ChastreCase",
    "ChastreRecipe",
    "audit_chastre",
    "chastre_bed_digest",
    "compile_tile",
    "define_bed",
    "fill_source_gaps",
    "generate_chastre_case",
    "load_chastre_recipe",
    "nearest_resize",
    "plan_only",
    "tile_bands",
    "verify_chastre_case",
]

RECIPE_SCHEMA = "maple_syrup.chastre_recipe.v1"
REPORT_SCHEMA = "maple_syrup.chastre_import_report.v1"
BINDING_SCHEMA = "maple_syrup.chastre_binding.v1"
SIDECAR_DIR = "syrup"
FIELDS_NAME = "chastre_fields.npz"
REPORT_NAME = "chastre_import_report.json"
BINDING_NAME = "chastre_binding.json"
FAILURE_NAME = "FAILED.json"
FORCING_NAME = "forcing/applied_forcing.json"  # the layout `rfid_case.rfid_schedule` reads
STAGED_TERRAIN_DIR = "source/mahleran_input_chastre"
TILES_DIR = "tiles"
HASH_CHUNK = 1 << 24
PEAK_FACTOR_ASSUMPTION = 10.0  # ASSUMED compile + load + check peak as a multiple of one tile's voxel array (a plan estimate)
DISK_FACTOR = 1.5  # plan-only refuses when free bytes < DISK_FACTOR x the estimated state size
_CLASS_COUNT = 6


# --------------------------------------------------------------------------------------------------------------------
# Streaming hashes (no whole-file reads)
# --------------------------------------------------------------------------------------------------------------------
def stream_sha256(path: str | Path) -> str:
    """SHA-256 of a file read in `HASH_CHUNK` pieces (bounded memory whatever the size)."""
    digest = hashlib.sha256()
    buffer = bytearray(HASH_CHUNK)
    view = memoryview(buffer)
    with Path(path).open("rb") as handle:
        while True:
            n = handle.readinto(buffer)
            if not n:
                break
            digest.update(view[:n])
    return digest.hexdigest()


def hash_tree(root: str | Path) -> dict[str, str]:
    """`{posix relative path: sha256}` of EVERY file under `root` (sorted; symlinks refused)."""
    root = Path(root)
    out: dict[str, str] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in sorted(filenames):
            path = Path(dirpath) / name
            if path.is_symlink():
                raise Plot1ImportError(f"symbolic link inside a hashed tile: {path}")
            out[path.relative_to(root).as_posix()] = stream_sha256(path)
    return out


def manifest_digest(tiles: list[dict[str, Any]], hashes: list[dict[str, str]] | None = None) -> str:
    """One digest over (tile index, relative path, sha256) of every tile file. `hashes` (parallel to `tiles`) overrides the
    bound `files` maps (used to digest the CURRENT files in the same way)."""
    digest = hashlib.sha256(b"maple_syrup.chastre_tiles.v1\n")
    for position, tile in enumerate(tiles):
        files = tile["files"] if hashes is None else hashes[position]
        for rel in sorted(files):
            digest.update(f"{tile['index']}:{rel}:{files[rel]}\n".encode())
    return digest.hexdigest()


def _peak_rss_kib() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)  # Linux: KiB, process-wide maximum so far


# --------------------------------------------------------------------------------------------------------------------
# Recipe
# --------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ChastreRecipe:
    raw: dict[str, Any]
    path: Path
    sha256: str

    @property
    def case_name(self) -> str:
        return self.raw["case_name"]

    @property
    def tile_rows(self) -> int:
        return int(self.raw["tiles"]["rows"])


def _keys(raw: Any, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(raw, dict) or set(raw) != keys:
        raise Plot1ImportError(f"recipe: {label} must be a mapping with exactly the keys {sorted(keys)}")
    return raw


def load_chastre_recipe(path: str | Path) -> ChastreRecipe:
    import yaml

    path = Path(path)
    payload = path.read_bytes()
    raw = yaml.safe_load(payload.decode("utf-8"))
    _keys(raw, {"schema", "case_name", "source_rfid", "terrain", "tiles", "rainfall_scaling"}, "the recipe")
    if raw["schema"] != RECIPE_SCHEMA:
        raise Plot1ImportError(f"recipe schema must be {RECIPE_SCHEMA!r}, got {raw['schema']!r}")
    if not isinstance(raw["case_name"], str) or not raw["case_name"]:
        raise Plot1ImportError("recipe: case_name must be a non-empty string")
    _keys(raw["source_rfid"], {"case_dir"}, "source_rfid")
    terrain = _keys(raw["terrain"], {"path", "sha256", "nrows", "ncols", "cellsize_m", "nodata_value", "expected_n_active",
                                     "expected_n_sinks", "expected_n_outlets", "expected_nz"}, "terrain")
    for key in ("nrows", "ncols", "expected_n_active", "expected_n_sinks", "expected_n_outlets", "expected_nz"):
        value = terrain[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise Plot1ImportError(f"recipe: terrain.{key} must be a non-negative integer")
    if terrain["nrows"] < 3 or terrain["ncols"] < 3:
        raise Plot1ImportError("recipe: terrain.nrows/ncols must be >= 3")
    if terrain["expected_nz"] < 1:
        raise Plot1ImportError("recipe: terrain.expected_nz must be a positive integer")
    if not (isinstance(terrain["sha256"], str) and re.fullmatch(r"[0-9a-f]{64}", terrain["sha256"])):
        raise Plot1ImportError("recipe: terrain.sha256 must be 64 lowercase hexadecimal characters")
    cellsize = terrain["cellsize_m"]
    if isinstance(cellsize, bool) or not isinstance(cellsize, (int, float)) or not math.isfinite(cellsize) or cellsize <= 0:
        raise Plot1ImportError("recipe: terrain.cellsize_m must be a finite number > 0")
    nodata = terrain["nodata_value"]
    if isinstance(nodata, bool) or not isinstance(nodata, (int, float)) or not math.isfinite(nodata) or nodata >= 0:
        raise Plot1ImportError("recipe: terrain.nodata_value must be a finite negative number")
    tiles = _keys(raw["tiles"], {"rows", "max_rss_gib"}, "tiles")
    if isinstance(tiles["rows"], bool) or not isinstance(tiles["rows"], int) or tiles["rows"] < 1:
        raise Plot1ImportError("recipe: tiles.rows must be a positive integer")
    limit = tiles["max_rss_gib"]
    if isinstance(limit, bool) or not isinstance(limit, (int, float)) or not math.isfinite(limit) or limit <= 0:
        raise Plot1ImportError("recipe: tiles.max_rss_gib must be a finite number > 0")
    scaling = _keys(raw["rainfall_scaling"], {"resize", "gap_fill"}, "rainfall_scaling")
    if scaling["resize"] != "nearest_pixel_centre" or scaling["gap_fill"] != "nearest_valid_euclid_first_row_major":
        raise Plot1ImportError("recipe: rainfall_scaling must be nearest_pixel_centre / nearest_valid_euclid_first_row_major")
    return ChastreRecipe(raw=raw, path=path.resolve(), sha256=_sha256_bytes(payload))


# --------------------------------------------------------------------------------------------------------------------
# Rainfall-scaling gap fill and nearest-pixel-centre resize (pure NumPy, deterministic)
# --------------------------------------------------------------------------------------------------------------------
def fill_source_gaps(scale: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Fill every cell where `valid` is False with the value of the nearest valid cell: squared Euclidean distance between
    PIXEL CENTRES in source index space; ties go to the FIRST tied valid cell in row-major order. Valid cells keep their exact
    values; the new values come only from valid source cells. Returns `(filled, record)`."""
    scale = np.asarray(scale, dtype=np.float64)
    valid = np.asarray(valid, dtype=np.bool_)
    if scale.shape != valid.shape or scale.ndim != 2:
        raise Plot1ImportError("gap fill: scale and valid must be 2-D arrays of one shape")
    if not valid.any():
        raise Plot1ImportError("gap fill: no valid source cell")
    vr, vc = np.nonzero(valid)  # row-major order
    vr, vc = vr.astype(np.int64), vc.astype(np.int64)
    filled = scale.copy()
    ties = 0
    missing = np.argwhere(~valid)
    for r, c in missing:
        d2 = (vr - int(r)) ** 2 + (vc - int(c)) ** 2
        j = int(np.argmin(d2))  # first minimum: the first tied valid cell in row-major order
        ties += int(np.count_nonzero(d2 == d2[j]) > 1)
        filled[int(r), int(c)] = scale[vr[j], vc[j]]
    return filled, {"n_filled": int(missing.shape[0]), "n_filled_with_distance_ties": ties,
                    "distance": "squared Euclidean between pixel centres in source index space",
                    "tie_rule": "first tied valid cell in row-major (south-first array) order"}


def nearest_index(n_src: int, n_dst: int) -> np.ndarray:
    """Source index of each target pixel under nearest-PIXEL-CENTRE mapping: target centre (i + 1/2) / n_dst of the extent maps
    to source floor((i + 1/2) * n_src / n_dst) = (2 i + 1) * n_src // (2 n_dst) (exact integer arithmetic, always < n_src).
    When a target centre falls EXACTLY on a source pixel boundary the HIGHER source index is taken (the floor convention), so
    the map is not mirror-symmetric at such ties. The RFID rows 104 -> 1393 have exactly one (target row 696, the centre row);
    the columns 58 -> 1604 have none. The convention is defined on the SOUTH-FIRST array used throughout this case."""
    if n_src < 1 or n_dst < 1:
        raise Plot1ImportError("resize: sizes must be positive")
    return ((2 * np.arange(n_dst, dtype=np.int64) + 1) * n_src) // (2 * n_dst)


def nearest_resize(array: np.ndarray, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nearest-pixel-centre resize of a 2-D array in the orientation it is given (here south-first); see `nearest_index` for
    the exact-tie convention. No interpolation: every output value is an input value. Returns `(resized, row_idx, col_idx)`."""
    array = np.asarray(array)
    rows = nearest_index(array.shape[0], shape[0])
    cols = nearest_index(array.shape[1], shape[1])
    return np.ascontiguousarray(array[rows][:, cols]), rows, cols


# --------------------------------------------------------------------------------------------------------------------
# Tiled bed (actual MAPLE compile / load / check, one band at a time)
# --------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class BedDefinition:
    """The GLOBAL bed definition every tile shares: recipe values (global legacy shape), the global `BedPlan` (datum offset,
    nz, extent, expected elevation), the bed DEM (placeholder-filled) and the single composition."""

    base: Plot1Recipe
    bed: BedPlan
    bed_dem: np.ndarray  # (ny, nx) legacy-datum metres, finite everywhere
    table: np.ndarray  # (1, 6) composition table
    particle_density_kg_m3: float


def define_bed(base: Plot1Recipe, bed_dem: np.ndarray, table: np.ndarray, particle_density_kg_m3: float) -> BedDefinition:
    bed_dem = np.ascontiguousarray(bed_dem, dtype=np.float64)
    if bed_dem.shape != base.interior_shape:
        raise Plot1ImportError(f"bed DEM shape {bed_dem.shape} != the recipe interior {base.interior_shape}")
    table = np.ascontiguousarray(table, dtype=np.float64)
    if table.shape != (1, _CLASS_COUNT):
        raise Plot1ImportError(f"this case supports ONE composition row of {_CLASS_COUNT} classes, got {table.shape}")
    return BedDefinition(base=base, bed=plan_bed(bed_dem, base), bed_dem=bed_dem, table=table,
                         particle_density_kg_m3=float(particle_density_kg_m3))


def _require_consistent(definition: BedDefinition) -> None:
    """The bed plan must be exactly what MAPLE's own `datum_offset` arithmetic gives for the bed DEM, with the allocated column
    holding it: a tile compiled against a different datum/nz would be a different bed from the global one."""
    base, bed = definition.base, definition.bed
    expected = definition.bed_dem - 0.0 + bed.datum_offset_m
    _require(np.array_equal(bed.elevation_m, expected), "global bed plan elevation differs from bed DEM + datum offset")
    _require(bed.voxel_dz_m == base.voxel_dz_m and bed.nz * bed.voxel_dz_m == bed.vertical_extent_m,
             "global bed plan voxel allocation is inconsistent")
    _require(float(expected.min()) >= base.minimum_fill_depth_m - 1e-9, "global datum leaves less than the minimum fill depth")
    _require(bed.vertical_extent_m - float(expected.max()) >= base.headroom_m - 1e-9, "global nz leaves less than the headroom")


def tile_bands(ny: int, rows: int) -> list[tuple[int, int]]:
    """Half-open MAPLE row ranges `[r0, r1)` (row 0 = south) covering `ny` rows in bands of at most `rows`."""
    if ny < 1 or rows < 1:
        raise Plot1ImportError("tile_bands needs positive sizes")
    return [(r0, min(r0 + rows, ny)) for r0 in range(0, ny, rows)]


def tile_audit(definition: BedDefinition, index: int, r0: int, r1: int, tile_dir: Path) -> Plot1Audit:
    """A `Plot1Audit` for rows `[r0, r1)` carrying the GLOBAL datum/nz: reused unchanged by `rfid_case._write_case_package` and
    `case_import.check_compiled_plot1`."""
    ny, nx = definition.base.interior_shape
    if not 0 <= r0 < r1 <= ny:
        raise Plot1ImportError(f"tile rows [{r0}, {r1}) outside [0, {ny})")
    height = r1 - r0
    base = dataclasses.replace(definition.base, case_name=f"{definition.base.case_name}_tile{index:03d}",
                               legacy_nrows=height + 2, legacy_ncols=nx + 2)
    bed = dataclasses.replace(definition.bed, elevation_m=np.ascontiguousarray(definition.bed.elevation_m[r0:r1]))
    final = np.ascontiguousarray(np.broadcast_to(definition.table[0], (height, nx, _CLASS_COUNT)))
    return Plot1Audit(
        recipe=base, case_dir=Path(tile_dir), report={}, fields={},
        elevation_source_m=np.ascontiguousarray(definition.bed_dem[r0:r1]), final_fractions=final,
        particle_density_kg_m3=definition.particle_density_kg_m3, bed=bed,
        composition_codes=np.ones((height, nx), dtype=np.int32), composition_table=definition.table,
        dem_staged_relpath="source/syrup_derived/elevation_bed_placeholder.npy")


def compile_tile(definition: BedDefinition, index: int, r0: int, r1: int, tile_dir: str | Path, *, keep: bool = False,
                 ) -> tuple[dict[str, Any], Any]:
    """Compile rows `[r0, r1)` with the ACTUAL MAPLE compiler into the NEW `tile_dir`, reload it, run `check_compiled_plot1`
    on both and require identical state, then hash every persisted file. Returns `(record, loaded_or_None)`; the loaded case
    is returned only when `keep` (tests); otherwise nothing of the tile stays in memory. A failure leaves the partial tile."""
    from maple.case_tools.compilers.case_compiler import compile_case as maple_compile
    from maple.case_tools.compilers.case_compiler import load_compiled_case

    _require_consistent(definition)
    tile_dir = Path(tile_dir)
    if tile_dir.exists():
        raise Plot1ImportError(f"refusing to write into existing tile {tile_dir}")
    audit = tile_audit(definition, index, r0, r1, tile_dir)
    tile_dir.mkdir(parents=True)
    package = _write_case_package(audit)
    t0 = time.perf_counter()
    with _git_optional_locks_disabled():
        compiled = maple_compile(tile_dir)
    compile_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    compiled_checks = check_compiled_plot1(compiled, audit)
    loaded = load_compiled_case(tile_dir)
    loaded_checks = check_compiled_plot1(loaded, audit)
    _require_same_state(compiled, loaded)
    load_check_s = time.perf_counter() - t1
    provenance = compiled.provenance_record
    record = {
        "index": index, "row_start": r0, "row_stop": r1, "rows": r1 - r0, "dir": f"{TILES_DIR}/tile_{index:03d}",
        "case_identity_sha256": provenance["case_identity_sha256"], "artifact_sha256": provenance["artifact_sha256"],
        "maple_code_version": provenance["code_version"], "package": package, "nz": definition.bed.nz,
        "datum_offset_m": definition.bed.datum_offset_m, "compile_s": compile_s, "load_and_check_s": load_check_s,
        "process_peak_rss_kib_after": _peak_rss_kib(),
        "checks": {"compiled": compiled_checks, "reloaded": loaded_checks, "compiled_equals_reloaded": True},
    }
    del compiled
    if not keep:
        del loaded
        loaded = None
    gc.collect()
    record["files"] = hash_tree(tile_dir)
    return record, loaded


# --------------------------------------------------------------------------------------------------------------------
# Global audit
# --------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ChastreAudit:
    definition: BedDefinition
    fields: dict[str, np.ndarray]
    report: dict[str, Any]


def _read_terrain(path: Path, terrain: dict[str, Any]):
    header, body = read_legacy_grid(path)
    nrows, ncols = int(terrain["nrows"]), int(terrain["ncols"])
    if (int(header["nrows"]), int(header["ncols"])) != (nrows, ncols):
        raise Plot1ImportError(f"terrain header {int(header['nrows'])} x {int(header['ncols'])} != recipe {nrows} x {ncols}")
    if header.get("cellsize") != float(terrain["cellsize_m"]):
        raise Plot1ImportError(f"terrain cellsize {header.get('cellsize')!r} != recipe {terrain['cellsize_m']!r}")
    if header.get("nodata_value") != float(terrain["nodata_value"]):
        raise Plot1ImportError(f"terrain nodata {header.get('nodata_value')!r} != recipe {terrain['nodata_value']!r}")
    return header, _to_maple(body)


def audit_chastre(recipe: ChastreRecipe, terrain_path: str | Path, source: VerifiedPlot1Case, *,
                  staged_record: dict[str, Any] | None = None) -> ChastreAudit:
    """Audit the terrain and the verified RFID source: build the global bed definition, the resized rainfall scaling, the
    routing graph with flat/pit storage and the sidecar arrays and report. Reads only. Raises on any disagreement."""
    from maple_syrup.routing import build_routing_graph

    raw = recipe.raw
    terrain = raw["terrain"]
    terrain_path = Path(terrain_path)
    nodata = float(terrain["nodata_value"])
    _header, dem_sf = _read_terrain(terrain_path, terrain)
    ring = _ring_mask(dem_sf.shape)
    if not np.all(dem_sf[ring] == nodata):
        raise Plot1ImportError("the terrain ring must be nodata everywhere (zero-outlet policy); a valid ring cell exists")
    interior = np.ascontiguousarray(dem_sf[1:-1, 1:-1])
    active = interior != nodata
    if not np.all(interior[active] > nodata):
        raise Plot1ImportError("an interior elevation is below the nodata sentinel")
    n_active = int(active.sum())
    if n_active != terrain["expected_n_active"]:
        raise Plot1ImportError(f"{n_active} active cells, the recipe expects {terrain['expected_n_active']}")

    # --- source RFID (verified by the caller): hydrology report, composition, scaling map, density ---
    sreport, sfields = source.report, source.fields
    src_recipe = load_rfid_recipe(sreport["recipe"]["recipe_path"], mahleran_root=sreport["mahleran"]["root"])
    ny, nx = interior.shape
    base = dataclasses.replace(
        src_recipe.base, case_name=recipe.case_name, recipe_path=recipe.path, recipe_sha256=recipe.sha256,
        legacy_nrows=ny + 2, legacy_ncols=nx + 2, cellsize_m=float(terrain["cellsize_m"]))
    table = np.asarray(sfields["composition_table"], dtype=np.float64)
    density = float(sreport["bed"]["particle_density_kg_m3"])
    placeholder = float(interior[active].min())
    bed_dem = np.where(active, interior, placeholder)
    definition = define_bed(base, bed_dem, table, density)
    if definition.bed.nz != terrain["expected_nz"]:
        raise Plot1ImportError(f"global nz {definition.bed.nz} != the recipe's {terrain['expected_nz']}")
    _require_consistent(definition)

    # --- rainfall scaling: fill source gaps, nearest-pixel-centre resize, no renormalization ---
    src_active = np.asarray(sfields["active"], dtype=np.bool_)
    src_scale = np.asarray(sfields["rainfall_scaling"], dtype=np.float64)
    if not (src_scale[src_active] > 0.0).all():
        raise Plot1ImportError("a valid RFID source cell has a non-positive rainfall scale")
    filled, fill_record = fill_source_gaps(src_scale, src_active)
    resized, row_index, col_index = nearest_resize(filled, (ny, nx))
    scale = np.where(active, resized, 0.0)
    if not (scale[active] > 0.0).all():
        raise Plot1ImportError("an active Chastre cell received a non-positive rainfall scale")
    if not np.isin(resized, filled).all():
        raise Plot1ImportError("internal error: the resized scaling holds a value that is not a source value")

    # --- routing graph: native 1 m, ring nodata, flat + strict sinks as terminal storage ---
    friction = np.full((ny, nx), float(sreport["hydrology"]["friction_factor"]))
    graph = build_routing_graph(dem_sf, ring, friction, float(terrain["cellsize_m"]), active_mask=active, nodata_value=nodata,
                                allow_masked_nodata=True, allow_pit_storage=True, allow_flat_storage=True)
    n_sinks = int(np.sum(graph.pit_storage))
    n_flat = int(np.sum(graph.flat_storage))
    n_outlets = int(np.sum(graph.outlet))
    if n_sinks != terrain["expected_n_sinks"] or n_outlets != terrain["expected_n_outlets"]:
        raise Plot1ImportError(f"topology differs from the recipe: sinks {n_sinks} (expected {terrain['expected_n_sinks']}), "
                               f"outlets {n_outlets} (expected {terrain['expected_n_outlets']})")
    summary = graph.summary()

    fields = {
        "legacy_full_elevation_m": np.ascontiguousarray(dem_sf),
        "active": np.ascontiguousarray(active),
        "inactive_interior": np.ascontiguousarray(~active),
        "rainfall_scaling": np.ascontiguousarray(scale),
        "rainfall_scaling_source_filled": np.ascontiguousarray(filled),
        "rainfall_scaling_source_valid": np.ascontiguousarray(src_active),
        "resize_row_index": row_index, "resize_col_index": col_index,
        "graph_aspect": np.ascontiguousarray(graph.aspect), "graph_slope": np.ascontiguousarray(graph.slope),
        "graph_pit_storage": np.ascontiguousarray(graph.pit_storage),
        "graph_flat_storage": np.ascontiguousarray(graph.flat_storage),
        "graph_outlet": np.ascontiguousarray(graph.outlet),
    }
    levels = [b - a for a, b in zip(graph.level_bounds[:-1], graph.level_bounds[1:], strict=True)]
    report = {
        "schema": REPORT_SCHEMA, "maple_syrup_version": maple_syrup.__version__,
        "recipe": {"recipe_path": str(recipe.path), "recipe_sha256": recipe.sha256, "authored": raw},
        "mahleran": {"root": sreport["mahleran"]["root"],
                     "note": "root of the RFID inputs (bound through source_rfid); the terrain is the staged DTM below"},
        "terrain_source": staged_record if staged_record is not None else {"file": terrain_path.name},
        "source_rfid": {"case_dir": str(source.case_dir), "binding_sha256": source.checks["binding_sha256"],
                        "report_sha256": source.binding["rfid_report_sha256"],
                        "fields_sha256": source.binding["rfid_fields_sha256"],
                        "applied_forcing_sha256": source.binding["applied_forcing_sha256"],
                        "case_identity_sha256": source.binding["maple_case_identity_sha256"]},
        "legacy_options": sreport["legacy_options"],
        "grid": {"legacy_shape": [ny + 2, nx + 2], "maple_shape": [ny, nx], "cellsize_m": float(terrain["cellsize_m"]),
                 "nodata_value": nodata, "n_active": n_active, "n_inactive_interior": int((~active).sum()),
                 "ring_nodata_cells": int(ring.sum()),
                 "elevation_range_active_m": [float(interior[active].min()), float(interior[active].max())],
                 "orientation": "ESRI file north-first; every array here is MAPLE south-first (row 0 = south)"},
        "hydrology": sreport["hydrology"], "forcing": sreport["forcing"],
        "rainfall_scaling": {
            "source": "RFID rainfall-scaling map interior (south-first), valid cells only; values unchanged",
            "source_shape": list(src_scale.shape), "target_shape": [ny, nx], "gap_fill": fill_record,
            "resize": "nearest pixel centre: source = (2 i + 1) * n_src // (2 n_dst) per axis; no interpolation",
            "global_renormalization": "none", "ksat_recalculation": "none (K copied from the RFID report)",
            "target_nodata_value": 0.0,
            "distinct_values_source": int(np.unique(filled).size), "distinct_values_target": int(np.unique(resized).size),
            "value_range_active": [float(scale[active].min()), float(scale[active].max())],
            "disclosure": f"the RFID pattern is stretched ~{ny / src_scale.shape[0]:.1f}x (rows) and "
                          f"~{nx / src_scale.shape[1]:.1f}x (columns); it is not Chastre rainfall"},
        "placeholder_terrain": {
            "rule": "min_active_elevation", "value_m": placeholder, "cells": int((~active).sum()),
            "scope": "MAPLE bed cells with no terrain ONLY (nodata cells); excluded from every hydraulic computation; the "
                     "original DEM with its nodata is kept in the sidecar",
            "maple_mask_limitation": "supplied as an already-filled prepared array: the compiled MAPLE masks do NOT distinguish "
                                     "these cells; the authoritative record is `inactive_interior` in the sidecar"},
        "terrain": {"summary": summary, "n_pit_storage": n_sinks, "n_flat_storage": n_flat, "n_strict_pit_storage": n_sinks - n_flat,
                    "n_outlets": n_outlets, "allow_flat_storage": True, "level_widths": levels, "n_levels": len(levels),
                    "graph_input_sha256": graph.input_sha256, "graph_policy": graph.policy,
                    "boundary_policy": "ring is nodata: never a receiver, faces closed, ZERO outlets; water stays on the surface "
                                       "in storage cells (no overtopping, iroute 6 excluded), infiltrates or drains from the soil",
                    "routing_reference": "topog_attrib.for 94-117 (strict '<', nodata neighbours skipped)"},
        "grain_maps": {"composition": [float(v) for v in table[0]], "source": "verified RFID composition table"},
        "bed": {**definition.bed.as_record(), "bulk_density_kg_m3": base.bulk_density_kg_m3,
                "active_layer_thickness_m": base.active_layer_thickness_m, "particle_density_kg_m3": density,
                "vertical_composition": "homogeneous per cell: one composition fills every column",
                "tiles": "real MAPLE compile of row bands sharing this datum/nz; see the binding"},
        "not_done": ["no sediment transport / splash / erosion / wind", "no evolving terrain", "no restart",
                     "not a MAPLE CompiledCase: not wind-ready, not evolving-bed-ready",
                     "not a Chastre flood prediction; rainfall is the stretched RFID pattern"],
    }
    return ChastreAudit(definition=definition, fields=fields, report=report)


# --------------------------------------------------------------------------------------------------------------------
# Plan, disk and memory estimates
# --------------------------------------------------------------------------------------------------------------------
def _existing_ancestor(path: Path) -> Path:
    path = path.resolve()
    while not path.exists():
        path = path.parent
    return path


def estimates(ny: int, nx: int, nz: int, tile_rows: int) -> dict[str, Any]:
    voxel_bytes = ny * nx * nz * _CLASS_COUNT * 8
    tile_voxel = min(tile_rows, ny) * nx * nz * _CLASS_COUNT * 8
    return {"dense_voxel_bytes_NOT_BUILT": voxel_bytes, "tile_voxel_bytes": tile_voxel,
            "estimated_state_bytes": int(1.1 * voxel_bytes) + 400_000_000,
            "assumed_compile_peak_bytes_per_tile": int(PEAK_FACTOR_ASSUMPTION * tile_voxel),
            "note": f"ESTIMATES (voxel array x 1.1 for the state, x {PEAK_FACTOR_ASSUMPTION:.0f} for the compile peak), "
                    "not measurements"}


def plan_only(recipe: ChastreRecipe, output_dir: str | Path, *, tile_rows: int | None = None) -> dict[str, Any]:
    """Print-only plan: reads the DTM (pin-checked) and the RFID report JSON, computes nz/bands/estimates and the free bytes of
    the output filesystem. Writes nothing. `refuse` is True when free < DISK_FACTOR x the estimated state."""
    terrain = recipe.raw["terrain"]
    path = Path(terrain["path"])
    digest = stream_sha256(path)
    if digest != terrain["sha256"]:
        raise Plot1ImportError(f"terrain sha256 {digest} != the recipe pin {terrain['sha256']}")
    _header, dem_sf = _read_terrain(path, terrain)
    nodata = float(terrain["nodata_value"])
    interior = dem_sf[1:-1, 1:-1]
    active = interior != nodata
    sreport = json.loads((Path(recipe.raw["source_rfid"]["case_dir"]) / SIDECAR_DIR / "rfid_import_report.json").read_text())
    src_recipe = load_rfid_recipe(sreport["recipe"]["recipe_path"], mahleran_root=sreport["mahleran"]["root"])
    ny, nx = interior.shape
    base = dataclasses.replace(src_recipe.base, legacy_nrows=ny + 2, legacy_ncols=nx + 2,
                               cellsize_m=float(terrain["cellsize_m"]))
    bed = plan_bed(np.where(active, interior, interior[active].min()), base)
    rows = tile_rows if tile_rows is not None else recipe.tile_rows
    bands = tile_bands(ny, rows)
    est = estimates(ny, nx, bed.nz, rows)
    free = shutil.disk_usage(_existing_ancestor(Path(output_dir))).free
    return {"interior_shape": [ny, nx], "n_active": int(active.sum()), "nz": bed.nz, "datum_offset_m": bed.datum_offset_m,
            "n_bands": len(bands), "tile_rows": rows, "terrain_sha256": digest, **est, "free_bytes": int(free),
            "required_free_bytes": int(DISK_FACTOR * est["estimated_state_bytes"]),
            "refuse": bool(free < DISK_FACTOR * est["estimated_state_bytes"]),
            "max_rss_gib_limit": recipe.raw["tiles"]["max_rss_gib"]}


# --------------------------------------------------------------------------------------------------------------------
# Generate
# --------------------------------------------------------------------------------------------------------------------
def _write_sidecar(case_dir: Path, audit: ChastreAudit) -> dict[str, str]:
    directory = case_dir / SIDECAR_DIR
    directory.mkdir(parents=True, exist_ok=True)
    fields_path = directory / FIELDS_NAME
    with fields_path.open("xb") as handle:
        np.savez(handle, **audit.fields)
    fields_sha = stream_sha256(fields_path)
    report = dict(audit.report)
    report["sidecar_fields"] = {"path": f"{SIDECAR_DIR}/{FIELDS_NAME}", "sha256": fields_sha,
                                "arrays": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in audit.fields.items()}}
    return {"fields_sha256": fields_sha, "report_sha256": _write_new_json(directory / REPORT_NAME, report)}


def generate_chastre_case(recipe: ChastreRecipe, output_dir: str | Path, *, source_case_dir: str | Path | None = None,
                          expected_maple_root: str | Path | None = None, allow_maple_source_change: bool = False,
                          tile_rows: int | None = None, progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Generate the tiled case into the NEW `output_dir`. A failure after the directory exists leaves `syrup/FAILED.json`
    (partial tiles preserved, no binding). Only a directory with `syrup/chastre_binding.json` is a completed import."""
    log = progress or (lambda message: None)
    started = time.time()
    dependency = resolve_maple_dependency(expected_maple_root)
    check_required_api(PHASE2_REQUIRED_MAPLE_API)
    maple_provenance = capture_maple_provenance(dependency)
    raw = recipe.raw
    rows = recipe.tile_rows if tile_rows is None else int(tile_rows)
    source_dir = Path(source_case_dir or raw["source_rfid"]["case_dir"]).resolve()
    log(f"verifying the RFID source case {source_dir}")
    source = verify_rfid_case(source_dir, expected_maple_root=expected_maple_root,
                              allow_maple_source_change=allow_maple_source_change)
    _require("applied_forcing_sha256" in source.binding, "the RFID source has no bound applied forcing")
    terrain = raw["terrain"]
    terrain_path = Path(terrain["path"])
    digest = stream_sha256(terrain_path)
    _require(digest == terrain["sha256"], f"terrain sha256 {digest} != the recipe pin {terrain['sha256']}")
    output_dir = Path(output_dir).resolve()
    _refuse_output(output_dir, {"MAPLE": dependency.source_root, "MAHLERAN": Path(source.report["mahleran"]["root"]),
                                "recipe": recipe.path.parent, "RFID source": source_dir})
    ny, nx = int(terrain["nrows"]) - 2, int(terrain["ncols"]) - 2
    est = estimates(ny, nx, int(terrain["expected_nz"]), rows)
    free = shutil.disk_usage(_existing_ancestor(output_dir.parent)).free
    _require(free >= DISK_FACTOR * est["estimated_state_bytes"],
             f"free disk {free} B < {DISK_FACTOR} x the estimated state {est['estimated_state_bytes']} B")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    try:
        staged = _stage_verbatim(terrain_path, output_dir / STAGED_TERRAIN_DIR / terrain_path.name)
        _require(staged["sha256"] == terrain["sha256"], "staged terrain hash differs from the pin")
        forcing_src = source_dir / SIDECAR_DIR / FORCING_NAME
        forcing_bytes = forcing_src.read_bytes()
        _require(_sha256_bytes(forcing_bytes) == source.binding["applied_forcing_sha256"], "source forcing differs from its binding")
        forcing_dst = output_dir / SIDECAR_DIR / FORCING_NAME
        forcing_dst.parent.mkdir(parents=True, exist_ok=True)
        with forcing_dst.open("xb") as handle:
            handle.write(forcing_bytes)
        log("auditing terrain, resizing the rainfall scaling, building the routing graph")
        audit = audit_chastre(recipe, output_dir / STAGED_TERRAIN_DIR / terrain_path.name, source, staged_record=staged)
        sidecar = _write_sidecar(output_dir, audit)
        definition = audit.definition
        bands = tile_bands(ny, rows)
        limit_kib = float(raw["tiles"]["max_rss_gib"]) * 1024.0 * 1024.0
        tiles: list[dict[str, Any]] = []
        for index, (r0, r1) in enumerate(bands):
            log(f"tile {index + 1}/{len(bands)} rows [{r0}, {r1}) compile")
            record, _ = compile_tile(definition, index, r0, r1, output_dir / TILES_DIR / f"tile_{index:03d}")
            tiles.append(record)
            log(f"tile {index + 1}/{len(bands)} done: compile {record['compile_s']:.1f} s, load+check "
                f"{record['load_and_check_s']:.1f} s, process peak RSS {record['process_peak_rss_kib_after'] / 1048576:.2f} GiB")
            if record["process_peak_rss_kib_after"] > limit_kib:
                raise RuntimeError(f"process peak RSS {record['process_peak_rss_kib_after']} KiB exceeds the recipe limit "
                                   f"{raw['tiles']['max_rss_gib']} GiB after tile {index}; stopping")
        digest_after = source_tree_digest(dependency.package_dir).digest_sha256
        _require(digest_after == maple_provenance["package_source_digest"]["digest_sha256"], "MAPLE source changed during the import")
        binding = {
            "schema": BINDING_SCHEMA, "status": "ok", "case_dir": str(output_dir), "recipe_sha256": recipe.sha256,
            "terrain_sha256": terrain["sha256"], "staged_terrain_sha256": staged["sha256"],
            "source_rfid": audit.report["source_rfid"],
            "fields_sha256": sidecar["fields_sha256"], "report_sha256": sidecar["report_sha256"],
            "applied_forcing_sha256": _sha256_bytes(forcing_bytes),
            "grid": {"ny": ny, "nx": nx, "nz": definition.bed.nz, "datum_offset_m": definition.bed.datum_offset_m},
            "tile_rows": rows, "tiles": tiles, "tiles_digest": manifest_digest(tiles),
            "tile_totals": {"n_tiles": len(tiles), "compile_s": sum(t["compile_s"] for t in tiles),
                            "load_and_check_s": sum(t["load_and_check_s"] for t in tiles),
                            "process_peak_rss_kib": max(t["process_peak_rss_kib_after"] for t in tiles),
                            "rss_note": "process-wide maximum (includes the audit and graph build), measured by getrusage"},
            "maple": maple_provenance, "maple_source_digest_after": digest_after, "maple_source_stable": True,
            "maple_syrup": capture_syrup_provenance(), "environment": environment_record(),
            "started_unix_s": started, "finished_unix_s": time.time(),
            "not_done": ["no wind run", "no sediment/erosion", "no restart", "no GPU claim",
                         "adapter is not a MAPLE CompiledCase"],
        }
        binding_sha = _write_new_json(output_dir / SIDECAR_DIR / BINDING_NAME, binding)
        return {"status": "ok", "case_dir": str(output_dir), "binding_sha256": binding_sha, "nz": definition.bed.nz,
                "n_tiles": len(tiles), "tiles_digest": binding["tiles_digest"], "tile_totals": binding["tile_totals"]}
    except BaseException as exc:
        with contextlib.suppress(Exception):
            _write_new_json(output_dir / SIDECAR_DIR / FAILURE_NAME, {
                "status": "failed", "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()})
        raise


# --------------------------------------------------------------------------------------------------------------------
# Adapter and runtime bed guard
# --------------------------------------------------------------------------------------------------------------------
class ChastreCase:
    """The small case object the hydrology input factory reads (`rfid_case.rfid_inputs`): `config.geometry` (a MAPLE
    `GeometrySpec` of the full interior), `config.grain_classes` (from tile 0), `water.depth_m` (zero, host). It is NOT a MAPLE
    `CompiledCase`, holds NO bed arrays and deliberately has no `voxel_column`, `active_layer` or `sediment_availability`:
    anything that needs a bed fails with AttributeError instead of silently using a fake one."""

    def __init__(self, case_dir: Path, geometry: Any, grain_classes: Any, tiles: list[dict[str, Any]], bound_tiles_digest: str,
                 progress: Callable[[str], None] | None = None) -> None:
        self.case_dir = Path(case_dir)
        self.config = SimpleNamespace(geometry=geometry, grain_classes=grain_classes)
        self.water = SimpleNamespace(depth_m=np.zeros((geometry.ny, geometry.nx), dtype=np.float64))
        self.tiles = tuple(tiles)
        self.bound_tiles_digest = bound_tiles_digest
        self._progress = progress

    def current_hashes(self) -> list[dict[str, str]]:
        out = []
        for position, tile in enumerate(self.tiles):
            if self._progress is not None:
                self._progress(f"hashing tile {position + 1}/{len(self.tiles)}")
            tile_dir = self.case_dir / tile["dir"]
            if not tile_dir.is_dir():
                raise Plot1ImportError(f"persisted tile {tile['dir']} is missing")
            out.append(hash_tree(tile_dir))
        return out

    def persisted_digest(self) -> str:
        """Stream-hash EVERY file of every persisted tile and digest (tile, path, hash). Not a hash of in-memory arrays."""
        return manifest_digest(list(self.tiles), self.current_hashes())

    def __repr__(self) -> str:
        g = self.config.geometry
        return f"ChastreCase({g.ny} x {g.nx}, {len(self.tiles)} persisted tiles, no bed arrays)"


def chastre_bed_digest(case: ChastreCase) -> str:
    """`RunGuard(bed_digest=...)` callback: the persisted-tile digest."""
    return case.persisted_digest()


# --------------------------------------------------------------------------------------------------------------------
# Verify
# --------------------------------------------------------------------------------------------------------------------
def verify_chastre_case(case_dir: str | Path, *, expected_maple_root: str | Path | None = None,
                        allow_maple_source_change: bool = False, reload_tiles: bool = True,
                        progress: Callable[[str], None] | None = None) -> VerifiedPlot1Case:
    """Re-verify a completed Chastre import (reads only; one tile in memory at a time). Every bound file is stream-hashed; the
    RFID source case is re-verified and its pins compared; the audit (terrain, gap fill, resize, routing graph, bed plan) is
    RE-COMPUTED from the staged DTM and the verified source and every sidecar array and report section must agree; every
    tile's files must equal the bound manifest exactly (no extra or missing file) and, with `reload_tiles`, each tile is
    reloaded by MAPLE and passes `check_compiled_plot1` (tile 0 is always loaded: it supplies the class table)."""
    from maple.case_tools.compilers.case_compiler import load_compiled_case

    log = progress or (lambda message: None)
    case_dir = Path(case_dir).resolve()
    sidecar = case_dir / SIDECAR_DIR
    binding_path = sidecar / BINDING_NAME
    _require(binding_path.is_file(), f"{binding_path} missing: not a completed Chastre import")
    _require(not (sidecar / FAILURE_NAME).exists(), f"{sidecar / FAILURE_NAME} exists: the import failed")
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    _require(binding.get("schema") == BINDING_SCHEMA and binding.get("status") == "ok", "not a completed Chastre import")
    bound_paths = {"report_sha256": sidecar / REPORT_NAME, "fields_sha256": sidecar / FIELDS_NAME,
                   "applied_forcing_sha256": sidecar / FORCING_NAME,
                   "staged_terrain_sha256": next((case_dir / STAGED_TERRAIN_DIR).glob("*"), case_dir / "missing")}
    for key, path in bound_paths.items():
        _require(path.is_file() and stream_sha256(path) == binding[key], f"{path.name} does not match binding {key}")
    report = json.loads((sidecar / REPORT_NAME).read_text(encoding="utf-8"))
    with np.load(sidecar / FIELDS_NAME, allow_pickle=False) as data:
        fields = {name: data[name] for name in data.files}

    dependency = resolve_maple_dependency(expected_maple_root)
    check_required_api(PHASE2_REQUIRED_MAPLE_API)
    maple_record = capture_maple_provenance(dependency)
    now = maple_record["package_source_digest"]["digest_sha256"]
    bound = binding["maple"]["package_source_digest"]["digest_sha256"]
    _require(allow_maple_source_change or now == bound, f"MAPLE source digest {now} differs from the bound {bound}")

    recipe = load_chastre_recipe(report["recipe"]["recipe_path"])
    _require(recipe.sha256 == report["recipe"]["recipe_sha256"] == binding["recipe_sha256"], "recipe changed since the import")
    terrain = recipe.raw["terrain"]
    staged = bound_paths["staged_terrain_sha256"]
    _require(binding["terrain_sha256"] == terrain["sha256"] == binding["staged_terrain_sha256"], "terrain pin differs from the binding")

    pins = binding["source_rfid"]
    source_dir = Path(pins["case_dir"])
    log(f"re-verifying the RFID source case {source_dir}")
    source = verify_rfid_case(source_dir, expected_maple_root=expected_maple_root,
                              allow_maple_source_change=allow_maple_source_change)
    _require(source.checks["binding_sha256"] == pins["binding_sha256"], "RFID source binding changed")
    _require(source.binding["rfid_report_sha256"] == pins["report_sha256"], "RFID source report changed")
    _require(source.binding["rfid_fields_sha256"] == pins["fields_sha256"], "RFID source fields changed")
    _require(source.binding["applied_forcing_sha256"] == pins["applied_forcing_sha256"] == binding["applied_forcing_sha256"],
             "RFID applied forcing changed")
    _require(source.binding["maple_case_identity_sha256"] == pins["case_identity_sha256"], "RFID source case identity changed")

    log("recomputing the audit (terrain, gap fill, resize, routing graph)")
    fresh = audit_chastre(recipe, staged, source, staged_record=report["terrain_source"])
    for key in ("grid", "hydrology", "forcing", "terrain", "rainfall_scaling", "placeholder_terrain", "bed", "source_rfid",
                "legacy_options", "grain_maps"):
        _require(_json_safe(fresh.report[key]) == _json_safe(report[key]), f"report section {key!r} differs from a fresh audit")
    _require(set(fields) == set(fresh.fields), "sidecar array names differ from a fresh audit")
    for name, array in fresh.fields.items():
        _require(fields[name].dtype == array.dtype and np.array_equal(fields[name], array),
                 f"sidecar array {name} differs from a fresh audit")
    definition = fresh.definition
    _require(binding["grid"] == {"ny": definition.base.interior_shape[0], "nx": definition.base.interior_shape[1],
                                 "nz": definition.bed.nz, "datum_offset_m": definition.bed.datum_offset_m},
             "bound global grid/datum differs from the audit")

    tiles = binding["tiles"]
    ny, _nx = definition.base.interior_shape
    bands = tile_bands(ny, int(binding["tile_rows"]))
    _require([(t["row_start"], t["row_stop"]) for t in tiles] == bands and [t["index"] for t in tiles] == list(range(len(bands))),
             "the bound tiles do not cover the interior in the declared bands")
    grain_classes = geometry_loaded = None
    for position, tile in enumerate(tiles):
        tile_dir = case_dir / tile["dir"]
        _require(tile_dir.is_dir(), f"persisted tile {tile['dir']} is missing")
        log(f"tile {position + 1}/{len(tiles)}: hashing")
        actual = hash_tree(tile_dir)
        _require(actual == tile["files"], f"tile {tile['dir']} files differ from the bound manifest (modified, added or missing)")
        if reload_tiles or position == 0:
            loaded = load_compiled_case(tile_dir)
            record = loaded.provenance_record
            _require(record["case_identity_sha256"] == tile["case_identity_sha256"], f"tile {tile['dir']} case identity differs")
            _require(record["artifact_sha256"] == tile["artifact_sha256"], f"tile {tile['dir']} artifact hashes differ")
            audit = tile_audit(definition, tile["index"], tile["row_start"], tile["row_stop"], tile_dir)
            check_compiled_plot1(loaded, audit)
            if position == 0:
                grain_classes = loaded.config.grain_classes
                geometry_loaded = loaded.config.geometry
            del loaded
            gc.collect()
    _require(manifest_digest(tiles) == binding["tiles_digest"], "bound tile manifest digest is inconsistent")

    geometry = _audit_geometry(definition.base)
    for attr in ("nx", "dx_m", "dy_m", "voxel_dz_m", "bulk_density_kg_m3", "active_layer_thickness_m"):
        _require(getattr(geometry, attr) == getattr(geometry_loaded, attr), f"global geometry {attr} differs from the tiles'")
    case = ChastreCase(case_dir, geometry, grain_classes, tiles, binding["tiles_digest"], progress=progress)
    return VerifiedPlot1Case(
        case_dir=case_dir, case=case, fields=fields, report=report, binding=binding, rainfall_path=sidecar / FORCING_NAME,
        checks={"binding_sha256": stream_sha256(binding_path), "maple_source_digest_now": now, "maple_source_digest_bound": bound,
                "maple_source_changed_since_import": now != bound,
                "fresh_audit": "terrain, gap fill, resize, graph, bed plan, sidecar arrays and report sections recomputed and equal",
                "tiles": {"n_tiles": len(tiles), "reloaded_and_checked": bool(reload_tiles), "files_equal_manifest": True,
                          "tiles_digest": binding["tiles_digest"]}},
        maple_dependency=dependency, maple_provenance=maple_record)


# --------------------------------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m maple_syrup.chastre_case",
                                     description="Plan, generate or verify the tiled Chastre/RFID water-only case.")
    parser.add_argument("--recipe", help="cases/chastre/recipe.yaml (plan / generate)")
    parser.add_argument("--output-dir", help="NEW case directory (plan / generate)")
    parser.add_argument("--plan-only", action="store_true", help="print bands, estimates and free disk; write nothing")
    parser.add_argument("--verify-case", help="verify a completed case directory")
    parser.add_argument("--tile-rows", type=int)
    parser.add_argument("--source-case-dir")
    parser.add_argument("--expected-maple-root")
    parser.add_argument("--allow-maple-source-change", action="store_true")
    parser.add_argument("--hash-only-tiles", action="store_true", help="verify: stream-hash tiles but reload only tile 0")
    args = parser.parse_args(argv)

    def progress(message: str) -> None:
        print(f"[chastre_case {time.strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)

    try:
        if args.verify_case:
            verified = verify_chastre_case(args.verify_case, expected_maple_root=args.expected_maple_root,
                                           allow_maple_source_change=args.allow_maple_source_change,
                                           reload_tiles=not args.hash_only_tiles, progress=progress)
            print(json.dumps(_json_safe(verified.checks), indent=2, sort_keys=True))
            return 0
        if not args.recipe or not args.output_dir:
            parser.error("--recipe and --output-dir are required")
        recipe = load_chastre_recipe(args.recipe)
        if args.plan_only:
            plan = plan_only(recipe, args.output_dir, tile_rows=args.tile_rows)
            print(json.dumps(_json_safe(plan), indent=2, sort_keys=True))
            return 3 if plan["refuse"] else 0
        summary = generate_chastre_case(recipe, args.output_dir, source_case_dir=args.source_case_dir,
                                        expected_maple_root=args.expected_maple_root,
                                        allow_maple_source_change=args.allow_maple_source_change,
                                        tile_rows=args.tile_rows, progress=progress)
    except MapleDependencyError as exc:
        print(f"MAPLE dependency check failed: {exc}", file=sys.stderr)
        return 2
    except Plot1ImportError as exc:
        print(f"Chastre import failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
