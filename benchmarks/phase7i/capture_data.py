"""Phase 7i: parse, validate and convert the full-precision capture of the heterogeneous MAHLERAN run.

The capture files are written by the read-only hooks that `prepare_capture.py` adds to an ISOLATED copy of
the whole application (see that module). This module only reads them. Nothing here executes a model.

Units of the capture (MAHLERAN internal): depths mm, rates mm/s, unit discharge mm2/s, velocity mm/s,
conductivity mm/s, suction mm, cell size mm. SYRUP: m and s. Fortran arrays are `(i, j)` with the exterior
ring at i = 1, nr2 and j = 1, nc2 and i increasing NORTH to south; SYRUP arrays are `(row, col)` of the
interior with row 0 = SOUTH. `interior_south_first` is the one conversion between them.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MAGIC = "SYRUP_HYDRO_CAPTURE_V1"
COMPLETE = "SYRUP_HYDRO_CAPTURE_COMPLETE"
STATIC_NAME = "syrup_hydro_static.txt"
STEPS_NAME = "syrup_hydro_steps.txt"
FINAL_NAME = "syrup_hydro_final.txt"
CAPTURE_FILES = tuple(f"Output/{n}" for n in (STATIC_NAME, STEPS_NAME, FINAL_NAME))
MM_TO_M = 1.0e-3
MM3_TO_M3 = 1.0e-9
MAX_STEP_ROWS = 1_000_000  # bounded: no unbounded history is accepted

STEP_COLUMNS = (
    "iter", "t_s", "rval_applied_mm_s", "sum_r2_active_mm_s", "max_r2_active_mm_s",
    "q_plot_single_mm2_s", "q_outlet_double_mm2_s", "cn_export_step_m3",
    "surface_sum_mm", "soil_sum_mm", "drain_sum_mm", "excess_sum_mm_s", "max_depth_mm", "max_velocity_mm_s",
)
# Declared input-consistency gates (checked BEFORE any simulation; not outcome targets).
STATIC_RTOL_EXACT = 1.0e-12  # quantities read as the same decimal values (theta_sat, psi, drainage, ...)
STATIC_RTOL_GEOMETRY = 1.0e-9  # slope and friction (legacy DEM arithmetic)
STATIC_RTOL_PAVE = 1.0e-6  # legacy pavement (percent * 1e-4) vs sidecar fraction * 1e-2
EXPECTED_CONFIGURATION = {"iroute": 5, "ff_type": 1, "inf_type": 2, "inf_model": 2, "ndirn": 4, "rain_type": 2}
# 4-significant-digit E-format map (0.dddE-xx): worst relative rounding of a printed value is 0.5e-4 / 0.1.
ROUNDED_MAP_RELATIVE_BOUND = 5.0e-4 * (1.0 + 1.0e-9)


class CaptureError(ValueError):
    """The capture is malformed, truncated, non-finite, inconsistent or unsafe; nothing is simulated."""


@dataclass(frozen=True)
class Capture:
    kind: str
    scalars: dict
    arrays: dict


def sha256_file(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _number(token: str, what: str):
    try:
        value = int(token)
    except ValueError:
        try:
            value = float(token)
        except ValueError:
            raise CaptureError(f"{what}: {token!r} is not a number") from None
    if isinstance(value, float) and not math.isfinite(value):
        raise CaptureError(f"{what}: non-finite value {token!r}")
    return value


def _integer(token: str, what: str) -> int:
    try:
        return int(token)
    except ValueError:
        raise CaptureError(f"{what}: {token!r} is not an integer token") from None


def parse_capture_text(text: str, kind: str) -> Capture:
    """Parse a `static` or `final` capture. Strict: the magic header and the completion marker (written only
    when the application reached the end of the loop) are required, names are unique, array row counts and
    widths are exact, every value is finite."""
    lines = text.splitlines()
    if len(lines) < 2 or lines[0].strip() != f"{MAGIC} {kind}":
        raise CaptureError(f"not a {MAGIC} {kind} capture")
    if lines[-1].strip() != f"{COMPLETE} {kind}":
        raise CaptureError(f"{kind} capture is truncated (no completion marker)")
    scalars, arrays = {}, {}
    i, end = 1, len(lines) - 1
    while i < end:
        tok = lines[i].split()
        if not tok:
            i += 1
            continue
        if tok[0] == "scalar" and len(tok) == 3:
            if tok[1] in scalars or tok[1] in arrays:
                raise CaptureError(f"duplicate name {tok[1]!r}")
            scalars[tok[1]] = _number(tok[2], f"scalar {tok[1]}")
            i += 1
        elif tok[0] == "array" and len(tok) == 5 and tok[2] in ("d", "i"):
            name = tok[1]
            try:
                n1, n2 = int(tok[3]), int(tok[4])
            except ValueError:
                raise CaptureError(f"array {name}: bad shape") from None
            if name in scalars or name in arrays or n1 < 1 or n2 < 1 or i + 1 + n1 > end:
                raise CaptureError(f"array {name}: duplicate name, bad shape or truncated rows")
            rows = []
            for line in lines[i + 1:i + 1 + n1]:
                if tok[2] == "i":  # strict: a fractional or exponent token is refused, never truncated
                    values = [_integer(t, f"array {name}") for t in line.split()]
                else:
                    values = [_number(t, f"array {name}") for t in line.split()]
                if len(values) != n2:
                    raise CaptureError(f"array {name}: row has {len(values)} values, expected {n2}")
                rows.append(values)
            arrays[name] = np.array(rows, dtype=np.float64 if tok[2] == "d" else np.int64)
            i += 1 + n1
        else:
            raise CaptureError(f"unrecognised line {lines[i][:60]!r}")
    return Capture(kind, scalars, arrays)


def load_capture(path: Path | str, kind: str) -> Capture:
    return parse_capture_text(Path(path).read_text(encoding="ascii"), kind)


def parse_steps_text(text: str) -> dict:
    """Per-step scalar history -> {column: float64 array}. Rows must be iterations 1..n, finite, with a
    strictly increasing time axis."""
    lines = text.splitlines()
    if len(lines) < 4 or lines[0].strip() != f"{MAGIC} steps":
        raise CaptureError("not a steps capture")
    if lines[-1].strip() != f"{COMPLETE} steps":
        raise CaptureError("steps capture is truncated (no completion marker)")
    if lines[1].split() != ["columns", *STEP_COLUMNS]:
        raise CaptureError("steps capture has unexpected columns")
    body = lines[2:-1]
    if not 1 <= len(body) <= MAX_STEP_ROWS:
        raise CaptureError(f"{len(body)} step rows outside 1..{MAX_STEP_ROWS}")
    rows = []
    for line in body:
        values = [_number(t, "steps") for t in line.split()]
        if len(values) != len(STEP_COLUMNS):
            raise CaptureError("steps row has the wrong number of columns")
        rows.append(values)
    data = np.array(rows, dtype=np.float64)
    columns = {name: data[:, k].copy() for k, name in enumerate(STEP_COLUMNS)}
    if not np.array_equal(columns["iter"], np.arange(1, data.shape[0] + 1, dtype=np.float64)):
        raise CaptureError("step iterations are not 1..n")
    if np.any(np.diff(np.concatenate(([0.0], columns["t_s"]))) <= 0.0):
        raise CaptureError("step times are not strictly increasing")
    if np.any(columns["rval_applied_mm_s"] < 0.0):
        raise CaptureError("negative applied rainfall")
    return columns


def load_steps(path: Path | str) -> dict:
    return parse_steps_text(Path(path).read_text(encoding="ascii"))


# --- orientation and units -------------------------------------------------------------------------------
def interior_south_first(full: np.ndarray, nr: int, nc: int) -> np.ndarray:
    """Fortran `(nr + 1, nc + 1)` array (ring at i = 1, nr + 1, j = 1, nc + 1; i north to south) -> the physical
    interior `(nr - 1, nc - 1)` (i, j = 2..nr, 2..nc) in SYRUP's south-first row order. A copy."""
    a = np.asarray(full)
    if a.shape != (nr + 1, nc + 1):
        raise CaptureError(f"array shape {a.shape} is not (nr + 1, nc + 1) = {(nr + 1, nc + 1)}")
    return a[1:nr, 1:nc][::-1].copy()


def legacy_outlet_mask(aspect: np.ndarray, rmask: np.ndarray, nr: int, nc: int) -> np.ndarray:
    """`output_hydro_data_xml.f90` 131-134 outlet-cell condition, full-size boolean: an active interior cell
    whose D4 receiver is a ring cell (rmask < 0)."""
    out = np.zeros(aspect.shape, dtype=bool)
    active = rmask[1:nr, 1:nc] >= 0.0
    a = aspect[1:nr, 1:nc]
    north, south = rmask[0:nr - 1, 1:nc] < 0.0, rmask[2:nr + 1, 1:nc] < 0.0
    east, west = rmask[1:nr, 2:nc + 1] < 0.0, rmask[1:nr, 0:nc - 1] < 0.0
    out[1:nr, 1:nc] = active & (((a == 1) & north) | ((a == 2) & east) | ((a == 3) & south) | ((a == 4) & west))
    return out


def validate_ksat_mm_s(ksat_full: np.ndarray, nr: int, nc: int) -> np.ndarray:
    """The sampled conductivity of the physical interior in SYRUP order (mm/s). Refused: wrong shape,
    non-finite, non-positive (the legacy draw is positive-truncated; zero or negative here means a corrupt
    capture). Returns a copy."""
    interior = interior_south_first(np.asarray(ksat_full, dtype=np.float64), nr, nc)
    if not np.all(np.isfinite(interior)):
        raise CaptureError("conductivity realization contains non-finite values")
    if np.any(interior <= 0.0):
        raise CaptureError(f"conductivity realization contains {int(np.sum(interior <= 0.0))} non-positive cells")
    return interior


def inject_ksat(host: dict, ksat_mm_s: np.ndarray) -> dict:
    """Copy of the `plot1_parameters` host dict with the captured conductivity (mm/s -> m/s). The input dict
    and its arrays are not modified; the shape must equal the case grid."""
    shape = host["ksat_m_per_s"].shape
    k = np.asarray(ksat_mm_s, dtype=np.float64)
    if k.shape != shape:
        raise CaptureError(f"conductivity shape {k.shape} != case grid {shape}")
    if not np.all(np.isfinite(k)) or np.any(k <= 0.0):
        raise CaptureError("conductivity must be finite and positive")
    out = {name: np.array(value, copy=True) if isinstance(value, np.ndarray) else value for name, value in host.items()}
    out["ksat_m_per_s"] = k * MM_TO_M
    return out


def bind_legacy_soil_initialization(static: Capture, host: dict) -> tuple[dict, np.ndarray, dict]:
    """Match the actual legacy REAL32 soil_thick representation, after verifying its
    derived storage against the XML inputs. No input/scientific tolerances are changed.
    initialize_values_xml.f90:115,228-229,360-362 assigns REAL32 soil_thick, then
    calculates FP64 ciinit/sminit using that rounded value. Use the captured SI
    storage/soil to preserve those initial holdings exactly through unit conversion.
    """
    nr, nc = int(static.scalars["nr"]), int(static.scalars["nc"])
    legacy_thickness = np.asarray(host["soil_thickness_m"], dtype=np.float32).astype(np.float64)
    storage = interior_south_first(static.arrays["stmax"], nr, nc) * MM_TO_M
    soil = interior_south_first(static.arrays["cum_inf"], nr, nc) * MM_TO_M
    checks, failures = {}, []
    _check(checks, failures, "legacy_storage_initialization", storage,
           host["theta_sat"] * legacy_thickness, STATIC_RTOL_EXACT)
    _check(checks, failures, "legacy_soil_initialization", soil,
           host["initial_theta"] * legacy_thickness, STATIC_RTOL_EXACT)
    if failures:
        raise CaptureError("legacy REAL32 soil initialization mismatch: " + "; ".join(failures))
    out = {k: np.array(v, copy=True) if isinstance(v, np.ndarray) else v for k, v in host.items()}
    out["soil_thickness_m"] = storage / host["theta_sat"]
    checks["decision"] = "verified legacy REAL32 soil_thick; captured initial soil and maximum storage converted to SI"
    checks["xml_thickness_m"] = float(np.asarray(host["soil_thickness_m"]).flat[0])
    checks["legacy_real32_thickness_m"] = float(legacy_thickness.flat[0])
    return out, soil.copy(), checks


# --- consistency of the captured setup with the SYRUP case --------------------------------------------------
def _check(report: dict, failures: list, name: str, got, want, rtol: float, atol: float = 0.0) -> None:
    got, want = np.asarray(got, dtype=np.float64), np.asarray(want, dtype=np.float64)
    if got.shape != want.shape:
        failures.append(f"{name}: shape {got.shape} != {want.shape}")
        report[name] = {"pass": False, "shape_mismatch": True}
        return
    diff = np.abs(got - want)
    ok = bool(np.all(diff <= atol + rtol * np.abs(want)))
    scale = np.maximum(np.abs(want), np.finfo(np.float64).tiny)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        max_relative = float((diff / scale).max())  # diagnostic only: a nonzero-vs-zero error can be infinite
    report[name] = {"max_abs_difference": float(diff.max()), "max_relative_difference": max_relative,
                    "rtol": rtol, "atol": atol, "pass": ok}
    if not ok:
        failures.append(f"{name}: max relative difference {max_relative:.3e} exceeds {rtol:g}")


def validate_order(order: np.ndarray, nr: int, nc: int) -> np.ndarray:
    """Routing order `(n, 3)` of integers with every 1-based Fortran index inside the full `(nr + 1, nc + 1)` grid.
    Checked before any array is indexed with it."""
    o = np.asarray(order)
    if o.ndim != 2 or o.shape[1] != 3 or o.shape[0] < 1 or o.dtype.kind not in "iu":
        raise CaptureError(f"routing order must be an integer (n, 3) array, got {o.dtype} {o.shape}")
    if np.any(o[:, 0] < 1) or np.any(o[:, 0] > nr + 1) or np.any(o[:, 1] < 1) or np.any(o[:, 1] > nc + 1):
        raise CaptureError("routing order holds a cell index outside the (nr + 1, nc + 1) grid")
    return o


def check_static_consistency(static: Capture, ref: dict) -> dict:
    """Explicit comparison of the application's post-setup state with the SYRUP case, BEFORE simulation.

    `ref` (SYRUP order, interior, SI): aspect, slope, friction, active, outlet, rainfall_scale, theta_sat,
    suction_m, drainage_parameter, soil_thickness_m, pavement_fraction, initial_theta, initial_soil_m, dx_m.
    Raises `CaptureError` listing every failed check; returns the per-check record when all pass."""
    s, a = static.scalars, static.arrays
    nr, nc = int(s["nr"]), int(s["nc"])
    failures: list[str] = []
    report: dict = {}

    def interior(name):
        return interior_south_first(a[name], nr, nc)

    for key, want in EXPECTED_CONFIGURATION.items():
        report[key] = {"value": s.get(key), "expected": want, "pass": s.get(key) == want}
        if s.get(key) != want:
            failures.append(f"{key} = {s.get(key)!r}, expected {want}")
    for key, want in (("ksat_mod", 1.0), ("psi_mod", 1.0), ("dt_s", 1.0)):
        report[key] = {"value": s.get(key), "expected": want, "pass": s.get(key) == want}
        if s.get(key) != want:
            failures.append(f"{key} = {s.get(key)!r}, expected {want}")
    _check(report, failures, "dx_m", [s["dx_mm"] * MM_TO_M], [ref["dx_m"]], 1.0e-15)
    _check(report, failures, "dy_equals_dx", [s["dy_mm"]], [s["dx_mm"]], 0.0)
    if int(s["nr1"]) != nr or int(s["nc1"]) != nc or int(s["nr2"]) != nr + 1 or int(s["nc2"]) != nc + 1:
        failures.append("grid extents nr1/nc1/nr2/nc2 inconsistent with nr/nc")
    aspect_equal = bool(np.array_equal(interior("aspect"), ref["aspect"]))
    report["aspect"] = {"exact": aspect_equal, "pass": aspect_equal}
    if not aspect_equal:
        failures.append("aspect differs from the SYRUP graph after row conversion")
    rmask = a["rmask"]
    ring = np.ones(rmask.shape, dtype=bool)
    ring[1:nr, 1:nc] = False
    report["ring_cells"] = {"export": int(np.sum(rmask[ring] < 0.0)),
                            "non_export": int(np.sum(rmask[ring] >= 0.0)),
                            "closed_zero": int(np.sum(rmask[ring] == 0.0)), "pass": True}
    if "legacy_full_rmask" in ref:
        _check(report, failures, "full_rainfall_mask", rmask, ref["legacy_full_rmask"], 0.0)
    _check(report, failures, "rainfall_scale", interior("rmask"), ref["rainfall_scale"], 0.0)
    if not np.array_equal(interior("rmask") >= 0.0, ref["active"]):
        failures.append("active mask (rmask >= 0) differs from the SYRUP graph")
    outlet = interior_south_first(legacy_outlet_mask(a["aspect"], rmask, nr, nc), nr, nc)
    if not np.array_equal(outlet, ref["outlet"]):
        failures.append("legacy outlet cells differ from the SYRUP graph outlets")
    report["outlet_cells"] = {"legacy": int(outlet.sum()), "syrup": int(np.sum(ref["outlet"])),
                              "pass": bool(np.array_equal(outlet, ref["outlet"]))}
    _check(report, failures, "slope", interior("slope"), ref["slope"], STATIC_RTOL_GEOMETRY)
    _check(report, failures, "friction_factor", interior("ff"), ref["friction"], STATIC_RTOL_GEOMETRY)
    _check(report, failures, "theta_sat", interior("theta_sat"), ref["theta_sat"], STATIC_RTOL_EXACT)
    _check(report, failures, "suction_m", interior("psi") * MM_TO_M, ref["suction_m"], STATIC_RTOL_EXACT)
    _check(report, failures, "drainage_parameter", interior("drain_par"), ref["drainage_parameter"], STATIC_RTOL_EXACT)
    _check(report, failures, "storage_max_m", interior("stmax") * MM_TO_M,
           ref["theta_sat"] * ref["soil_thickness_m"], STATIC_RTOL_EXACT)
    _check(report, failures, "pavement_fraction", interior("pave") * 100.0, ref["pavement_fraction"], STATIC_RTOL_PAVE,
           atol=STATIC_RTOL_PAVE)
    _check(report, failures, "initial_theta", interior("theta"), ref["initial_theta"], STATIC_RTOL_EXACT)
    _check(report, failures, "initial_soil_water_m", interior("cum_inf") * MM_TO_M, ref["initial_soil_m"],
           STATIC_RTOL_EXACT)
    if np.any(a["d_initial_mm"] != 0.0) or np.any(a["q_initial_mm2_s"] != 0.0) or np.any(a["cum_drain_initial_mm"] != 0.0):
        failures.append("initial surface depth, discharge or cumulative drainage is not zero (not a dry start)")
    order = validate_order(a["order"], nr, nc)
    cells = [(int(i), int(j)) for i, j in order[:, :2] if i >= 2 and j >= 2 and rmask[int(i) - 1, int(j) - 1] >= 0.0]
    active_cells = {(i + 2, j + 2) for i, j in zip(*np.nonzero(rmask[1:nr, 1:nc] >= 0.0), strict=True)}
    report["routing_order"] = {"rows": int(order.shape[0]), "active_cells_in_order": len(cells),
                               "n_active": len(active_cells),
                               "pass": len(cells) == len(set(cells)) and set(cells) == active_cells}
    if not report["routing_order"]["pass"]:
        failures.append("legacy routing order is not a permutation of the active cells")
    if failures:
        raise CaptureError("captured setup disagrees with the SYRUP case: " + "; ".join(failures))
    return report


def reference_arrays(graph, host: dict, soil0) -> dict:
    """The SYRUP-side `ref` dict of `check_static_consistency` from a graph, the `plot1_parameters` host dict
    and the initial soil water (all interior, SYRUP order)."""
    return {
        "aspect": np.asarray(graph.aspect), "slope": np.asarray(graph.slope),
        "friction": np.asarray(graph.friction_factor), "active": np.asarray(graph.active),
        "outlet": np.asarray(graph.outlet), "rainfall_scale": np.asarray(host["rainfall_scale"]),
        "theta_sat": np.asarray(host["theta_sat"]), "suction_m": np.asarray(host["suction_m"]),
        "drainage_parameter": np.asarray(host["drainage_parameter"]),
        "soil_thickness_m": np.asarray(host["soil_thickness_m"]),
        "pavement_fraction": np.asarray(host["pavement_cover_fraction"]),
        "initial_theta": np.asarray(host["initial_theta"]), "initial_soil_m": np.asarray(soil0, dtype=np.float64),
        "dx_m": float(graph.dx_m),
    }


# --- provenance / safety -----------------------------------------------------------------------------------------
def refuse_output(output: Path | str, protected: dict[str, Path | str | None]) -> Path:
    """A NEW output path only: refuse an existing path, a path inside any protected tree and a path that
    contains one (resolved, so symlinks cannot launder it). Returns the resolved path."""
    out = Path(output).resolve()
    for label, root in protected.items():
        if root is None:
            continue
        root = Path(root).resolve()
        if out == root or out.is_relative_to(root):
            raise CaptureError(f"refusing to write inside the {label} tree {root}")
        if root.is_relative_to(out):
            raise CaptureError(f"refusing an output directory that contains the {label} tree {root}")
    if out.exists() or Path(output).is_symlink():
        raise CaptureError(f"refusing to write into existing path {out}")
    return out


def protected_paths(record: dict, *extra: tuple[str, Path | str | None]) -> dict:
    """Trees no output may be written into or contain: the capture run, its prepared source and build, the
    original reference tree and (if set) the MAPLE root, plus any `extra` (label, path)."""
    import os

    out: dict = {"capture run": record.get("cwd")}
    for label, key in (("build", "build_manifest"),):
        if record.get(key):
            out[label] = Path(record[key]).parent
            try:
                out["prepared source"] = json.loads(Path(record[key]).read_text()).get("source_root")
                manifest = Path(out["prepared source"]) / "benchmark_manifest.json"
                out["MAHLERAN reference"] = json.loads(manifest.read_text()).get("reference_root")
            except (OSError, ValueError, TypeError):
                pass  # the explicit extra trees below still apply
    if os.environ.get("MAPLE_SYRUP_EXPECTED_MAPLE_ROOT"):
        out["MAPLE source"] = os.environ["MAPLE_SYRUP_EXPECTED_MAPLE_ROOT"]
    out.update(dict(extra))
    return out


def capture_digest(run_dir: Path | str) -> dict:
    """SHA-256 of the three capture files and of the run's own execution record, for before/after comparison."""
    run = Path(run_dir)
    names = (*CAPTURE_FILES, "execution.json")
    return {name: sha256_file(run / name) for name in names}


def verify_capture_run(run_dir: Path | str) -> dict:
    """The derivative's run record (`run_mahleran.py` execution.json): completed, inputs/reference/prepared
    unchanged, and every recorded output (including the three capture files) still hashes as recorded."""
    run = Path(run_dir)
    try:
        record = json.loads((run / "execution.json").read_text())
    except (OSError, ValueError) as exc:
        raise CaptureError(f"no readable execution.json in {run}: {exc}") from None
    if record.get("returncode") != 0 or not record.get("completion_marker"):
        raise CaptureError("the capture run did not complete")
    for key in ("input_unchanged", "reference_unchanged", "prepared_unchanged"):
        if record.get(key) is not True:
            raise CaptureError(f"execution record flag {key} is not true")
    inputs = record.get("input_sha256")
    if not isinstance(inputs, dict) or not inputs:
        raise CaptureError("execution record has no input_sha256")
    for name, digest in inputs.items():
        path = run / name
        if not path.is_file() or sha256_file(path) != digest:
            raise CaptureError(f"saved run input {name} is missing or no longer matches its recorded hash")
    outputs = record.get("outputs", {})
    for name in CAPTURE_FILES:
        if name not in outputs:
            raise CaptureError(f"{name} is not among the recorded outputs")
    for name, entry in outputs.items():
        path = run / name
        if not path.is_file() or sha256_file(path) != entry["sha256"]:
            raise CaptureError(f"output {name} is missing or no longer matches its recorded hash")
    return record
