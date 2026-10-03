"""EXPERIMENTAL: build and run the water-only ORIGINAL-routine timing driver for the RFID_2014 benchmark (task rfid_timing).

The driver `rfid_water_driver.f90` links the UNCHANGED hash-pinned MAHLERAN `shared_data.f90`, `ff_type8.for`, `infilt.for`,
`route_water.for` (iroute 2 = native Newton-Crank-Nicolson, iroute 5 = bisection-Crank-Nicolson) and
`update_water_flow.for`; it holds no hydrology equation. This module only: pins the reference sources by hash (reusing
`tests/phase4/fortran_reference.py`), compiles (a TIMING build `-O2`, or a CHECKED build `-fcheck=all -fbacktrace -O0` for
correctness pilots), writes the common-array input file from the SAME `rfid_inputs` namespace the SYRUP contenders use, runs one
fresh process per sample (its own working directory: `route_water` may write `fort.51`) and validates the outputs.

A sample QUALIFIES for timing only if: the process exits 0 and the completion marker is present (a legacy STOP returns status 0);
the reported steps/method equal the request; the history has exactly ceil(n/cadence) rows whose LAST row is the requested end
time and whose rain total equals the common forcing integral; the final cells cover exactly the active cells, once, with finite
non-negative depth, soil water and discharge; and the executable, input, driver and reference hashes are unchanged across the
sample. Anything else is a recorded FAILURE with its reason, never a time.

Honest scope: this times the ORIGINAL routines on the benchmark arrays; it is NOT the MAHLERAN application (no setup, sediment,
output; `update_water_flow` still copies the six dummy sediment classes, disclosed). The originals' stale-inflow behaviour and
bracket truncation are retained, not corrected, so the Fortran water budget is REPORTED and not claimed conservative.

Nothing here was run by its author (file-only tools); Codex records results.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "tests" / "phase4"))
import fortran_reference as fr

DRIVER = HERE / "rfid_water_driver.f90"
MARKER = "SYRUP_RFID_DRIVER_COMPLETE"
# dependency order: the module first
SOURCES = (
    "src/Program_Control/shared_data.f90",
    "src/Subroutines_Water/ff_type8.for",
    "src/Subroutines_Water/infilt.for",
    "src/Subroutines_Water/route_water.for",
    "src/Subroutines_Water/update_water_flow.for",
)
BASE_FLAGS = ("-std=legacy", "-ffixed-line-length-none", "-ffree-line-length-none")
BUILD_FLAGS = {"timing": ("-O2",), "checked": ("-O0", "-fcheck=all", "-fbacktrace")}
NODATA = -9999.0
MIN_DT_S = 1.0 / 1024.0  # the retry floor of the SYRUP drivers
MAX_STEPS = 1_000_000
RAIN_RTOL = 1.0e-10
# trees no build/run output may be written into (the caller may add the case, MAPLE and recipe trees)
PROTECTED = {"MAHLERAN reference": fr.MAHLERAN_ROOT, "SYRUP src": ROOT / "src", "SYRUP benchmarks": ROOT / "benchmarks",
             "SYRUP tests": ROOT / "tests", "SYRUP cases": ROOT / "cases"}


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def toolchain_works() -> bool:
    """True if the CONFIGURED toolchain (`fr.locate_toolchain`: env override or PATH) exists and answers `--version`."""
    toolchain = fr.locate_toolchain()
    if toolchain is None:
        return False
    try:
        return subprocess.run([toolchain.compiler, "--version"], capture_output=True, timeout=60, check=False).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def safe_destination(path, extra_protected: dict | None = None) -> Path:
    """Resolved NEW path outside every protected tree (reference, SYRUP sources/tests/benchmarks/cases, and `extra_protected`)
    and not containing one. An existing path is refused."""
    resolved = Path(path).resolve()
    for label, root in {**PROTECTED, **(extra_protected or {})}.items():
        root = Path(root).resolve()
        if resolved == root or resolved.is_relative_to(root) or root.is_relative_to(resolved):
            raise ValueError(f"refusing {resolved}: it is, lies inside or contains the {label} tree {root}")
    if resolved.exists():
        raise FileExistsError(f"refusing to write into existing path {resolved}")
    return resolved


def validate_run_request(n_steps, dt_s, report_every_steps, iroute) -> None:
    """Bounds checked BEFORE anything is compiled or written. dt must be exactly representable in default REAL and at least the
    drivers' retry floor; steps and cadence are bounded integers."""
    for name, value in (("n_steps", n_steps), ("report_every_steps", report_every_steps)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be an int >= 1, got {value!r}")
    if n_steps > MAX_STEPS:
        raise ValueError(f"n_steps {n_steps} exceeds the bound {MAX_STEPS}")
    if not (isinstance(dt_s, (int, float)) and math.isfinite(dt_s) and dt_s >= MIN_DT_S):
        raise ValueError(f"dt must be a finite number >= {MIN_DT_S} s (the drivers' retry floor), got {dt_s!r}")
    if float(np.float32(dt_s)) != dt_s:
        raise ValueError("dt must be exactly representable in default REAL (the originals' kind)")
    if iroute not in (2, 5):
        raise ValueError("iroute must be 2 or 5")


def _reference_state(root: Path) -> dict:
    return {"reference_sha256": fr.reference_hashes(root), "driver_sha256": sha256_file(DRIVER)}


def build(build_dir: Path, *, variant: str = "timing", root: Path = fr.MAHLERAN_ROOT, extra_protected: dict | None = None) -> dict:
    """Compile into the NEW `build_dir`. Refuses a changed reference source and an unsafe destination. Returns the record (also saved)."""
    if variant not in BUILD_FLAGS:
        raise ValueError(f"variant must be one of {tuple(BUILD_FLAGS)}")
    toolchain = fr.locate_toolchain()
    if toolchain is None:
        raise RuntimeError("gfortran unavailable; set MAPLE_SYRUP_GFORTRAN (and flags) as in tests/phase4/fortran_reference.py")
    hashes = fr.reference_hashes(root)
    changed = {k: v for k, v in hashes.items() if v != fr.REFERENCE_SOURCES[k]}
    if changed:
        raise RuntimeError(f"reference sources differ from the audited revision: {changed}")
    listing = fr.watched_listing(root)
    build_dir = safe_destination(build_dir, extra_protected)
    build_dir.mkdir(parents=True, exist_ok=False)
    flags = [*toolchain.flags, *BASE_FLAGS, *BUILD_FLAGS[variant]]
    base = [toolchain.compiler, *flags, f"-J{build_dir}", f"-I{build_dir}"]
    commands, objects, logs = [], [], []
    start = time.perf_counter()
    for source in [*(root / rel for rel in SOURCES), DRIVER]:
        obj = build_dir / (source.stem + ".o")
        commands.append([*base, "-c", str(source), "-o", str(obj)])
        objects.append(str(obj))
    executable = build_dir / "rfid_water_driver"
    commands.append([toolchain.compiler, *flags, *toolchain.link_flags, *objects, "-o", str(executable)])
    for command in commands:
        done = subprocess.run(command, cwd=build_dir, capture_output=True, text=True, timeout=600, check=False)
        logs.append({"command": command, "returncode": done.returncode, "stderr": done.stderr[-4000:]})
        if done.returncode != 0:
            raise RuntimeError(f"command failed ({done.returncode}): {shlex.join(command)}\n{done.stdout}\n{done.stderr}")
    version = subprocess.run([toolchain.compiler, "--version"], capture_output=True, text=True, check=False).stdout.splitlines()
    record = {
        "variant": variant, "executable": str(executable), "executable_sha256": sha256_file(executable),
        "compiler": toolchain.compiler, "compiler_version": version[0] if version else "", "flags": flags,
        "link_flags": list(toolchain.link_flags), "commands": logs, "source_sha256": hashes,
        "driver_sha256": sha256_file(DRIVER), "build_wall_s": time.perf_counter() - start,
        "reference_tree_unchanged": fr.reference_hashes(root) == hashes and fr.watched_listing(root) == listing,
    }
    if not record["reference_tree_unchanged"]:
        raise RuntimeError("the MAHLERAN reference tree changed during the build")
    (build_dir / "build.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def load_build(executable: Path, *, variant: str = "timing", root: Path = fr.MAHLERAN_ROOT) -> dict:
    """The record of an EXISTING build, accepted only if it describes this executable, the variant, the CURRENT driver source
    and the CURRENT audited reference sources (so a stale executable cannot be reused)."""
    executable = Path(executable).resolve()
    record = json.loads((executable.parent / "build.json").read_text())
    problems = []
    if record.get("variant") != variant:
        problems.append(f"variant {record.get('variant')!r} != {variant!r}")
    if Path(record.get("executable", "")).resolve() != executable or sha256_file(executable) != record.get("executable_sha256"):
        problems.append("the executable differs from its build record")
    if record.get("driver_sha256") != sha256_file(DRIVER):
        problems.append("the driver source changed since the build")
    current = fr.reference_hashes(root)
    if current != fr.REFERENCE_SOURCES or record.get("source_sha256") != current:
        problems.append("the reference sources differ from the audited revision or from the build record")
    if not record.get("reference_tree_unchanged"):
        problems.append("the build recorded a changed reference tree")
    if problems:
        raise RuntimeError("unusable Fortran build: " + "; ".join(problems))
    return record


# --------------------------------------------------------------------------------------------------------------------------
# Common input arrays (SYRUP south-first interior -> legacy north-first full grid, millimetres)
# --------------------------------------------------------------------------------------------------------------------------
def _full(interior_sf: np.ndarray, fill: float) -> np.ndarray:
    ny, nx = interior_sf.shape
    out = np.full((ny + 2, nx + 2), fill, dtype=np.float64)
    out[1:-1, 1:-1] = np.asarray(interior_sf, dtype=np.float64)[::-1]
    return out


def common_arrays(inputs) -> dict:
    """The legacy arrays of a SYRUP `rfid_inputs` namespace (NumPy backend). Matched input: the ring and the inactive interior
    cells have rmask -9999 (non-computed); active cells carry the rainfall scale; slope/aspect/order come from the SAME graph
    (edge-rule slope included) every SYRUP method uses, so all contenders see identical static data."""
    graph, h, p = inputs.graph, inputs.host, inputs.params
    ny, nx = graph.shape
    active = np.asarray(graph.active)
    scale = np.where(active, h["rainfall_scale"], 0.0)
    flat = np.asarray(graph.level_order_host)
    r, c = np.divmod(flat, nx)
    order = np.stack([ny + 1 - r, c + 2, np.asarray(graph.level).reshape(-1)[flat]], axis=1).astype(np.int64)
    soil = np.asarray(inputs.soil0) * 1.0e3
    return {
        "order": order,
        "aspect": _full(np.asarray(graph.aspect), 0.0).astype(np.int64),
        "rmask": _full(np.where(active, scale, NODATA), NODATA),
        "slope": _full(np.asarray(graph.slope), 0.0),
        "ff": _full(np.where(active, np.asarray(graph.friction_factor), 1.0), 1.0),
        "ksat": _full(np.asarray(p.ksat_m_per_s) * 1.0e3, 0.0),
        "psi": _full(np.asarray(p.suction_m) * 1.0e3, 0.0),
        "pave": _full(np.zeros((ny, nx)), 0.0),
        "drain_par": _full(np.asarray(p.drainage_parameter), 0.0),
        "theta_sat": _full(np.asarray(p.theta_sat), 0.4),
        "theta": _full(np.where(active, h["initial_theta"], 0.0), 0.0),
        "cum_inf": _full(soil, 0.0),
        "stmax": _full(np.asarray(p.storage_max_m) * 1.0e3, 1.0),
        "scale": _full(scale, 0.0),
        "outlet": _full(np.asarray(graph.outlet).astype(float), 0.0).astype(np.int64),
        "active": _full(active.astype(float), 0.0).astype(np.int64),
        "dx_mm": float(graph.dx_m) * 1000.0,
        "n_active": int(active.sum()),
    }


def expected_active_ij(arrays: dict) -> np.ndarray:
    """1-based legacy (i, j) of the active cells in the row-major order the driver writes its final cells."""
    return np.argwhere(arrays["active"].astype(bool)) + 1


def write_input(path: Path, arrays: dict, *, dt_s: float, rates_mm_s: np.ndarray, iroute: int, report_every_steps: int) -> str:
    """Write the driver input (NEW file) and return its SHA-256. All forcing is in the file: the driver reads it before the timer.
    The request is validated first (bounded steps, dt >= retry floor and REAL-exact, cadence >= 1, method 2 or 5)."""
    rates = np.asarray(rates_mm_s, dtype=np.float64)
    if rates.ndim != 1 or rates.size < 1 or not (np.all(np.isfinite(rates)) and np.all(rates >= 0.0)):
        raise ValueError("rates must be a non-empty finite non-negative 1-D array")
    validate_run_request(int(rates.size), dt_s, report_every_steps, iroute)
    nr2, nc2 = arrays["aspect"].shape
    order = arrays["order"]
    int_cols = (0, 13, 14)
    names = ("aspect", "rmask", "slope", "ff", "ksat", "psi", "pave", "drain_par", "theta_sat", "theta", "cum_inf",
             "stmax", "scale", "outlet", "active")
    columns = [arrays[n] for n in names]
    lines = [f"{nr2} {nc2} {order.shape[0]} {rates.size} {dt_s:.17e} {arrays['dx_mm']:.17e} {iroute} {report_every_steps}"]
    lines += [" ".join(str(int(v)) for v in row) for row in order]
    for i in range(nr2):
        for j in range(nc2):
            lines.append(" ".join(str(int(col[i, j])) if k in int_cols else format(float(col[i, j]), ".17e")
                                  for k, col in enumerate(columns)))
    lines += [format(float(x), ".17e") for x in rates]
    text = "\n".join(lines) + "\n"
    with Path(path).open("x", encoding="ascii") as handle:
        handle.write(text)
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def _wait_with_timeout(proc, timeout_s: float):
    deadline = time.monotonic() + timeout_s
    while True:
        pid, status, usage = os.wait4(proc.pid, os.WNOHANG)
        if pid:
            return pid, status, usage
        if time.monotonic() > deadline:
            proc.kill()
            os.wait4(proc.pid, 0)
            raise TimeoutError
        time.sleep(0.05)


def _fail(record: dict, reason: str) -> dict:
    record.update(status="failed", reason=reason)
    return record


def run_once(executable: Path, input_path: Path, workdir: Path, *, expected: dict, build_record: dict | None = None,
             timeout_s: float = 3600.0, extra_protected: dict | None = None) -> dict:
    """One fresh process in the NEW `workdir`. `expected` = {n_steps, dt_s, report_every_steps, iroute, rain_expected_m3,
    active_ij}. Returns timings (the driver's own loop timer, whole-process wall, child rusage), the parsed history and final
    cells, and `status` "complete" ONLY if every qualification condition of the module docstring holds; otherwise `status`
    "failed" with `reason` (or "timeout"). `build_record` (from `build` / `load_build`), when given, is re-verified before and
    after the sample (executable, driver and reference hashes)."""
    executable, input_path = Path(executable).resolve(), Path(input_path).resolve()
    validate_run_request(expected["n_steps"], expected["dt_s"], expected["report_every_steps"], expected["iroute"])
    guard_before = {"executable": sha256_file(executable), "input": sha256_file(input_path), **_reference_state(fr.MAHLERAN_ROOT)}
    if build_record is not None and (guard_before["executable"] != build_record["executable_sha256"]
                                     or guard_before["driver_sha256"] != build_record["driver_sha256"]
                                     or guard_before["reference_sha256"] != build_record["source_sha256"]):
        raise RuntimeError("the executable, driver or reference sources differ from the build record; refusing to run")
    workdir = safe_destination(workdir, extra_protected)
    workdir.mkdir(parents=True, exist_ok=False)
    hist, final = workdir / "history.dat", workdir / "final.dat"
    command = [str(executable), str(input_path), str(hist), str(final)]
    toolchain = fr.locate_toolchain()
    env = os.environ.copy()
    if toolchain is not None and toolchain.run_library_path:
        env["LD_LIBRARY_PATH"] = os.pathsep.join([*toolchain.run_library_path, env.get("LD_LIBRARY_PATH", "")])
    start = time.perf_counter()
    with (workdir / "stdout.log").open("wb") as out, (workdir / "stderr.log").open("wb") as err:
        proc = subprocess.Popen(command, cwd=workdir, stdout=out, stderr=err, env=env)
        try:
            _, status, usage = _wait_with_timeout(proc, timeout_s)
        except TimeoutError:
            return {"status": "timeout", "command": command}
    wall = time.perf_counter() - start
    returncode = os.waitstatus_to_exitcode(status)
    stdout = (workdir / "stdout.log").read_text(errors="replace")
    record: dict = {"command": command, "returncode": returncode, "process_wall_s": wall,
                    "max_rss_kib": int(usage.ru_maxrss), "user_cpu_s": usage.ru_utime, "system_cpu_s": usage.ru_stime,
                    "stdout_tail": stdout[-2000:]}
    guard_after = {"executable": sha256_file(executable), "input": sha256_file(input_path), **_reference_state(fr.MAHLERAN_ROOT)}
    record["guard_unchanged"] = guard_before == guard_after
    if not record["guard_unchanged"]:
        return _fail(record, "executable, input, driver or reference hashes changed during the sample")
    tokens = {ln.split()[0]: ln.split()[1] for ln in stdout.splitlines() if len(ln.split()) == 2 and ln.startswith("RFID_")}
    complete = hist.is_file() and hist.read_text().rstrip().endswith(MARKER)
    if returncode != 0:
        return _fail(record, "nonzero exit status")
    if not complete or "RFID_LOOP_SECONDS" not in tokens:
        return _fail(record, "no completion marker (a legacy STOP returns status 0)")
    try:
        loop_s = float(tokens["RFID_LOOP_SECONDS"].replace("D", "E"))
        steps, method = int(tokens["RFID_STEPS"]), int(tokens["RFID_IROUTE"])
    except (KeyError, ValueError):
        return _fail(record, "unparseable driver report")
    if not (math.isfinite(loop_s) and loop_s > 0.0):
        return _fail(record, "the loop timer is not a positive finite number")
    if steps != expected["n_steps"] or method != expected["iroute"]:
        return _fail(record, f"driver reports {steps} steps / method {method}, requested {expected['n_steps']} / {expected['iroute']}")
    rows, peak = [], None
    for ln in hist.read_text().splitlines()[1:]:
        parts = ln.split()
        if ln.startswith("PEAK"):
            peak = (float(parts[1]), float(parts[2]))
        elif parts and not ln.startswith(MARKER):
            rows.append([float(t) for t in parts])
    data = np.array(rows)
    n_rows = -(-expected["n_steps"] // expected["report_every_steps"])
    end_s = expected["n_steps"] * expected["dt_s"]
    if data.shape != (n_rows, 7) or peak is None or not np.all(np.isfinite(data)):
        return _fail(record, f"history shape {data.shape} != ({n_rows}, 7), no peak, or non-finite values")
    if np.any(np.diff(data[:, 0]) <= 0.0) or data[-1, 0] != end_s:
        return _fail(record, f"history times are not strictly increasing or the last row is at {data[-1, 0]} s, not the end {end_s} s")
    if not math.isclose(data[-1, 5], expected["rain_expected_m3"], rel_tol=RAIN_RTOL, abs_tol=0.0):
        return _fail(record, f"rain total {data[-1, 5]} m3 differs from the common forcing integral {expected['rain_expected_m3']} m3")
    try:
        cells = np.loadtxt(final, ndmin=2)
    except ValueError:
        return _fail(record, "unreadable final cells")
    ij = np.asarray(expected["active_ij"])
    if cells.shape != (ij.shape[0], 5) or not np.all(np.isfinite(cells)):
        return _fail(record, f"final cells shape {cells.shape} != ({ij.shape[0]}, 5) or non-finite")
    if not np.array_equal(cells[:, :2].astype(np.int64), ij) or not np.array_equal(cells[:, :2], np.rint(cells[:, :2])):
        return _fail(record, "final cell indices are not exactly the active cells, once each")
    if np.any(cells[:, 2:] < 0.0):
        return _fail(record, "negative final depth, soil water or discharge")
    record.update(status="complete", loop_s=loop_s, history=data, peak_outlet_m3_s=peak[0], time_of_peak_s=peak[1],
                  final_cells=cells, output_sha256={p.name: sha256_file(p) for p in (hist, final)},
                  scratch=sorted(p.name for p in workdir.iterdir()))
    return record


def budget(record: dict, inputs, rain_expected_m3: float) -> dict:
    """The ORIGINAL routines' own water budget from the final history row (which is at the requested end for a qualified sample):
    surface + soil + drainage + export - rain - initial soil. REPORTED; the originals' stale-inflow/bracket behaviour means it is
    not expected to close."""
    row = record["history"][-1]
    area = inputs.area
    soil0 = float(np.sum(np.asarray(inputs.soil0))) * area
    residual = float(row[2] + row[3] + row[4] + row[1] - row[5] - soil0)
    return {"time_s": float(row[0]), "surface_m3": float(row[2]), "soil_m3": float(row[3]), "drainage_m3": float(row[4]),
            "export_m3": float(row[1]), "rain_m3": float(row[5]), "soil_initial_m3": soil0,
            "rain_expected_m3": rain_expected_m3, "residual_m3": residual,
            "relative_to_rain": residual / row[5] if row[5] else math.nan,
            "note": "original routines; not claimed conservative"}
