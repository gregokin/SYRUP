"""Phase 7i: replay the captured heterogeneous field and exact forcing through the EXISTING, UNCHANGED controlled
original-routine driver (benchmarks/phase4/reference_storm_driver.f90 -> original infilt.for / route_water.for /
update_water_flow.for), at dt = 1, 1/2, 1/4 s.

    python benchmarks/phase7i/run_controlled_fortran.py --capture-run outputs/phase7i/mahleran_capture_run \\
        --output outputs/phase7i/controlled_dt1 [--dt-s 1|0.5|0.25]

This diagnoses what the whole application cannot print: the driver reports, per step, the legacy budget residual,
the stale-inflow water gain (`qin(1)` left from the previous step) and the Crank-Nicolson closure sum, which together
explain the original conservation departures; and it provides a timestep-sensitivity series for the SAME exact field
without interpolating any reference output. Every driver input comes from the application's own post-setup capture
(order, aspect, rmask, slope, friction, ksat, psi, pave, drainage, theta_sat, theta, cum_inf, stmax) and its exact
applied rate sequence, each repeated for every sub-step of its second. Nothing is compiled or edited here: the
executable is hash-pinned to the one built and used by the earlier controlled audits.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import capture_data as cd

REPO = HERE.parents[1]
DRIVER = REPO / "benchmarks/phase4/reference_storm_driver.f90"
DEFAULT_EXE = REPO / "outputs/phase4_reference_final/dt1/build/reference_storm"
DEFAULT_EXE_SUMMARY = REPO / "outputs/phase4_reference_final/dt1/reference_summary.json"
PINNED_EXE_SHA256 = "33e742587532e020bda23edb88cbcc6b0b96a0dc09034e99c04c2e216afeb35b"
MARKER = "SYRUP_COUPLED_AUDIT_COMPLETE"
MAX_STEPS = 100_000
DT_ALLOWED = (1.0, 0.5, 0.25)
HISTORY_COLUMNS = ("time_s", "outlet_m3_s", "export_m3", "surface_m3", "soil_m3", "drain_m3", "rain_m3", "residual_m3",
                   "stale_gain_m3", "closure_sum_m3", "max_closure_m", "bracket_end_count")


def build_input_text(static: cd.Capture, rval_mm_s: np.ndarray, dt: float) -> str:
    """The driver's `input.dat`: header, routing order, one 15-column row per Fortran cell, then one rate per step."""
    s, a = static.scalars, static.arrays
    nr, nc = int(s["nr"]), int(s["nc"])
    nr2, nc2, ncell1 = int(s["nr2"]), int(s["nc2"]), int(s["ncell1"])
    m = round(1.0 / dt)
    if abs(m * dt - 1.0) > 0.0 or float(np.float32(dt)) != dt:
        raise cd.CaptureError("dt must divide one second exactly and be exact in default REAL")
    n = int(rval_mm_s.size) * m
    if not 1 <= n <= MAX_STEPS:
        raise cd.CaptureError(f"{n} steps outside 1..{MAX_STEPS}")
    if not np.all(np.isfinite(rval_mm_s)) or np.any(rval_mm_s < 0.0):
        raise cd.CaptureError("applied rates must be finite and non-negative")
    ksat = a["ksat"]
    cd.validate_ksat_mm_s(ksat, nr, nc)
    if cd.validate_order(a["order"], nr, nc).shape != (ncell1, 3):
        raise cd.CaptureError(f"order shape {a['order'].shape} != ({ncell1}, 3)")
    rmask = a["rmask"]
    active = np.zeros(rmask.shape, dtype=bool)
    active[1:nr, 1:nc] = rmask[1:nr, 1:nc] >= 0.0
    outlet = cd.legacy_outlet_mask(a["aspect"], rmask, nr, nc)
    scale = np.where(active, rmask, 0.0)
    columns = [a["aspect"], rmask, a["slope"], a["ff"], ksat, a["psi"], a["pave"], a["drain_par"], a["theta_sat"],
               a["theta"], a["cum_inf"], a["stmax"], scale, outlet.astype(np.int64), active.astype(np.int64)]
    for name, col in zip(("aspect", "rmask", "slope", "ff", "ksat", "psi", "pave", "drain_par", "theta_sat", "theta",
                          "cum_inf", "stmax", "scale", "outlet", "active"), columns, strict=True):
        if col.shape != (nr2, nc2):
            raise cd.CaptureError(f"{name} shape {col.shape} != ({nr2}, {nc2})")
    lines = [f"{nr2} {nc2} {ncell1} {n} {dt:.17e} {s['dx_mm']:.17e}"]
    lines += [" ".join(str(int(v)) for v in row) for row in a["order"]]
    int_cols = (0, 13, 14)
    for i in range(nr2):
        for j in range(nc2):
            lines.append(" ".join(str(int(col[i, j])) if k in int_cols else format(float(col[i, j]), ".17e")
                                  for k, col in enumerate(columns)))
    lines += [format(float(r), ".17e") for r in np.repeat(rval_mm_s, m)]
    return "\n".join(lines) + "\n"


def parse_history(text: str, n: int) -> np.ndarray:
    lines = text.splitlines()
    if not lines or lines[-1].strip() != MARKER:
        raise cd.CaptureError("controlled driver stopped before completion (STOP may return status zero)")
    if lines[0].split() != list(HISTORY_COLUMNS):
        raise cd.CaptureError("unexpected controlled-driver columns")
    data = np.array([[float(x) for x in ln.split()] for ln in lines[1:-1]], dtype=np.float64)
    if data.shape != (n, len(HISTORY_COLUMNS)) or not np.all(np.isfinite(data)):
        raise cd.CaptureError("controlled-driver history has the wrong shape or non-finite values")
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--capture-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="NEW directory")
    parser.add_argument("--dt-s", type=float, default=1.0, choices=DT_ALLOWED)
    parser.add_argument("--executable", type=Path, default=DEFAULT_EXE)
    parser.add_argument("--timeout-s", type=float, default=900.0)
    args = parser.parse_args(argv)

    exe = args.executable.resolve()
    if not exe.is_file() or cd.sha256_file(exe) != PINNED_EXE_SHA256:
        raise SystemExit(f"controlled driver executable missing or not the pinned build ({PINNED_EXE_SHA256}); "
                         "rebuild it with benchmarks/phase4/storm_reference.py first")
    driver_sha = cd.sha256_file(DRIVER)
    if DEFAULT_EXE_SUMMARY.is_file():
        built = json.loads(DEFAULT_EXE_SUMMARY.read_text()).get("driver_sha256")
        if built != driver_sha:
            raise SystemExit("the driver source differs from the one the pinned executable was built from")
    capture_run = args.capture_run.resolve()
    record_run = cd.verify_capture_run(capture_run)
    out = cd.refuse_output(args.output, cd.protected_paths(
        record_run, ("capture run", capture_run), ("SYRUP src", REPO / "src"), ("driver tree", REPO / "benchmarks/phase4"),
        ("controlled driver build", exe.parent), ("MAPLE-SYRUP outputs of earlier phases", REPO / "outputs/phase7")))
    digest_before = cd.capture_digest(capture_run)
    static = cd.load_capture(capture_run / "Output" / cd.STATIC_NAME, "static")
    steps = cd.load_steps(capture_run / "Output" / cd.STEPS_NAME)
    text = build_input_text(static, steps["rval_applied_mm_s"], args.dt_s)
    n = int(steps["iter"].size * round(1.0 / args.dt_s))
    out.mkdir(parents=True)
    inp, hist, final = out / "input.dat", out / "hydrograph.dat", out / "final.dat"
    inp.write_text(text, encoding="ascii")
    # absolute paths; the driver runs INSIDE the new output directory so any file the original routines write
    # (e.g. fort.51 from route_water) lands there, is listed below and never touches another tree
    command = [str(exe), str(inp.resolve()), str(hist.resolve()), str(final.resolve())]
    record = {"status": "started", "command": command, "cwd": str(out), "dt_s": args.dt_s,
              "n_steps": n, "executable_sha256": cd.sha256_file(exe), "driver_source_sha256": driver_sha,
              "input_sha256": cd.sha256_file(inp), "capture_run": str(capture_run),
              "capture_executable_sha256": record_run["executable_sha256"],
              "capture_files_sha256": {name: record_run["outputs"][name]["sha256"] for name in cd.CAPTURE_FILES},
              "departures_from_application": [
                  "accumulate_flow omitted (method 5 does not use it); no sediment/chemistry; zero dummy sediment arrays",
                  "rate repeated for each sub-step of its second; applied rate is the application's exact single-precision rval",
                  "literal stale qin(1) and the legacy bisection bracket are retained; the driver's diagnostics never change state"],
              "script_sha256": cd.sha256_file(__file__), "capture_data_sha256": cd.sha256_file(HERE / "capture_data.py")}
    start = time.perf_counter()
    try:
        result = subprocess.run(command, cwd=out, capture_output=True, timeout=args.timeout_s, check=False)
    except subprocess.TimeoutExpired as exc:
        (out / "stdout.log").write_bytes(exc.stdout or b"")
        (out / "stderr.log").write_bytes(exc.stderr or b"")
        record.update(status="timeout", timeout_s=args.timeout_s, wall_s=time.perf_counter() - start)
        (out / "controlled_summary.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record, indent=2), file=sys.stderr)
        return 1
    record.update(returncode=result.returncode, wall_s=time.perf_counter() - start)
    (out / "stdout.log").write_bytes(result.stdout)
    (out / "stderr.log").write_bytes(result.stderr)
    try:
        if result.returncode != 0:
            raise cd.CaptureError(f"driver exit status {result.returncode}")
        data = parse_history(hist.read_text(), n)
        if cd.capture_digest(capture_run) != digest_before:
            raise cd.CaptureError("the capture run changed while the controlled driver ran")
        record.update(status="complete", output_sha256={p.name: cd.sha256_file(p) for p in (hist, final)},
                      peak_outlet_m3_s=float(data[:, 1].max()), peak_time_s=float(data[np.argmax(data[:, 1]), 0]),
                      final_budget_residual_m3=float(data[-1, 7]), final_stale_gain_m3=float(data[-1, 8]),
                      final_closure_sum_m3=float(data[-1, 9]),
                      residual_minus_stale_minus_closure_m3=float(data[-1, 7] - data[-1, 8] - data[-1, 9]))
    except (cd.CaptureError, OSError, ValueError) as exc:
        record.update(status="failed", error=str(exc))
    # every other file the driver left in the output directory (e.g. fort.51) is recorded, not discarded
    record["ancillary_files"] = {p.name: {"sha256": cd.sha256_file(p), "bytes": p.stat().st_size} for p in sorted(out.iterdir())
                                 if p.is_file() and p.name not in ("input.dat", "hydrograph.dat", "final.dat",
                                                                   "controlled_summary.json")}
    (out / "controlled_summary.json").write_text(json.dumps(record, indent=2) + "\n")
    if record["status"] != "complete":
        print(json.dumps(record, indent=2), file=sys.stderr)
        return 1
    print(json.dumps({k: record[k] for k in ("status", "dt_s", "n_steps", "peak_outlet_m3_s", "peak_time_s",
                                             "final_budget_residual_m3", "residual_minus_stale_minus_closure_m3")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
