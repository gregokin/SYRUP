"""EXPERIMENTAL RFID_2014 water-only timing harness (task rfid_timing): the original MAHLERAN routines (methods 2 and 5) and the
SYRUP legacy / explicit / local-inertial forms on ONE verified case with ONE forcing, ONE graph and ONE set of column
parameters.

    python benchmarks/rfid/run_rfid_timing.py --case-dir outputs/rfid/case --output-dir <NEW> \\
        [--contenders ...] [--end-s 2700] [--max-dt-s 1] [--rounds 3] [--fortran-build-dir <NEW>] \\
        [--allow-maple-source-change]

Reuse, not copies: setup, validation and sampling are `benchmarks/hydraulic_candidates/compare_plot1.py` (`build_runner` with
the `rfid_inputs` factory, `RunGuard`, `time_sample`, `sample_record`) -- the same shared scheduler `storm.evolve` /
`experimental_storm.evolve_experimental`, the same per-sample water budget (MAPLE-derived bound, unchanged), bed digest and
source-digest guards. Fortran samples are fresh processes of `rfid_water_driver` (see `fortran_timing.py`).

Protocol. (1) The case is re-verified (hashes, MAPLE reload, fresh audit). (2) Preparation (static contexts, geometry,
compilation of the Fortran build) is timed apart and is never part of a sample. (3) Every contender runs ONE complete untimed
warm-up storm (JIT, first use, pool growth; it is validated). (4) `--rounds` balanced rounds forward / reverse / forward (a
different order each round) of complete storms; each sample is validated AFTER its timer (SYRUP: water budget, bed and source
digests; Fortran: completion marker, finite/non-negative states, reported budget). (5) Median, min, max per contender.
Failures are recorded as failures with the reason: no time is ever reported for a failed or relabelled run, and a different
dt is never silently substituted.

What is inside the SYRUP timer: the shared scheduler's step loop with its required validation, accumulation, 60 s report rows
and the final device synchronisation. Outside: preparation, JIT, downloads, budget/bed/source checks. What is inside the
Fortran timer: the driver's step loop (forcing, infilt, route_water, update_water_flow with its dummy 6-class copy, the finite
checks and running totals); outside: input parsing, allocation, outputs. The two workloads are NOT identical and no speed
ratio is claimed to be an algorithmic comparison; accepted/rejected steps are reported per contender.

SYRUP legacy contenders use `bisection_iterations=64` (not the default 40): terminal storage cells have conveyance 0, so the
bisection root is the right-hand side itself and the default 40 halvings leave `rhs 2^-40` which exceeds the unchanged 1e-11 m
root tolerance for depths above ~10 m. The tolerance is unchanged; the iteration count is a control field.

Nothing here was run by its author (file-only tools); Codex records results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import resource
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks" / "hydraulic_candidates"))
sys.path.insert(0, str(HERE))

SYRUP_CONTENDERS = ("legacy_numba_prepared", "legacy_cuda", "explicit_numpy", "explicit_numba", "explicit_cuda",
                    "local_inertial_numpy", "local_inertial_numba", "local_inertial_cuda")
FORTRAN_CONTENDERS = ("fortran_iroute2", "fortran_iroute5")
ALL_CONTENDERS = (*FORTRAN_CONTENDERS, *SYRUP_CONTENDERS)
IROUTE = {"fortran_iroute2": 2, "fortran_iroute5": 5}
REFERENCE = "legacy_numba_prepared"
MIN_DT_S = 1.0 / 1024.0  # the retry floor of the drivers (same value as fortran_timing.MIN_DT_S)
MAX_STEPS = 1_000_000


def validate_args(args) -> list[str]:
    names = [n for n in args.contenders.split(",") if n]
    if not names or len(set(names)) != len(names) or any(n not in ALL_CONTENDERS for n in names):
        raise ValueError(f"--contenders must be a non-empty duplicate-free subset of {ALL_CONTENDERS}")
    for label, value in (("--end-s", args.end_s), ("--max-dt-s", args.max_dt_s), ("--report-every-s", args.report_every_s)):
        if not (math.isfinite(value) and value > 0.0):
            raise ValueError(f"{label} must be finite and > 0, got {value!r}")
    if args.max_dt_s < MIN_DT_S:
        raise ValueError(f"--max-dt-s must be >= the drivers' retry floor {MIN_DT_S} s")
    ratio = args.end_s / args.max_dt_s
    if ratio != round(ratio) or float(np.float32(args.max_dt_s)) != args.max_dt_s:
        raise ValueError("--end-s must be an integer multiple of --max-dt-s and dt exactly representable in REAL32 "
                         "(the fixed-step Fortran driver needs both)")
    if ratio > MAX_STEPS:
        raise ValueError(f"--end-s / --max-dt-s = {ratio:g} steps exceeds the bound {MAX_STEPS}")
    cadence = args.report_every_s / args.max_dt_s
    if cadence != round(cadence) or cadence < 1:
        raise ValueError("--report-every-s must be an integer multiple of --max-dt-s")
    if isinstance(args.rounds, bool) or args.rounds < 1:
        raise ValueError("--rounds must be an int >= 1")
    if isinstance(args.bisection_iterations, bool) or not 1 <= args.bisection_iterations <= 200:
        raise ValueError("--bisection-iterations must be an int in [1, 200]")
    if not (math.isfinite(args.cfl_max) and 0.0 < args.cfl_max <= 0.5):
        raise ValueError("--cfl-max must lie in (0, 0.5]")
    if any(n in FORTRAN_CONTENDERS for n in names) and not (args.fortran_build_dir or args.fortran_exe):
        raise ValueError("Fortran contenders need --fortran-build-dir (NEW, built here) or --fortran-exe (a timing build record)")
    if args.fortran_build_dir and args.fortran_exe:
        raise ValueError("give --fortran-build-dir OR --fortran-exe, not both")
    return names


def orders(names: list[str], rounds: int) -> list[list[str]]:
    """Balanced: round 0 forward, 1 reverse, 2 forward, ... (a different order each round)."""
    return [list(names) if k % 2 == 0 else list(reversed(names)) for k in range(rounds)]


def last_ok_sample(samples: list[dict]) -> dict:
    """The last SUCCESSFUL sample (a failed last sample must never be used as a comparison reference)."""
    ok = [x for x in samples if x.get("status") == "ok"]
    if not ok:
        raise ValueError("no successful sample")
    return ok[-1]


def summarize(values: list[float]) -> dict:
    return {"median_s": statistics.median(values), "min_s": min(values), "max_s": max(values), "samples_s": values}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case-dir", required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="NEW directory (an existing path is refused)")
    parser.add_argument("--contenders", default=",".join(ALL_CONTENDERS))
    parser.add_argument("--end-s", type=float, default=2700.0)
    parser.add_argument("--max-dt-s", type=float, default=1.0)
    parser.add_argument("--report-every-s", type=float, default=60.0)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--bisection-iterations", type=int, default=64)
    parser.add_argument("--root-solver", choices=("bisection", "newton"), default="bisection",
                        help="root solver of the SYRUP legacy contenders (default bisection, unchanged); 'newton' selects the "
                             "safeguarded Newton of the CPU Numba form and of legacy_cuda (CUDA Newton variant; see "
                             "benchmarks/newton_cpu/compare_cases.py / benchmarks/gpu_newton)")
    parser.add_argument("--newton-max-iterations", type=int, default=50)
    parser.add_argument("--cfl-max", type=float, default=0.5)
    parser.add_argument("--limiter", default="off", choices=("off", "donor"))
    parser.add_argument("--fortran-build-dir", type=Path, help="NEW directory: build the timing driver here")
    parser.add_argument("--fortran-exe", type=Path, help="a timing build's rfid_water_driver (its build.json must sit beside it)")
    parser.add_argument("--allow-maple-source-change", action="store_true")
    args = parser.parse_args(argv)
    try:
        names = validate_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        parser.error(f"refusing to overwrite {output_dir}")

    import compare_plot1 as cmp
    import fortran_timing as ft
    from maple.core.backend import read_transfer_counters

    from maple_syrup.case_import import Plot1ImportError, _refuse_output
    from maple_syrup.column_experiment import _source_digests, _syrup_provenance
    from maple_syrup.dependency import MapleDependencyError
    from maple_syrup.provenance import environment_record
    from maple_syrup.rfid_case import rfid_inputs, verify_rfid_case

    try:
        verified = verify_rfid_case(args.case_dir, allow_maple_source_change=args.allow_maple_source_change)
        syrup = _syrup_provenance()
        dependency = verified.maple_dependency
        _refuse_output(output_dir, {"MAPLE source": dependency.source_root, "MAPLE package": dependency.package_dir,
                                    "recorded MAHLERAN": Path(verified.report["mahleran"]["root"]).resolve(),
                                    "bound case": Path(args.case_dir).resolve(),
                                    "SYRUP package": Path(syrup["package_dir"]).resolve()})
    except (Plot1ImportError, MapleDependencyError) as exc:
        parser.error(str(exc))
    package_dirs = (Path(syrup["package_dir"]), Path(dependency.package_dir))
    digests_start = _source_digests(*package_dirs)
    end_s, dt = args.end_s, args.max_dt_s
    n_steps = round(end_s / dt)
    opt = SimpleNamespace(report_every_s=args.report_every_s, cfl_max=args.cfl_max, limiter=args.limiter)
    output_dir.mkdir(parents=True)

    prepared: dict = {}
    records: dict = {n: {"preparation": None, "samples": [], "warmup": None, "status": "pending"} for n in names}

    # ---------------- Fortran preparation (separate from every sample) ----------------
    fortran_inputs: dict = {}
    fortran_build = None
    if any(n in FORTRAN_CONTENDERS for n in names):
        protected = {"bound case": Path(args.case_dir).resolve(), "MAPLE source": dependency.source_root}
        try:
            if args.fortran_build_dir:
                fortran_build = ft.build(args.fortran_build_dir, variant="timing", extra_protected=protected)
            else:
                fortran_build = ft.load_build(args.fortran_exe, variant="timing")  # current driver + audited sources only
        except (RuntimeError, ValueError, OSError) as exc:
            parser.error(f"Fortran build unusable: {exc}")
        base_inputs = rfid_inputs(verified, "numpy", with_geometry=False)
        arrays = ft.common_arrays(base_inputs)
        rates = np.array([base_inputs.schedule.rate_after_m_per_s(k * dt) * 1000.0 for k in range(n_steps)])
        rain_expected = base_inputs.schedule.depth_m(0.0, end_s) * float(base_inputs.host["rainfall_scale"].sum()) * base_inputs.area
        fdir = output_dir / "fortran"
        fdir.mkdir()
        cadence = round(args.report_every_s / dt)
        active_ij = ft.expected_active_ij(arrays)

        def expect(name):  # what a QUALIFIED Fortran sample must reproduce (steps, method, end time, forcing integral, cells)
            return {"n_steps": n_steps, "dt_s": dt, "report_every_steps": cadence, "iroute": IROUTE[name],
                    "rain_expected_m3": rain_expected, "active_ij": active_ij}

        for n in names:
            if n in FORTRAN_CONTENDERS:
                path = fdir / f"input_{n}.dat"
                t0 = time.perf_counter()
                digest = ft.write_input(path, arrays, dt_s=dt, rates_mm_s=rates, iroute=IROUTE[n], report_every_steps=cadence)
                records[n]["preparation"] = {"input_write_s": time.perf_counter() - t0, "input_sha256": digest}
                fortran_inputs[n] = path

    # ---------------- SYRUP preparation + complete untimed warm-up ----------------
    guards: dict = {}
    for n in names:
        if n in FORTRAN_CONTENDERS:
            continue
        try:
            run, prep, inputs, provenance = cmp.build_runner(
                n, dt, verified, opt, inputs_factory=rfid_inputs,
                storm_overrides={"bisection_iterations": args.bisection_iterations,
                                 **({} if args.root_solver == "bisection"
                                    else {"root_solver": args.root_solver, "newton_max_iterations": args.newton_max_iterations})})
            guard = cmp.RunGuard(inputs, package_dirs, n)
            t0 = time.perf_counter()
            warm = guard.validate(run(end_s, []), end_s)  # the complete untimed warm-up, validated
            records[n].update(preparation=prep, provenance=provenance, status="ready",
                              warmup={"wall_s": time.perf_counter() - t0, "accepted": warm["host"]["accepted"],
                                      "rejected": warm["host"]["rejected"]})
            prepared[n] = run
            guards[n] = guard
        except Exception as exc:  # noqa: BLE001 - a failing contender is a recorded failure, never a time
            records[n].update(status="failed_preparation_or_warmup", reason=f"{type(exc).__name__}: {exc}")
    for n in names:
        if n in FORTRAN_CONTENDERS:
            warm = ft.run_once(Path(fortran_build["executable"]), fortran_inputs[n], fdir / f"warmup_{n}",
                               expected=expect(n), build_record=fortran_build)
            if warm["status"] != "complete":
                records[n].update(status="failed_warmup", reason=warm.get("reason", warm["status"]),
                                  returncode=warm.get("returncode"))
            else:
                records[n].update(status="ready", warmup={"loop_s": warm["loop_s"], "process_wall_s": warm["process_wall_s"]})

    # ---------------- balanced timed rounds ----------------
    live = [n for n in names if records[n]["status"] == "ready"]
    plan = orders(live, args.rounds)
    for round_index, order in enumerate(plan):
        for n in order:
            rec = records[n]
            if n in FORTRAN_CONTENDERS:
                out = ft.run_once(Path(fortran_build["executable"]), fortran_inputs[n], fdir / f"{n}_round{round_index}",
                                  expected=expect(n), build_record=fortran_build)
                if out["status"] != "complete":
                    rec["samples"].append({"round": round_index, "status": "failed", "reason": out.get("reason", out["status"])})
                    continue
                bud = ft.budget(out, base_inputs, rain_expected)
                rec["samples"].append({
                    "round": round_index, "status": "ok", "wall_s": out["loop_s"], "process_wall_s": out["process_wall_s"],
                    "max_rss_kib": out["max_rss_kib"], "steps": n_steps, "export_m3": bud["export_m3"],
                    "peak_outlet_m3_s": out["peak_outlet_m3_s"], "time_of_peak_s": out["time_of_peak_s"], "budget": bud,
                    "output_sha256": out["output_sha256"], "scratch": out["scratch"]})
            else:
                try:
                    wall, raw, transfers = cmp.time_sample(prepared[n], end_s, [], read_transfer_counters)
                    checked = guards[n].validate(raw, end_s)  # AFTER the timer, BEFORE the sample may count
                    s = cmp.sample_record(wall, transfers, checked)
                    s.update(round=round_index, status="ok", accepted=checked["host"]["accepted"],
                             rejected=checked["host"]["rejected"], export_m3=checked["host"]["export"],
                             peak_outlet_m3_s=checked["host"]["peak_q"], time_of_peak_s=checked["host"]["peak_t"])
                    rec["samples"].append(s)
                except Exception as exc:  # noqa: BLE001
                    rec["samples"].append({"round": round_index, "status": "failed", "reason": f"{type(exc).__name__}: {exc}"})
    for n, rec in records.items():
        ok = [s for s in rec["samples"] if s.get("status") == "ok"]
        rec["n_ok"], rec["n_failed"] = len(ok), len(rec["samples"]) - len(ok)
        if ok:
            rec["timing"] = summarize([s["wall_s"] for s in ok])
            if n in FORTRAN_CONTENDERS:
                rec["process_timing"] = summarize([s["process_wall_s"] for s in ok])
                rec["max_rss_kib"] = max(s["max_rss_kib"] for s in ok)
        if rec["status"] == "ready" and rec["n_failed"]:
            rec["status"] = "completed_with_failures"
        elif rec["status"] == "ready":
            rec["status"] = "completed"

    # ---------------- cross-quantities (reported, not judged) ----------------
    comparison: dict = {}
    ref = records.get(REFERENCE)
    if ref and ref.get("n_ok"):
        r = last_ok_sample(ref["samples"])
        for n, rec in records.items():
            if n == REFERENCE or not rec.get("n_ok"):
                continue
            s = last_ok_sample(rec["samples"])
            comparison[n] = {"export_rel_diff_vs_legacy_numba": (s["export_m3"] - r["export_m3"]) / r["export_m3"] if r["export_m3"] else None,
                             "peak_rel_diff_vs_legacy_numba": (s["peak_outlet_m3_s"] - r["peak_outlet_m3_s"]) / r["peak_outlet_m3_s"]
                             if r["peak_outlet_m3_s"] else None,
                             "time_of_peak_diff_s": s["time_of_peak_s"] - r["time_of_peak_s"]}

    memory = {"host_maxrss_kib_so_far": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "gpu": "pool total bytes after all samples (NOT a true peak)"}
    try:
        import cupy

        pool = cupy.get_default_memory_pool()
        memory["gpu_pool_total_bytes"], memory["gpu_pool_used_bytes"] = pool.total_bytes(), pool.used_bytes()
    except Exception:  # noqa: BLE001
        memory["gpu"] = "not measured (CuPy/device unavailable)"
    digests_end = _source_digests(*package_dirs)
    if digests_end != digests_start:
        raise RuntimeError("SYRUP/MAPLE source changed during the benchmark; no result written")
    payload = {
        "schema": "maple_syrup.rfid.timing.v1", "python": platform.python_version(),
        "arguments": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment": environment_record(), "syrup_provenance": syrup, "maple_provenance": verified.maple_provenance,
        "case_binding": dict(verified.binding), "case_checks": verified.checks,
        "source_digests": {"start": digests_start, "end": digests_end, "stable": True},
        "fortran_build": fortran_build, "orders": plan, "records": records, "comparison": comparison, "memory": memory,
        "scope": "water only, fixed terrain, one verified RFID_2014 case; SYRUP legacy contenders use the explicit "
                 f"bisection_iterations={args.bisection_iterations}; timers are not identical workloads (see module docstring); "
                 "the Fortran budget is the originals' own and not claimed conservative; no sediment/erosion/wind/restart claim",
    }
    with (output_dir / "timing.json").open("x") as handle:
        json.dump(payload, handle, indent=2, default=str)
        handle.write("\n")
    print(json.dumps({n: {"status": r["status"], "median_s": r.get("timing", {}).get("median_s")} for n, r in records.items()}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
