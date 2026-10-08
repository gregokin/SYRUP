"""EXPERIMENTAL timing of the SYRUP water solvers on the tiled Chastre/RFID case (task chastre_timing). WATER ONLY.

    python benchmarks/chastre/run_chastre_timing.py --case-dir <generated case> --output-dir <NEW> \\
        [--contenders bisection_numba,newton_numba,bisection_cuda,newton_cuda] [--end-s 2700] [--rounds 3] [--order balanced]

A thin entry over the existing harness: `benchmarks/newton_cpu/compare_cases.py` (`build_syrup_runner`, `timed_validated`,
`round_orders`, `compare_finals`) and `benchmarks/hydraulic_candidates/compare_plot1.py` (`time_sample`, `sample_record`,
`RunGuard`). `compare_cases.py` is unchanged; `compare_plot1.RunGuard` gained ONE additive optional `bed_digest` callback
(default unchanged; used here for the persisted-tile digest). Contenders are the four explicit forms only (Numba CPU and CUDA, bisection and Newton): no
NumPy, Fortran or candidate-hydraulics contender (a Python-loop NumPy storm on 1.14 M cells is not a meaningful timing).

Protocol (as the RFID harness): case verification (one tile in memory at a time, outside every timer), static preparation, ONE
first short call (JIT included), ONE complete untimed warm-up, then `--rounds` complete storms from a FRESH state in `--order`.
Each sample is validated AFTER its own timer and BEFORE it counts. Inside a sample timer: the scheduler loop and the final
synchronisation only. No progress output is printed inside a timer. `--end-s` shorter than 2700 s is a PILOT and is recorded
as such. There is no per-step callback (the existing API has none that stays outside the physics).

Bed guard. The hydrology never receives bed arrays. `RunGuard(bed_digest=chastre_bed_digest)` stream-hashes every persisted tile
file before the run and after every validated sample and requires equality (and, once up front, equality with the bound
manifest): it reports ARTIFACT immutability, not an in-memory bed digest. Each digest reads the whole persisted bed (~33 GB):
expect it to dominate wall time outside the timers.

Memory honesty: `host_maxrss_kib_so_far` is the process-wide maximum from `getrusage`; the CuPy pool figure is the allocator's
pool, not the whole-GPU peak. GPU freshness: with a CUDA contender the free device memory is recorded and a minimum is
required (`--min-free-gpu-gib`); this is NOT a utilization check, the operator must ensure the GPU is idle.

Results are not judged here: deviations between contenders are reported with the declared backend bounds (rtol 2e-12, atol
1e-14) as FLAGS only; cross-solver flags are retained even when not identical. Whether a same-solver CPU/GPU comparison
qualifies is judged independently by Codex after the results are recorded. Nothing here was run by its
author (file-only tools); Codex records results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for sub in ("src", "benchmarks/hydraulic_candidates", "benchmarks/rfid", "benchmarks/newton_cpu"):
    sys.path.insert(0, str(ROOT / sub))

CONTENDERS = ("bisection_numba", "newton_numba", "bisection_cuda", "newton_cuda")
CUDA = ("bisection_cuda", "newton_cuda")
FULL_STORM_S = 2700.0
PAIRS = (("newton_numba", "bisection_numba"), ("bisection_cuda", "bisection_numba"), ("newton_cuda", "newton_numba"),
         ("newton_cuda", "bisection_cuda"))


def say(message: str) -> None:
    """Progress line on stderr (never called inside a timer)."""
    print(f"[chastre {time.strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


def write_status(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    os.replace(tmp, path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case-dir", required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="NEW directory (an existing path is refused)")
    parser.add_argument("--contenders", default=",".join(CONTENDERS))
    parser.add_argument("--end-s", type=float, default=FULL_STORM_S, help="default 2700 (full storm); shorter is a recorded PILOT")
    parser.add_argument("--max-dt-s", type=float, default=1.0)
    parser.add_argument("--report-every-s", type=float, default=60.0)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--order", choices=("balanced", "forward", "reverse"), default="balanced")
    parser.add_argument("--bisection-iterations", type=int, default=64)
    parser.add_argument("--newton-max-iterations", type=int, default=50)
    parser.add_argument("--cuda-mode", choices=("auto", "fused", "split"), default="auto")
    parser.add_argument("--min-free-gpu-gib", type=float, default=9.0,
                        help="refuse CUDA contenders when less device memory is free (a freshness proxy, not utilization)")
    parser.add_argument("--hash-only-tile-verify", action="store_true",
                        help="startup verification stream-hashes all tiles but reloads only tile 0 (default reloads every tile)")
    parser.add_argument("--reference")
    parser.add_argument("--allow-maple-source-change", action="store_true")
    args = parser.parse_args(argv)
    # fields compare_cases.validate_args / build_syrup_runner read
    args.case, args.fortran_build_dir, args.fortran_exe = "rfid", None, None
    if not (math.isfinite(args.min_free_gpu_gib) and args.min_free_gpu_gib > 0.0):  # before any CuPy or case work
        parser.error(f"--min-free-gpu-gib must be a finite number > 0, got {args.min_free_gpu_gib!r}")

    import compare_cases as cc
    import compare_plot1 as cmp
    import run_rfid_timing as rrt

    names = [n for n in args.contenders.split(",") if n]
    if not names or len(set(names)) != len(names) or any(n not in CONTENDERS for n in names):
        parser.error(f"--contenders must be a non-empty duplicate-free subset of {CONTENDERS}")
    try:
        cc.validate_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        parser.error(f"refusing to overwrite {output_dir}")

    from maple.core.backend import read_transfer_counters

    from maple_syrup import hydrology_numba as hn
    from maple_syrup import routing_newton, routing_newton_cuda
    from maple_syrup.case_import import Plot1ImportError, _refuse_output
    from maple_syrup.chastre_case import chastre_bed_digest, verify_chastre_case
    from maple_syrup.column_experiment import _source_digests, _syrup_provenance
    from maple_syrup.dependency import MapleDependencyError
    from maple_syrup.provenance import environment_record
    from maple_syrup.rfid_case import rfid_inputs

    cupy = None
    gpu_start = None
    if any(n in CUDA for n in names):
        from maple.core.backend import cupy_module, gpu_execution_available

        if not gpu_execution_available():
            parser.error("a *_cuda contender needs CuPy and a CUDA device; none is available (no CPU substitution)")
        cupy = cupy_module()

        def gpu_memory():
            free, total = cupy.cuda.runtime.memGetInfo()
            return {"free_bytes": int(free), "total_bytes": int(total),
                    "allocator_pool_bytes": int(cupy.get_default_memory_pool().total_bytes())}

        gpu_start = gpu_memory()
        if gpu_start["free_bytes"] < args.min_free_gpu_gib * 2 ** 30:
            parser.error(f"only {gpu_start['free_bytes'] / 2 ** 30:.2f} GiB of device memory is free "
                         f"(< --min-free-gpu-gib {args.min_free_gpu_gib}); the GPU is not idle enough")

    status_path = None
    status = {"pid": os.getpid(), "argv": sys.argv, "start_unix_s": time.time(), "status": "starting", "output_dir": str(output_dir)}
    t_verify = time.perf_counter()
    try:
        say("verifying the case (one tile in memory at a time)")
        verified = verify_chastre_case(args.case_dir, allow_maple_source_change=args.allow_maple_source_change,
                                       reload_tiles=not args.hash_only_tile_verify, progress=say)
        syrup = _syrup_provenance()
        dependency = verified.maple_dependency
        _refuse_output(output_dir, {"MAPLE source": dependency.source_root, "MAPLE package": dependency.package_dir,
                                    "recorded MAHLERAN": Path(verified.report["mahleran"]["root"]).resolve(),
                                    "bound case": Path(args.case_dir).resolve(),
                                    "SYRUP package": Path(syrup["package_dir"]).resolve()})
    except (Plot1ImportError, MapleDependencyError) as exc:
        parser.error(str(exc))
    verification_s = time.perf_counter() - t_verify
    package_dirs = (Path(syrup["package_dir"]), Path(dependency.package_dir))
    digests_start = _source_digests(*package_dirs)
    end_s, dt = args.end_s, args.max_dt_s
    output_dir.mkdir(parents=True)
    status_path = output_dir / "run_status.json"
    status["status"] = "running"
    write_status(status_path, status)
    records = {n: {"preparation": None, "samples": [], "warmup": None, "status": "pending"} for n in names}
    prepared, guards, last = {}, {}, {}
    memory_log = []

    def memory_point(label):
        point = {"label": label, "host_maxrss_kib_so_far": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        if cupy is not None:
            point["gpu"] = gpu_memory()
        memory_log.append(point)

    try:
        # ---------------- preparation, first call, complete warm-up (all outside the sample timers) ----------------
        say("verifying the persisted bed manifest once against the binding")
        t0 = time.perf_counter()
        bound_digest = verified.case.persisted_digest()
        if bound_digest != verified.case.bound_tiles_digest:
            raise RuntimeError("persisted tiles differ from the bound manifest before the run; no timing")
        initial_bed_check_s = time.perf_counter() - t0
        memory_point("after verification")
        for n in names:
            say(f"{n}: preparation")
            try:
                if n in CUDA:
                    records[n]["gpu_before_preparation"] = gpu_memory()
                run, prep, inputs, provenance = cc.build_syrup_runner(n, dt, verified, args, rfid_inputs)
                guard = cmp.RunGuard(inputs, package_dirs, n, bed_digest=chastre_bed_digest)
                say(f"{n}: first call")
                first = cc.timed_validated(run, guard, dt)
                say(f"{n}: complete warm-up storm")
                warm = cc.timed_validated(run, guard, end_s)
                records[n].update(
                    preparation=prep, provenance=provenance, status="ready", root_solver=n.split("_")[0],
                    first_call={"end_s": dt, "evolution_wall_s": first["evolution_wall_s"], "guard_wall_s": first["guard_wall_s"],
                                "accepted": first["checked"]["host"]["accepted"],
                                "note": "includes JIT compilation, first step and driver setup; not an isolated compiler cost"},
                    warmup={"end_s": end_s, "evolution_wall_s": warm["evolution_wall_s"], "guard_wall_s": warm["guard_wall_s"],
                            "accepted": warm["checked"]["host"]["accepted"], "rejected": warm["checked"]["host"]["rejected"],
                            "note": "complete untimed storm; evolution and (bed + budget) validation timed apart"})
                prepared[n], guards[n] = run, guard
                memory_point(f"{n} after warm-up")
                say(f"{n}: ready (warm-up {warm['evolution_wall_s']:.1f} s)")
            except Exception as exc:  # noqa: BLE001 - a failing contender is a recorded failure, never a time
                records[n].update(status="failed_preparation_or_warmup", reason=f"{type(exc).__name__}: {exc}")
                say(f"{n}: FAILED preparation/warm-up: {type(exc).__name__}: {exc}")

        # ---------------- timed rounds; validation and capture OUTSIDE every timer ----------------
        live = [n for n in names if records[n]["status"] == "ready"]
        plan = cc.round_orders(live, args.rounds, args.order)
        for round_index, order in enumerate(plan):
            for n in order:
                say(f"round {round_index + 1}/{len(plan)} {n}: timed sample starts")
                try:
                    wall, raw, transfers = cmp.time_sample(prepared[n], end_s, [], read_transfer_counters)
                    say(f"round {round_index + 1}/{len(plan)} {n}: sample {wall:.2f} s; validating")
                    checked = guards[n].validate(raw, end_s)  # AFTER the timer, BEFORE the sample may count
                    sample = cmp.sample_record(wall, transfers, checked)
                    host = checked["host"]
                    sample.update(round=round_index, status="ok", accepted=host["accepted"], rejected=host["rejected"],
                                  export_m3=host["export"], peak_outlet_m3_s=host["peak_q"], time_of_peak_s=host["peak_t"])
                    records[n]["samples"].append(sample)
                    last[n] = {"finals": checked["finals"], "export_m3": host["export"], "peak_outlet_m3_s": host["peak_q"],
                               "time_of_peak_s": host["peak_t"], "series_q": host["rows"][:, 8].tolist(), "rows": host["rows"]}
                    memory_point(f"round {round_index} {n}")
                except Exception as exc:  # noqa: BLE001
                    records[n]["samples"].append({"round": round_index, "status": "failed",
                                                  "reason": f"{type(exc).__name__}: {exc}"})
                    say(f"round {round_index + 1} {n}: FAILED {type(exc).__name__}: {exc}")

        for n, rec in records.items():
            ok = [s for s in rec["samples"] if s.get("status") == "ok"]
            rec["n_ok"], rec["n_failed"] = len(ok), len(rec["samples"]) - len(ok)
            if ok:
                rec["timing"] = rrt.summarize([s["wall_s"] for s in ok])
            if rec["status"] == "ready":
                rec["status"] = "completed_with_failures" if rec["n_failed"] else "completed"
        for n in ("bisection_numba", "newton_numba"):
            if n in records and records[n].get("provenance") is not None:
                records[n]["provenance"]["kernels"] = {**hn.kernel_provenance(), "selected_root_solver": n.split("_")[0]}

        comparisons = {}
        ref = args.reference
        if ref in last:
            for n in last:
                if n != ref:
                    comparisons[f"{n}_vs_{ref}"] = cc.compare_finals(n, last, ref)
        for a, b in PAIRS:
            if a in last and b in last and f"{a}_vs_{b}" not in comparisons:
                comparisons[f"{a}_vs_{b}"] = cc.compare_finals(a, last, b)

        digests_end = _source_digests(*package_dirs)
        if digests_end != digests_start:
            raise RuntimeError("SYRUP/MAPLE source changed during the benchmark; no result written")
        memory_point("end")
        graph_terrain = verified.report["terrain"]
        payload = {
            "schema": "maple_syrup.chastre.timing.v1", "python": platform.python_version(),
            "pilot": end_s != FULL_STORM_S, "end_s": end_s,
            "arguments": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "routing_newton_sha256": hashlib.sha256(Path(routing_newton.__file__).read_bytes()).hexdigest(),
            "routing_newton_cuda_sha256": hashlib.sha256(Path(routing_newton_cuda.__file__).read_bytes()).hexdigest(),
            "environment": environment_record(), "syrup_provenance": syrup, "maple_provenance": verified.maple_provenance,
            "case_binding": {k: v for k, v in verified.binding.items() if k != "tiles"},
            "case_tiles": [{k: v for k, v in t.items() if k != "files"} for t in verified.binding["tiles"]],
            "case_checks": verified.checks, "case_verification_s": verification_s,
            "initial_bed_manifest_check_s": initial_bed_check_s,
            "terrain": {k: graph_terrain[k] for k in ("n_pit_storage", "n_flat_storage", "n_strict_pit_storage", "n_outlets",
                                                      "n_levels", "graph_input_sha256", "graph_policy", "boundary_policy")},
            "source_digests": {"start": digests_start, "end": digests_end, "stable": True},
            "gpu_at_start": gpu_start, "orders": plan, "records": records,
            "median_wall_s": {n: r["timing"]["median_s"] for n, r in records.items() if r.get("timing")},
            "comparisons": comparisons, "memory_log": memory_log,
            "memory_note": "host figures are process-wide getrusage maxima so far; CuPy figures are allocator pool bytes and "
                           "device free memory, not a whole-GPU peak",
            "scope": ("water only, fixed terrain, zero outlets (ring nodata), terminal storage in strict and flat sinks, no "
                      "sediment/erosion/wind/restart; the bed guard hashes persisted MAPLE tile artifacts, the benchmark does not "
                      "receive bed arrays; deviations are reported, not judged; rainfall is the stretched RFID pattern, not "
                      "Chastre rainfall; not a flood prediction"),
        }
        with (output_dir / "comparison.json").open("x") as handle:
            json.dump(payload, handle, indent=2, default=str)
            handle.write("\n")
        with (output_dir / "final_fields.npz").open("xb") as handle:
            np.savez(handle, **{f"{n}_{k}": np.asarray(v) for n, d in last.items() for k, v in d["finals"].items()})
        with (output_dir / "hydrographs.npz").open("xb") as handle:
            np.savez(handle, **{f"{n}_rows": d["rows"] for n, d in last.items()},
                     **{f"{n}_outlet_m3_s": np.asarray(d["series_q"]) for n, d in last.items()})
        status.update(status="completed", end_unix_s=time.time())
        write_status(status_path, status)
    except BaseException as exc:
        status.update(status="failed", error=f"{type(exc).__name__}: {exc}", end_unix_s=time.time())
        write_status(status_path, status)
        raise
    print(json.dumps({n: {"status": r["status"], "median_s": r.get("timing", {}).get("median_s")} for n, r in records.items()},
                     indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
