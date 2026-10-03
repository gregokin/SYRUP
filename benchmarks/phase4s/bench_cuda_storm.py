"""Phase 4S Task B: warm timing of `storm.evolve` with `implementation="cuda"` against the CPU `array` and `numba`
implementations of the SAME scheduler, on synthetic analytic storms (valley terrain, rain on/off/on), with transfer
counters and kernel-launch counts of the CUDA loop.

    python benchmarks/phase4s/bench_cuda_storm.py --networks small,valley_128x129 --repeats 5 --output <NEW>.json

Compile/load/preparation are timed apart (`prepare_cuda_hydrology`); every timed region ends with a stream
synchronization (the CUDA loop reads one packet per attempt, so it is host-synchronous anyway). Synthetic states, not a
Plot 1 storm: the root scripts own the qualified Plot 1 / heterogeneous comparisons. Raw observations of one process; the
source asserts nothing about what they will be. Nothing here was run by its author (file-only tools); Codex records
results. Select the device with CUDA_VISIBLE_DEVICES.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for sub in ("tests/phase4", "tests/phase4s"):
    sys.path.insert(0, str(ROOT / sub))
sys.path.insert(0, str(ROOT / "src"))

import hydro_cases as hcs

NETWORKS = {"small": "valley:60x21", "valley_128x129": "valley:128x129"}
EDGES = [0.0, 60.0, 120.0, 240.0, 300.0]
INTENSITY = [220.0, 0.0, 300.0, 0.0]


def timed(fn, repeats: int) -> dict:
    fn()  # warm-up (JIT / first-use costs are not part of the samples)
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1e3)
    s = sorted(samples)
    return {"median_ms": statistics.median(s), "min_ms": s[0], "max_ms": s[-1], "n": len(s)}


@contextlib.contextmanager
def scoped_coupled_step(replacement):
    """BENCHMARK ONLY: temporarily point `storm.coupled_step` at `replacement` so the shared scheduler runs a prepared
    CPU step; restored in `finally`. Not a production change (no source file is edited)."""
    from maple_syrup import storm

    original = storm.coupled_step
    storm.coupled_step = replacement
    try:
        yield
    finally:
        storm.coupled_step = original


def run_network(label: str, kind: str, repeats: int, mode: str) -> dict:
    import cupy as cp
    from maple.core import backend as mb

    from maple_syrup import hydrology_cuda as hc
    from maple_syrup import routing_numba
    from maple_syrup.rainfall import (
        RainfallProvenance,
        RainfallSchedule,
        rainfall_field,
    )
    from maple_syrup.storm import StormControl, evolve

    case = hcs.build_case(3, kind=kind, model="pavement_hawkins")
    ny, nx = case.graph.shape
    scale = np.where(case.graph.active, np.random.default_rng(5).uniform(1.0, 1.5, (ny, nx)), 0.0)
    schedule = RainfallSchedule(edges_s=EDGES, intensity_mm_per_h=INTENSITY, provenance=RainfallProvenance(kind="constant"))
    host_field = rainfall_field(ny, nx, scale=scale)
    dev = hcs.device_case(case, cp)
    dev_field = rainfall_field(ny, nx, scale=cp.asarray(scale))
    out: dict = {"network": label, "kind": kind, "n_cells": ny * nx, "n_active": case.graph.n_active,
                 "n_levels": case.graph.n_levels, "max_level_width": case.graph.max_level_width,
                 "storm": {"edges_s": EDGES, "intensity_mm_h": INTENSITY, "report_every_s": 60.0, "dt_max_s": 1.0}}

    t0 = time.perf_counter()
    ctx = hc.prepare_cuda_hydrology(dev.graph, dev.params, mode=mode)
    out["cuda_preparation_s"] = time.perf_counter() - t0
    out["cuda_context"] = ctx.summary()
    sync = cp.cuda.get_current_stream().synchronize

    def cuda_run():
        res = evolve(dev.graph, dev.params, dev_field, schedule, dev.state, EDGES[-1],
                     StormControl(implementation="cuda"), report_every_s=60.0, cuda_context=ctx)
        sync()
        return res

    before = mb.read_transfer_counters()
    result = cuda_run()
    delta = mb.read_transfer_counters().delta(before)
    attempts = result.n_accepted_steps + result.n_rejected_attempts
    out["cuda_one_run"] = {"accepted": result.n_accepted_steps, "rejected": result.n_rejected_attempts,
                           "transfer_counters": dataclasses.asdict(delta), "attempts": attempts,
                           "launches_per_attempt": ctx.summary()["launches_per_step"]}
    out["cuda_evolve"] = timed(cuda_run, repeats)

    def cpu_run(implementation):
        return lambda: evolve(case.graph, case.params, host_field, schedule, case.state, EDGES[-1],
                              StormControl(implementation=implementation), report_every_s=60.0)

    out["cpu_array_evolve_unprepared"] = {
        **timed(cpu_run("array"), max(1, repeats // 2)),
        "label": "reference coupled_step, NumPy array sweep, unprepared (not the CPU comparator)"}
    if routing_numba.numba_available():
        from maple_syrup import hydrology_numba as hn

        out["cpu_numba_evolve_unprepared"] = {
            **timed(cpu_run("numba"), repeats),
            "label": "reference coupled_step with the compiled sweep, unprepared per-step NumPy column/branch"}
        t0 = time.perf_counter()
        prepared = hn.prepare_hydrology(case.graph, case.params)  # once; static copies, no JIT yet
        out["cpu_numba_prepare_s"] = time.perf_counter() - t0

        def direct(graph, params, rate, state, dt, control):  # plain closure: no mock, no wrapper layers
            return hn.prepared_coupled_step(prepared, rate, state, dt, control)

        with scoped_coupled_step(direct):  # the same shared scheduler, steps on the BEST prepared batched CPU path
            out["cpu_numba_prepared_evolve_best"] = {
                **timed(cpu_run("numba"), repeats),  # the timed() warm-up call absorbs the JIT compilation
                "label": "BEST CPU comparator: prepared batched Numba hydrology inside the shared scheduler; "
                         "preparation and JIT excluded from the samples"}
    else:
        for key in ("cpu_numba_evolve_unprepared", "cpu_numba_prepared_evolve_best"):
            out[key] = "Numba not installed: not measured"
    out["pool_bytes_after"] = int(cp.get_default_memory_pool().total_bytes())
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--networks", default="small,valley_128x129")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--mode", choices=("auto", "fused", "split"), default="auto")
    parser.add_argument("--output", type=Path, required=True, help="NEW json file (an existing path is refused)")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    unknown = [n for n in args.networks.split(",") if n not in NETWORKS]
    if unknown:
        parser.error(f"unknown networks {unknown}; choose from {sorted(NETWORKS)}")
    from maple_syrup import hydrology_cuda as hc

    results = {"schema": "maple_syrup.phase4s.storm_bench.v1", "python": platform.python_version(),
               "provenance": hc.kernel_provenance(), "repeats": args.repeats,
               "networks": [run_network(n, NETWORKS[n], args.repeats, args.mode) for n in args.networks.split(",")],
               "scope": "warm synthetic analytic storms of the shared scheduler; not Plot 1, not Fortran, not scaling"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(results, handle, indent=2, default=str)
        handle.write("\n")
    print(json.dumps({n["network"]: {k: v["median_ms"] for k, v in n.items()
                                    if isinstance(v, dict) and "median_ms" in v} for n in results["networks"]},
                     indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
