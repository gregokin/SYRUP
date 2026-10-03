"""Phase 4S: warm timing of the prepared CUDA coupled hydrology step and the routing sweep modes, with the BEST CPU
prepared (batched, Numba) step as the CPU comparator (never the original serial wrapper).

    python benchmarks/phase4s/bench_cuda_hydrology.py --networks plot1like,valley_128x129,random_256x257 \\
        --repeats 30 --output <NEW>.json

Synthetic states on controlled terrains (labelled so in the output), NOT storm snapshots. Every timed region ends with a
stream synchronization (CUDA) or is host-synchronous (CPU). Compile/load/preparation are timed apart from the warm
steps. Reported per network: preparation and kernel load, launches per step, counted transfers of one step (packet
only), warm medians/IQR of the CUDA step per mode (fused / split / auto), the CPU prepared step if Numba exists, and the
raw sweep per mode. The numbers are raw observations of one process; the source says nothing about what they will be.
Nothing here was run by its author (file-only tools); Codex records results. Select the device with CUDA_VISIBLE_DEVICES.
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for sub in ("tests/phase4", "tests/phase4s", "tests/phase4r"):
    sys.path.insert(0, str(ROOT / sub))
sys.path.insert(0, str(ROOT / "src"))

import hydro_cases as hcs

NETWORKS = {"plot1like": "valley:60x21", "valley_128x129": "valley:128x129", "random_256x257": "random:256x257"}


def summarize(samples_ms: list[float]) -> dict:
    s = sorted(samples_ms)
    q = statistics.quantiles(s, n=4) if len(s) >= 4 else [s[0], s[len(s) // 2], s[-1]]
    return {"median_ms": statistics.median(s), "q1_ms": q[0], "q3_ms": q[2], "min_ms": s[0], "n": len(s)}


def timed(fn, sync, repeats: int, warmup: int) -> dict:
    for _ in range(warmup):
        fn()
    sync()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        sync()
        samples.append((time.perf_counter() - t0) * 1e3)
    return summarize(samples)


def run_network(label: str, kind: str, repeats: int, warmup: int) -> dict:
    import cupy as cp
    from maple.core import backend as mb

    from maple_syrup import hydrology_cuda as hc
    from maple_syrup import routing_cuda, routing_numba
    from maple_syrup.storm import StormControl

    case = hcs.build_case(3, kind=kind, model="pavement_hawkins")
    dev = hcs.device_case(case, cp)
    cuda = StormControl(implementation="cuda")
    sync = cp.cuda.get_current_stream().synchronize
    out: dict = {"network": label, "kind": kind, "n_cells": int(np.prod(case.graph.shape)),
                 "n_active": case.graph.n_active, "n_levels": case.graph.n_levels,
                 "max_level_width": case.graph.max_level_width, "states": "synthetic wet state, uniform scaled rain"}
    out["cuda_step"] = {}
    for mode in ("fused", "split", "auto"):
        t0 = time.perf_counter()
        ctx = hc.prepare_cuda_hydrology(dev.graph, dev.params, mode=mode)
        prep_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        hc.prepared_coupled_step(ctx, dev.rate_on, dev.state, 1.0, cuda)  # first call: not part of steady state
        sync()
        first_s = time.perf_counter() - t0
        before = mb.read_transfer_counters()
        hc.prepared_coupled_step(ctx, dev.rate_on, dev.state, 1.0, cuda)
        delta = mb.read_transfer_counters().delta(before)
        out["cuda_step"][mode] = {
            "resolved_mode": ctx.mode, "preparation_s": prep_s, "kernel_load_s": ctx.kernel_load_s,
            "first_step_s": first_s, "launches_per_step": ctx.summary()["launches_per_step"],
            "static_bytes": ctx.static_bytes, "kernel_attributes": ctx.kernel_attributes,
            "step_device_to_host": [int(delta.device_to_host), int(delta.device_to_host_bytes)],
            "step_host_to_device": [int(delta.host_to_device), int(delta.host_to_device_bytes)],
            "warm": timed(lambda ctx=ctx: hc.prepared_coupled_step(ctx, dev.rate_on, dev.state, 1.0, cuda), sync,
                          repeats, warmup),
        }
    if routing_numba.numba_available():
        from maple_syrup import hydrology_numba as hn

        cpu_ctx = hn.prepare_hydrology(case.graph, case.params)
        cpu_control = StormControl(implementation="numba")
        out["cpu_prepared_step"] = {"warm": timed(
            lambda: hn.prepared_coupled_step(cpu_ctx, case.rate_on, case.state, 1.0, cpu_control), lambda: None,
            repeats, warmup), "note": "best CPU path: prepared batched Numba hydrology; first call compiled in warm-up"}
    else:
        out["cpu_prepared_step"] = "Numba not installed: no CPU comparator measured"
    # raw sweep modes on the same graph (levelled base from a random positive state)
    rng = np.random.default_rng(3)
    base = cp.asarray(rng.uniform(1e-4, 3e-3, case.graph.n_active))
    out["raw_sweep"] = {mode: timed(lambda mode=mode: routing_cuda.run_sweep(dev.graph, base, 0.5 / 3.0, 40, mode=mode),
                                    sync, repeats, warmup) for mode in ("level", "block", "auto")}
    out["pool_bytes_after"] = int(cp.get_default_memory_pool().total_bytes())
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--networks", default="plot1like,valley_128x129,random_256x257")
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True, help="NEW json file (an existing path is refused)")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    unknown = [n for n in args.networks.split(",") if n not in NETWORKS]
    if unknown:
        parser.error(f"unknown networks {unknown}; choose from {sorted(NETWORKS)}")
    from maple_syrup import hydrology_cuda as hc

    results = {"schema": "maple_syrup.phase4s.bench.v1", "python": platform.python_version(),
               "provenance": hc.kernel_provenance(), "repeats": args.repeats, "warmup": args.warmup,
               "networks": [run_network(n, NETWORKS[n], args.repeats, args.warmup) for n in args.networks.split(",")],
               "scope": "warm synthetic-state steps; not a storm, not Fortran, not a scaling study"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(results, handle, indent=2, default=str)
        handle.write("\n")
    print(json.dumps({n["network"]: {m: v["warm"]["median_ms"] for m, v in n["cuda_step"].items()}
                      for n in results["networks"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
