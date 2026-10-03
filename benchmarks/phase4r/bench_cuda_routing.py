"""Phase 4R CUDA routing sweep benchmark (routing only; NOT a hydrology, storm or coupled-model benchmark).

    CUDA_VISIBLE_DEVICES=<idle physical gpu> python benchmarks/phase4r/bench_cuda_routing.py \
        [--output out.json] [--repeats 30] [--plot1-case DIR] [--networks valley_60x21,valley_128x129,random_256x257,plot1]

Compares, on identical states and graphs:
  * raw sweeps: `cuda` (routing_cuda.run_sweep), `cupy_array` (the previous routing._sweep_array on CuPy) and
    `cpu_batched` (routing_numba.compiled_sweep_batched, the fast compiled CPU sweep);
  * whole `route_step` wrappers: `cuda`, `cupy_array`, and `numba_wrapper` -- which is the ORIGINAL SERIAL compiled sweep
    inside the shared wrapper (route_step has no batched CPU wrapper; the batched sweep is used by the prepared
    hydrology, which is not timed here). The two kinds of rows are different things and are labelled as such.

Reported SEPARATELY: kernel compile/load time, graph preparation (download, validation, upload, sync) and its transfer
bytes; warm synchronized wall time including host launch overhead; warm CUDA-event time over the whole launch sequence
(including inter-level gaps) and an instrumented per-level event pass (events between launches perturb timing); host
enqueue time; pool/device memory after one call. Bitwise parity of the CUDA sweep against the CPU sweep is checked
first and a mismatch aborts before timing. The numbers are whatever the machine produced; no speed claim is encoded and
nothing here says anything about full GPU hydrology or storm speed. Run on an otherwise idle device and record which
physical device CUDA_VISIBLE_DEVICES selected. Nothing here was run by its author.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "phase4"))
sys.path.insert(0, str(ROOT / "src"))

from maple.core import backend as mb

from maple_syrup import routing, routing_cuda
from maple_syrup import routing_numba as rn

from test_routing import make_graph, random_full, valley_full  # isort: skip

C = 0.5 / 3.0


def stats(samples):
    s = list(samples)
    q = statistics.quantiles(s, n=4) if len(s) >= 4 else [min(s), statistics.median(s), max(s)]
    return {"median": statistics.median(s), "min": min(s), "iqr": q[2] - q[0], "n": len(s), "samples": s}


def bits(a):
    return np.ascontiguousarray(np.asarray(a, dtype=np.float64)).view(np.uint64)


def jsonable(obj):
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    return obj


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def host_graph_spec(name):
    rng = np.random.default_rng(1)
    if name == "valley_60x21":
        z, ff = valley_full(60, 21), 5.0
    elif name == "valley_128x129":
        z, ff = valley_full(128, 129), 5.0
    elif name == "random_256x257":
        z, ff = random_full(rng, 256, 257), rng.uniform(5.0, 30.0, (256, 257))
    else:
        raise SystemExit(f"unknown network {name}")
    return lambda xp=None: make_graph(z, ff=ff, xp=xp)


def build_pair(name, plot1_case, cp, *, allow_maple_source_change=False):
    if name == "plot1":
        from maple_syrup.case_import import verify_plot1_case

        case = verify_plot1_case(plot1_case, allow_maple_source_change=allow_maple_source_change)
        binding = {"case_dir": str(plot1_case), "allow_maple_source_change": allow_maple_source_change,
                   "maple_dependency": jsonable(getattr(case, "maple_dependency", None)),
                   "report_mahleran": case.report.get("mahleran") if isinstance(case.report, dict) else None}
        return (routing.plot1_routing_graph(case.fields, case.report),
                routing.plot1_routing_graph(case.fields, case.report, xp=cp), binding)
    build = host_graph_spec(name)
    return build(), build(cp), None


def cuda_sweep_events(cp, graph, base_d, iterations, per_level):
    """Whole-sequence CUDA event time (ms), host enqueue time (s) and, optionally, per-level event intervals (ms)."""
    ctx = routing_cuda.prepare_cuda_routing(graph)
    kernel = routing_cuda._get_kernel()
    n = ctx.n_active
    outs = tuple(cp.empty(n, dtype=np.float64) for _ in range(4))
    c64, it32, n64 = np.float64(C), np.int32(iterations), np.int64(n)
    levels = [(a, b) for a, b in zip(ctx.level_bounds[:-1], ctx.level_bounds[1:], strict=True) if b > a]
    start, end = cp.cuda.Event(), cp.cuda.Event()
    marks = [cp.cuda.Event() for _ in range(len(levels) + 1)] if per_level else []
    start.record()
    t0 = time.perf_counter()
    if per_level:
        marks[0].record()
    for i, (b0, b1) in enumerate(levels):
        m = b1 - b0
        kernel(((m + routing_cuda.BLOCK_THREADS - 1) // routing_cuda.BLOCK_THREADS,), (routing_cuda.BLOCK_THREADS,),
               (np.int64(b0), np.int64(m), n64, ctx.conveyance_lo, ctx.donor_position, ctx.donor_mask, base_d, c64,
                it32, *outs))
        if per_level:
            marks[i + 1].record()
    enqueue = time.perf_counter() - t0
    end.record()
    end.synchronize()
    total_ms = cp.cuda.get_elapsed_time(start, end)
    lv = [cp.cuda.get_elapsed_time(marks[i], marks[i + 1]) for i in range(len(levels))] if per_level else None
    return total_ms, enqueue, lv


def time_wall(cp, fn, sync):
    t0 = time.perf_counter()
    fn()
    if sync:
        cp.cuda.get_current_stream().synchronize()
    return time.perf_counter() - t0


def memory_after_one_call(cp, fn):
    pool = cp.get_default_memory_pool()
    cp.cuda.get_current_stream().synchronize()
    pool.free_all_blocks()
    free0, total = cp.cuda.runtime.memGetInfo()
    used0 = pool.used_bytes()
    fn()
    cp.cuda.get_current_stream().synchronize()
    free1, _ = cp.cuda.runtime.memGetInfo()
    rec = {"scope": "pool reserved / device free change after one call; not true peak used",
           "pool_total_bytes_after": int(pool.total_bytes()), "pool_used_bytes_before": int(used0),
           "device_free_delta_bytes": int(free0 - free1), "device_total_bytes": int(total)}
    pool.free_all_blocks()
    return rec


def run_network(cp, name, wet_fractions, args, report_kernel_load):
    host, dev, case_binding = build_pair(name, args.plot1_case, cp,
                                         allow_maple_source_change=args.allow_maple_source_change)
    before = mb.read_transfer_counters()
    ctx = routing_cuda.prepare_cuda_routing(dev)  # compile/load already done by the caller; this is graph-only
    prep = ctx.summary()
    prep["counter_delta"] = dataclasses.asdict(mb.read_transfer_counters().delta(before))
    n = ctx.n_active
    record = {"network": name, "shape": list(dev.shape), "n_active": n, "n_levels": dev.n_levels,
              "max_level_width": dev.max_level_width, "level_widths": [b - a for a, b in zip(
                  dev.level_bounds[:-1], dev.level_bounds[1:], strict=True)] if n <= 5000 else "omitted (large)",
              "preparation": prep, "kernel_load": report_kernel_load, "case_binding": case_binding,
              "state_note": "random states on this topology (not captured storm states); for Plot1 the topology is "
                            "the real verified graph", "cases": []}
    rng = np.random.default_rng(7)
    sweep_cpu = rn.compiled_sweep_batched()
    for frac in wet_fractions:
        base = rng.uniform(1e-4, 3e-3, n) * (rng.random(n) < frac)
        base_d = cp.asarray(base)
        h_old = rng.uniform(1e-4, 2e-3, dev.shape) * (rng.random(dev.shape) < frac)
        h_start = h_old + rng.uniform(0.0, 1e-3, dev.shape)
        hs_d, ho_d = cp.asarray(h_start), cp.asarray(h_old)

        # Like the GPU inputs, CPU static/dynamic inputs remain resident during timing.
        # Only the four result arrays are allocated per invocation on either backend.
        cpu_inputs = (np.asarray(host.level_bounds, dtype=np.int64), np.array(host.conveyance_lo),
                      host.donor_position, host.donor_mask, base, C, args.iterations)

        def cpu_args(cpu_inputs=cpu_inputs):
            return (*cpu_inputs, *(np.empty(n) for _ in range(4)))

        cpu = cpu_args()
        sweep_cpu(*cpu)
        got = routing_cuda.run_sweep(dev, base_d, C, args.iterations)
        arr = routing._sweep_array(dev, base_d, C, args.iterations, cp)
        parity = {}
        for label, ref, g, a in zip(("qin", "q", "flow", "rhs"), cpu[7:], got, arr, strict=True):
            g_h = cp.asnumpy(g)
            parity[label] = {"cuda_vs_cpu_bitwise": bool(np.array_equal(bits(ref), bits(g_h))),
                             "cuda_vs_cupy_array_bitwise": bool(np.array_equal(bits(g_h), bits(cp.asnumpy(a))))}
        if not all(v["cuda_vs_cpu_bitwise"] for v in parity.values()):
            raise SystemExit(f"PARITY FAILURE on {name} wet={frac}: {parity}; not timing")

        raw = {"cuda": lambda base_d=base_d: routing_cuda.run_sweep(dev, base_d, C, args.iterations),
               "cupy_array": lambda base_d=base_d: routing._sweep_array(dev, base_d, C, args.iterations, cp),
               "cpu_batched": lambda cpu_args=cpu_args: sweep_cpu(*cpu_args())}
        whole = {"cuda": lambda hs_d=hs_d, ho_d=ho_d: routing.route_step(dev, hs_d, ho_d, 1.0, implementation="cuda"),
                 "cupy_array":
                     lambda hs_d=hs_d, ho_d=ho_d: routing.route_step(dev, hs_d, ho_d, 1.0, implementation="array"),
                 "numba_wrapper_original_serial_sweep":
                     lambda h_start=h_start, h_old=h_old: routing.route_step(
                         host, h_start, h_old, 1.0, implementation="numba")}
        timing = {}
        for group, table in (("raw_sweep", raw), ("route_step", whole)):
            names = list(table)
            samples = {nm: [] for nm in names}
            for nm in names:  # warm-up (CuPy array path is expensive: fewer warm-ups and repeats)
                for _ in range(args.array_warmup if nm == "cupy_array" else args.warmup):
                    table[nm]()
                cp.cuda.get_current_stream().synchronize()
            reps = max(args.repeats, 1)
            for r in range(reps):
                for nm in names[r % len(names):] + names[:r % len(names)]:
                    if nm == "cupy_array" and r >= args.array_repeats:
                        continue
                    samples[nm].append(time_wall(cp, table[nm], nm in ("cuda", "cupy_array")))
            timing[group] = {nm: stats(s) for nm, s in samples.items()}
        events = []
        for r in range(args.repeats):
            events.append(cuda_sweep_events(cp, dev, base_d, args.iterations, per_level=False))
        timing["cuda_sweep_events"] = {"total_ms": stats(e[0] for e in events),
                                       "host_enqueue_s": stats(e[1] for e in events)}
        per = [cuda_sweep_events(cp, dev, base_d, args.iterations, per_level=True) for _ in range(max(3, args.repeats // 3))]
        lv = np.array([p[2] for p in per])
        timing["cuda_per_level_events_instrumented"] = {
            "note": "events between launches perturb timing; each entry is kernel plus launch gap for one level",
            "total_ms": stats(p[0] for p in per), "per_level_median_ms": np.median(lv, axis=0).tolist()
            if n <= 5000 else {"median_of_level_medians": float(np.median(np.median(lv, axis=0))),
                               "max_level_median": float(np.max(np.median(lv, axis=0)))}}
        memory = {"cuda_sweep": memory_after_one_call(cp, raw["cuda"]),
                  "cupy_array_sweep": memory_after_one_call(cp, raw["cupy_array"]),
                  "cuda_route_step": memory_after_one_call(cp, whole["cuda"]),
                  "cupy_array_route_step": memory_after_one_call(cp, whole["cupy_array"])}
        record["cases"].append({"wet_fraction": frac, "base_positive_fraction": float(np.mean(base > 0.0)),
                                "parity_sweep": parity, "timing_s": timing, "memory": memory})
        t = timing
        print(f"{name:16s} wet={frac:4.2f} raw median ms: " + "  ".join(
            f"{k}={v['median'] * 1e3:9.3f}" for k, v in t["raw_sweep"].items()) + " | route_step: " + "  ".join(
            f"{k.split('_')[0]}={v['median'] * 1e3:9.3f}" for k, v in t["route_step"].items()))
    return record


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--output", type=Path)
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--array-repeats", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--array-warmup", type=int, default=3)
    ap.add_argument("--iterations", type=int, default=40)
    ap.add_argument("--wet-fractions", default="1.0,0.1")
    ap.add_argument("--networks", default="valley_60x21,valley_128x129,random_256x257")
    ap.add_argument("--plot1-case", type=Path, help="verified Plot1 import directory; ALSO adds the 'plot1' network to --networks "
                                                       "if it is not listed")
    ap.add_argument("--allow-maple-source-change", action="store_true",
                    help="Explicitly allow the verified case to use the currently pinned MAPLE revision.")
    args = ap.parse_args(argv)
    if args.repeats < 1 or args.warmup < 1 or args.array_warmup < 1 or args.array_repeats < 1:
        raise SystemExit("repeats and warm-ups must be >= 1")
    if not rn.numba_available():
        raise SystemExit("Numba is required for the CPU rows")
    if not mb.gpu_execution_available():
        raise SystemExit("no CUDA device available")
    cp = mb.cupy_module()
    wet = [float(f) for f in args.wet_fractions.split(",")]
    networks = [s for s in args.networks.split(",") if s]
    if args.plot1_case is not None and "plot1" not in networks:
        networks.insert(0, "plot1")
    if "plot1" in networks and args.plot1_case is None:
        raise SystemExit("--plot1-case is required for the plot1 network")

    t0 = time.perf_counter()
    kernel = routing_cuda._get_kernel()
    routing_cuda._load_kernel(kernel)
    kernel_load = {"compile_and_load_s": time.perf_counter() - t0,
                   "note": "first use in this process; CuPy's on-disk kernel cache may make this a cache load"}
    records = [run_network(cp, nm, wet, args, kernel_load if i == 0 else None) for i, nm in enumerate(networks)]
    dev = cp.cuda.Device()
    report = {
        "method": "warm, bitwise-parity-checked, alternating order, perf_counter with stream sync, CUDA events",
        "iterations": args.iterations, "c": C, "repeats": args.repeats, "array_repeats": args.array_repeats,
        "warmup": args.warmup, "array_warmup": args.array_warmup,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "visible_device_id": int(dev.id),
        "allow_maple_source_change": args.allow_maple_source_change,
        "kernel_provenance": routing_cuda.kernel_provenance(), "numba": rn.numba_versions(),
        "platform": platform.platform(), "python": sys.version, "numpy": np.__version__,
        "scope": "routing sweep and route_step only; no infiltration, sediment, storm or full-GPU claim",
        "timing_scope_notes": {
            "raw_sweep_wall": "perf_counter around one call, plus a stream synchronize for the GPU rows only. The cuda "
                              "row includes the cache-hit validation in run_sweep and the four output allocations; "
                              "the CPU row includes its four np.empty output allocations (inputs are resident).",
            "cuda_sweep_events": "CUDA events around the direct launch loop only (no cache-hit validation, no output "
                                 "allocation); NOT the same scope as the wall rows and not comparable to the CPU rows.",
            "per_level_events": "instrumented; events between launches perturb timing.",
            "route_step_rows": "include the shared validation/reductions and the one flag read. The CPU row is the "
                               "ORIGINAL SERIAL sweep inside route_step, not the batched sweep or the prepared "
                               "hydrology; it is not a statement about the production CPU default.",
            "states": "random states on the stated topology, not captured storm states.",
            "memory": "CuPy pool reserved bytes and device free-memory change after ONE call, not true peak used.",
            "not_measured": "infiltration, sediment, a full storm, fully prepared/resident hydrology",
        },
        "provenance": {
            "baseline_commit": "031bce7 (Phase 4R work is uncommitted; the working tree is modified relative to it)",
            "source_sha256": {p: file_sha256(ROOT / p) for p in (
                "src/maple_syrup/routing_cuda.py", "src/maple_syrup/routing.py", "src/maple_syrup/storm.py",
                "src/maple_syrup/routing_numba.py", "benchmarks/phase4r/bench_cuda_routing.py")},
        },
        "records": records,
    }
    if args.output:
        args.output.write_text(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
