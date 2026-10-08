"""Bounded microbenchmark: the all-class reference record strategy versus the compact one on ACTUAL wet inputs (task gpu_sediment, B2).

    python benchmarks/legacy_gpu/record_strategy_microbench.py --case-kind chastre --case <case dir> --state-npz <wet_state.npz> \
        --steps 60 --repeats 3 --output <NEW report.json> [--order all,compact | compact,all] [--allow-maple-source-change] \
        [--applied-rainfall <csv>] [--gpu-memory-gib G]

`--state-npz` holds float64 `depth_m`, `velocity_m_s`, `rain_m_s` of the case grid (ny, nx) - for example the final fields of a completed
600 s CPU/GPU run (`final_depth_m`, `final_velocity_m_s` re-saved under these names) with the applied rain rate of that time - or a leading
axis of K states that are cycled. The SAME state is fed to both strategies; the sediment pools evolve (it is a replay, not a storm).

Per strategy it separates (all exploratory, never a science or acceptance claim; Codex must select and verify an idle device before the run):
  setup      host table build + context build + compile (NVRTC, first time per (device, nc, ne))  [wall]
  warmup     one full measured-shape pass, then `reset()`                                           [wall, device events]
  loop       `--repeats` passes of `--steps` sediment steps, CUDA events around each pass, flags read once per pass
  memory     the context's estimate, `memGetInfo` allocation delta, record-class counts, the record array bytes, the pool statistics
and compares the two strategies' ledgers and maps at the predeclared sediment bound (2e-11 / 1e-14) with exact tallies/flags, and
reports the per-step launches. The order of the strategies is an option so that the result can be repeated in the opposite order.
Nothing here was run by its author.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

RTOL, ATOL = 2.0e-11, 1.0e-14


def load_state(path: Path, shape: tuple[int, int]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        out = {k: np.asarray(z[k]) for k in ("depth_m", "velocity_m_s", "rain_m_s") if k in z.files}
    if set(out) != {"depth_m", "velocity_m_s", "rain_m_s"}:
        raise SystemExit("--state-npz needs depth_m, velocity_m_s and rain_m_s")
    for key, a in out.items():
        if a.dtype != np.float64 or a.shape[-2:] != tuple(shape) or a.ndim not in (2, 3) or not np.isfinite(a).all() or (a < 0.0).any():
            raise SystemExit(f"{key} must be a finite non-negative float64 array of shape {shape} (or K x that)")
        out[key] = np.ascontiguousarray(a if a.ndim == 3 else a[None])
    if len({a.shape[0] for a in out.values()}) != 1:
        raise SystemExit("the three state arrays must have the same number of states")
    return out


def run_strategy(cp, network, physics, graph, strategy: str, state, steps: int, repeats: int, budget: int | None) -> tuple[dict, Any]:
    from maple_syrup import legacy_native as N
    from maple_syrup.legacy_native_cuda import CudaLegacyContext

    pool = cp.get_default_memory_pool()
    cp.cuda.Device().synchronize()
    free0, _ = cp.cuda.runtime.memGetInfo()
    t0 = time.perf_counter()
    ctx = CudaLegacyContext(network, physics, graph, limits=N.walk_limits(network.dx_m), dt=1.0, n_steps=steps,
                            memory_budget_bytes=budget, record_strategy=strategy)
    cp.cuda.Device().synchronize()
    setup_s = time.perf_counter() - t0
    k = state["depth_m"].shape[0]
    dev = [tuple(cp.asarray(state[key][j]) for key in ("depth_m", "velocity_m_s", "rain_m_s")) for j in range(k)]  # uploaded once

    def one_pass() -> tuple[float, float]:
        start, stop = cp.cuda.Event(), cp.cuda.Event()
        w0 = time.perf_counter()
        start.record()
        for row in range(steps):
            ctx.step(row, *dev[row % k])
        stop.record()
        stop.synchronize()
        ctx.check_flags()
        return cp.cuda.get_elapsed_time(start, stop) / 1000.0, time.perf_counter() - w0

    warm_dev, warm_wall = one_pass()
    ctx.reset()
    passes = []
    for _ in range(repeats):
        ctx.reset()
        d, w = one_pass()
        passes.append({"device_loop_s": d, "wall_including_flag_read_s": w})
    result = {"strategy": strategy, "setup_wall_s": setup_s, "compile_included_in_setup": "yes on the first use of (device, nc, ne), else cached",
              "warmup": {"device_loop_s": warm_dev, "wall_s": warm_wall}, "passes": passes,
              "device_loop_s_per_step": [p["device_loop_s"] / steps for p in passes],
              "record_strategy": ctx.record_info, "estimate_bytes": ctx.estimate, "memory": ctx.memory,
              "record_values_bytes_allocated": int(ctx.values.nbytes), "walk_table_bytes": int(ctx.tables.nbytes()),
              "launches_per_step": ctx.stats["launches"] // max(1, ctx.stats["steps"]), "transfers": dict(ctx.stats),
              "pool_used_bytes": int(pool.used_bytes()), "pool_total_bytes": int(pool.total_bytes()),
              "free_before_bytes": int(free0)}
    return result, ctx


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--case-kind", required=True, choices=("plot1", "rfid", "chastre"))
    p.add_argument("--case", type=Path, required=True)
    p.add_argument("--state-npz", type=Path, required=True)
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--order", default="all,compact", help="comma list of strategies in run order; repeat in the opposite order too")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--end-s", type=float, default=None)
    p.add_argument("--applied-rainfall", type=Path, default=None)
    p.add_argument("--allow-maple-source-change", action="store_true")
    p.add_argument("--hash-only-tile-verify", action="store_true")
    p.add_argument("--gpu-memory-gib", type=float, default=None)
    a = p.parse_args(argv)
    order = [s.strip() for s in a.order.split(",")]
    if sorted(order) != ["all", "compact"] or a.steps < 1 or a.repeats < 1 or (a.gpu_memory_gib is not None and not (
            math.isfinite(a.gpu_memory_gib) and a.gpu_memory_gib > 0.0)):
        print("--order must be 'all,compact' or 'compact,all'; steps/repeats >= 1; --gpu-memory-gib finite > 0", file=sys.stderr)
        return 1
    if os.path.lexists(a.output):
        print(f"refusing to overwrite {a.output}", file=sys.stderr)
        return 1
    from maple_syrup import legacy_native as N
    from maple_syrup.legacy_case import legacy_case_for
    from maple_syrup.legacy_physics_numba import prepare_legacy_physics
    from maple_syrup.routing_cuda import _cupy

    cp = _cupy()
    case = legacy_case_for(a.case_kind, a.case, allow_maple_source_change=a.allow_maple_source_change,
                           hash_only_tile_verify=a.hash_only_tile_verify, end_s=a.end_s, applied_rainfall=a.applied_rainfall)
    network = N.native_network(case.graph)
    physics = prepare_legacy_physics(case.sediment, case.grid, case.vegetation, case.holdings_kg)
    state = load_state(a.state_npz, tuple(case.shape))
    budget = int(a.gpu_memory_gib * 2**30) if a.gpu_memory_gib is not None else None
    results: dict[str, dict] = {}
    contexts: dict[str, Any] = {}
    for strategy in order:
        results[strategy], ctx = run_strategy(cp, network, physics, case.graph, strategy, state, a.steps, a.repeats, budget)
        contexts[strategy] = ctx
        if strategy != order[-1]:  # keep only the first context's host copies; free the device buffers before the next strategy
            contexts[strategy] = {"ledger": ctx.host_ledger.copy(), "counts": ctx.host_counts.copy(), "flags": ctx.host_flags.copy(),
                                  "maps": ctx.download_maps()}
            del ctx
            cp.get_default_memory_pool().free_all_blocks()
    last = contexts[order[-1]]
    contexts[order[-1]] = {"ledger": last.host_ledger.copy(), "counts": last.host_counts.copy(), "flags": last.host_flags.copy(),
                           "maps": last.download_maps()}
    ref, cmp = contexts["all"], contexts["compact"]
    equivalence: dict[str, Any] = {"integers_exact": bool(np.array_equal(ref["counts"], cmp["counts"]) and np.array_equal(ref["flags"], cmp["flags"]))}
    for name, x, y in (("ledger", cmp["ledger"], ref["ledger"]), *((k, cmp["maps"][k], ref["maps"][k]) for k in ref["maps"])):
        ok = bool(np.isfinite(x).all() and np.isfinite(y).all() and np.all(np.abs(x - y) <= ATOL + RTOL * np.abs(y)))
        equivalence[name] = {"within_declared_sediment_bound": ok, "max_abs": float(np.abs(x - y).max()) if x.size else 0.0,
                             "bitwise_equal": bool(x.tobytes() == y.tobytes())}
    equivalence["pass"] = bool(equivalence["integers_exact"] and all(v["within_declared_sediment_bound"] for k, v in equivalence.items()
                                                                     if isinstance(v, dict)))
    allp = [p_["device_loop_s"] for p_ in results["all"]["passes"]]
    cmpp = [p_["device_loop_s"] for p_ in results["compact"]["passes"]]
    report = {"purpose": "exploratory record-strategy microbenchmark on supplied wet inputs; not a science or acceptance result",
              "case_kind": a.case_kind, "steps": a.steps, "repeats": a.repeats, "order": order,
              "state_file": str(a.state_npz), "n_states": int(state["depth_m"].shape[0]), "results": results, "equivalence": equivalence,
              "summary": {"all_device_loop_s_median": float(np.median(allp)), "compact_device_loop_s_median": float(np.median(cmpp)),
                          "ratio_all_over_compact_median": float(np.median(allp) / np.median(cmpp)) if np.median(cmpp) > 0 else None,
                          "record_values_bytes": {"all": results["all"]["record_values_bytes_allocated"],
                                                  "compact": results["compact"]["record_values_bytes_allocated"]},
                          "note": "single device, one case; repeat in the opposite --order before any conclusion"}}
    with a.output.open("x") as handle:
        handle.write(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps({"equivalence_pass": equivalence["pass"], **report["summary"]}, indent=2, default=str))
    return 0 if equivalence["pass"] else 2


if __name__ == "__main__":
    sys.exit(main())
