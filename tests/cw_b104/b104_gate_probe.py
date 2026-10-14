"""CW-B104 resource probe of the per-step net-elevation gate (values-only JSON; asserts nothing).

On the dense 4 x 5 valley and the B102 seeded resident-window 64 x 32 valley, in the process's namespace:
  * the held routing source: bytes, dtype, namespace (one persistent `(ny, nx)` float64 array);
  * one gate call (`sediment_event.routing_gate_m`): counted MAPLE transfers (scalar reads, device<->host bytes);
  * gate wall time: the first call (includes lazy compilation of the windowed decode kernel) separately from the
    median of 50 repeated calls (synchronized on a device);
  * transient allocations of one gate call: host -- tracemalloc peak; device -- the bytes of every CuPy pool
    allocation made during the call (a MemoryHook sum; an upper bound on the transient peak) and the pool's used
    bytes before / after (nothing retained).

Usage: python -B b104_gate_probe.py --backend numpy|cupy --tmp DIR --out JSON
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from b104_compare_driver import (  # noqa: F401  (also sets the test-helper sys.path)
    TESTS,
    b102_build,
)


def sync(xp):
    if xp is not np:
        xp.cuda.Device().synchronize()


def probe(xp, state, ctx):
    from maple.core.backend import read_transfer_counters

    from maple_syrup.sediment_event import routing_gate_m

    def call():
        return routing_gate_m(state.bed, ctx, state.graph, state.routing_source_elevation_m)

    src = state.routing_source_elevation_m
    out = {"shape": list(state.graph.shape), "routing_source_bytes": int(src.nbytes), "routing_source_dtype": str(src.dtype),
           "routing_source_namespace": xp.__name__}
    sync(xp)
    t0 = time.perf_counter()
    before = read_transfer_counters()
    value = call()
    sync(xp)
    out["first_call_wall_s"] = time.perf_counter() - t0
    delta = read_transfer_counters().delta(before)
    out["gate_value_m"] = value
    before = read_transfer_counters()
    call()
    delta = read_transfer_counters().delta(before)
    out["transfers_per_call"] = {"scalar_reads": int(delta.scalar_reads), "device_to_host": int(delta.device_to_host),
                                 "device_to_host_bytes": int(delta.device_to_host_bytes),
                                 "host_to_device": int(delta.host_to_device),
                                 "host_to_device_bytes": int(delta.host_to_device_bytes)}
    walls = []
    for _ in range(50):
        sync(xp)
        t0 = time.perf_counter()
        call()
        sync(xp)
        walls.append(time.perf_counter() - t0)
    out["steady_median_wall_s"] = statistics.median(walls)
    out["steady_max_wall_s"] = max(walls)
    if xp is np:
        tracemalloc.start()
        tracemalloc.reset_peak()
        base = tracemalloc.get_traced_memory()[0]
        call()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        out["host_transient_peak_bytes"] = int(peak - base)
        out["host_retained_bytes"] = int(current - base)
    else:
        import cupy
        from cupy.cuda import memory_hook

        class Sum(memory_hook.MemoryHook):
            name = "b104_sum"

            def __init__(self):
                self.bytes = 0
                self.count = 0

            def malloc_postprocess(self, **kw):
                self.bytes += int(kw["mem_size"])
                self.count += 1

        pool = cupy.get_default_memory_pool()
        sync(xp)
        used0 = pool.used_bytes()
        hook = Sum()
        with hook:
            call()
            sync(xp)
        out["device_allocations_during_call"] = hook.count
        out["device_allocated_bytes_during_call"] = hook.bytes
        out["device_pool_used_before"] = int(used0)
        out["device_pool_used_after"] = int(pool.used_bytes())
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=("numpy", "cupy"), required=True)
    p.add_argument("--tmp", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    from maple.core.backend import resolve_backend
    from maple.core.parameters.topographic_commit import TopographicCommitSpec
    from test_coupled_backends import setup_xp
    from test_sediment_event import make_bed, valley_elevation

    xp = resolve_backend(a.backend).xp
    tmp = Path(a.tmp)
    tmp.mkdir(parents=True, exist_ok=False)
    bed, ctx = make_bed(valley_elevation(4, 5), commit_spec=TopographicCommitSpec(commit_interval_s=10.0))
    s = setup_xp(bed, ctx, xp, south_ring=valley_elevation(5, 5)[0] - 0.01)
    result = {"backend": a.backend, "dense_valley_4x5": probe(xp, s["state0"], s["ctx"])}
    ws, wctx, _, _ = b102_build(xp, tmp / "window")
    result["window_seeded_64x32"] = probe(xp, ws["state0"], wctx)
    Path(a.out).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
