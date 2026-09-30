"""Local benchmark: fixed state routing, not a coupled storm or peak RSS."""

import argparse
import json
import time
import tracemalloc

import numpy as np

from maple_syrup.routing import build_routing_graph, route_step

p = argparse.ArgumentParser()
p.add_argument("--implementation", default="array")
a = p.parse_args()
rows = []
for ny, nx in [(60, 20), (128, 128), (512, 512)]:
    z = np.repeat((np.arange(ny + 2) * 0.01)[:, None], nx + 2, axis=1)
    z[:, 0] += 10
    z[:, -1] += 10
    export = np.zeros(z.shape, dtype=bool)
    export[0, :] = True
    t = time.perf_counter()
    g = build_routing_graph(z, export, np.full((ny, nx), 21.45), 0.5)
    setup = time.perf_counter() - t
    h = np.full(g.shape, 0.001)
    kw = {} if a.implementation == "array" else {"implementation": a.implementation}
    t = time.perf_counter()
    r = route_step(g, h, h, 1.0, **kw)
    first = time.perf_counter() - t
    elapsed = []
    for _ in range(3):
        t = time.perf_counter()
        r = route_step(g, h, h, 1.0, **kw)
        elapsed.append(time.perf_counter() - t)
    tracemalloc.start()
    route_step(g, h, h, 1.0, **kw)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rows.append(
        {
            "shape": g.shape,
            "levels": len(g.level_bounds) - 1,
            "max_width": g.max_level_width,
            "setup_s": setup,
            "first_step_s": first,
            "warm_step_s": elapsed,
            "extra_traced_peak_bytes": peak,
            "residual_m3": float(r.budget_residual_m3),
        }
    )
print(
    json.dumps(
        {
            "implementation": a.implementation,
            "scope": "synthetic planar routing-only fixed states; extra traced memory excludes existing state/graph and compiler native memory",
            "results": rows,
        },
        indent=2,
    )
)
