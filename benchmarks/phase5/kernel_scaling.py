"""CPU wet-law/transport scaling; excludes MAPLE bed, hydrology and setup.

Run from SYRUP with its source and actual MAPLE on PYTHONPATH. Separate
tracemalloc passes measure traced incremental allocation, not total RSS.
These fixed-input kernel measurements are not full-event throughput.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import statistics
import time
import tracemalloc
from pathlib import Path

import numpy as np

from maple_syrup.routing import build_routing_graph
from maple_syrup.sediment_physics import (
    physics_grid_from_graph,
    plot1_sediment_parameters,
    sediment_physics_step,
)
from maple_syrup.sediment_transport import transport_network, transport_step


def measure(fn, repeats):
    fn()  # imports, allocation and cache warm-up are outside timing
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        samples.append(time.perf_counter() - start)
        del result
    gc.collect()
    tracemalloc.start()
    result = fn()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del result
    return {"median_s": statistics.median(samples), "samples_s": samples,
            "traced_retained_bytes": current, "traced_peak_bytes": peak}


def run(ny, nx, repeats):
    # South-draining sloping plane, side walls and all south ring cells open.
    z = np.repeat(np.arange(ny + 2, dtype=float)[:, None] * .015, nx + 2, axis=1)
    z[:, [0, -1]] += 10
    exits = np.zeros(z.shape, bool)
    exits[0, :] = True
    graph = build_routing_graph(z, exits, np.full((ny, nx), 21.45), .5)
    net = transport_network(graph)
    grid = physics_grid_from_graph(graph)
    params = plot1_sediment_parameters()
    shape = (ny, nx)
    fractions = np.array([.1, .15, .25, .25, .15, .1])
    active = np.broadcast_to(fractions, (*shape, 6)).copy()
    memory = np.zeros_like(active)
    h, v, rain, cover = (np.full(shape, value) for value in (.005, .02, 1e-5, .2))

    def physics():
        return sediment_physics_step(params, grid, h, v, rain, cover, active, memory, 1.)

    laws = physics()
    mobile = np.full_like(active, 1e-4)

    def transport():
        return transport_step(net, mobile, laws.sediment_velocity_m_s,
                              laws.deposition_rate_per_m, laws.settle_mask, 1.)

    return {"ny": ny, "nx": nx, "n_cells": ny * nx, "n_classes": 6,
            "physics": measure(physics, repeats), "transport": measure(transport, repeats)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=7)
    args = parser.parse_args()
    if args.repeats < 1 or args.output.exists():
        parser.error('positive repeats and a new output file required')
    import maple_syrup.routing as routing_module
    import maple_syrup.sediment_physics as physics_module
    import maple_syrup.sediment_transport as transport_module
    from maple_syrup.dependency import resolve_maple_dependency
    from maple_syrup.provenance import capture_maple_provenance

    sources = {str(Path(m.__file__).resolve()): hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
               for m in (physics_module, transport_module, routing_module)}
    sources[str(Path(__file__).resolve())] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    cpu_info = Path('/proc/cpuinfo')
    cpu_model = next((line.split(':', 1)[1].strip() for line in cpu_info.read_text().splitlines()
                      if line.startswith('model name')), None) if cpu_info.exists() else None
    report = {"scope": "fixed-input CPU kernels, excludes MAPLE exchange/hydrology/terrain/setup",
              "memory_scope": "incremental tracemalloc allocations during one warmed call; not RSS",
              "retained_memory_scope": "allocations still live with the call result referenced; not a leak estimate",
              "python": platform.python_version(), "numpy": np.__version__,
              "platform": platform.platform(), "source_sha256": sources,
              "cpu_model": cpu_model, "logical_cpus": os.cpu_count(), "gc_enabled": gc.isenabled(),
              "thread_environment": {k: os.environ.get(k) for k in
                                     ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')},
              "maple": capture_maple_provenance(resolve_maple_dependency()),
              "cases": [run(ny, nx, args.repeats) for ny, nx in [(60, 20), (240, 80), (480, 160)]]}
    if any(hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest for path, digest in sources.items()):
        raise RuntimeError('benchmark source changed during measurement')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
