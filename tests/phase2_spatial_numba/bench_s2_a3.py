"""Modest CPU timings for the S2-A.3 compiled body (NOT a pytest module; run as `python bench_s2_a3.py <json-out>`). Compile startup (a fresh process: first `compiled_kernel()` call) is reported SEPARATELY from
warmed per-call medians. Compared: the pre-A3 common Numba kernel (evidence copy) vs the new common Numba column (same physics, same inputs), and the spatial Numba step (shared and per-cell layouts) vs the explicit
NumPy spatial path. Single thread, no GPU. A small-scale kernel timing: it makes NO whole-storm, large-domain or scaling claim."""
from __future__ import annotations

import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from conftest import PRE_A3, a2, layout_case, spec_from, sp, ss  # noqa: E402

BIG = 1 << 40


def median_ms(fn, repeats):
    fn()
    times = []
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t) * 1e3)
    return {"median_ms": statistics.median(times), "min_ms": min(times), "max_ms": max(times), "repeats": repeats}


def main(out):
    from maple_syrup import subsurface_numba as new

    t = time.perf_counter()
    new.compiled_kernel()
    compile_s = time.perf_counter() - t
    t = time.perf_counter()
    new.compiled_spatial_kernel()
    compile_spatial_s = time.perf_counter() - t
    spec = importlib.util.spec_from_file_location("old_nb_bench", PRE_A3)
    old = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = old
    spec.loader.exec_module(old)
    rng = np.random.default_rng(0)
    result = {"compile_startup_s_common_variant_fresh_process": compile_s, "compile_startup_s_spatial_variant_after_common": compile_spatial_s, "numba_versions": new.numba_versions(), "cases": []}
    for L in (3, 5):
        for N in (10_000, 200_000):
            edges = np.concatenate([[0.0], np.cumsum(np.linspace(0.08, 0.3, L))])
            lay = ss.soil_layers(edges, theta_residual=np.linspace(0.01, 0.04, L), theta_field_capacity=np.linspace(0.18, 0.25, L), theta_saturation=np.linspace(0.4, 0.5, L),
                                 theta_evaporation_stress=0.5, diffusivity_m2_s=np.linspace(2e-7, 9e-7, L - 1), bottom_drainage="closed")
            mask = rng.random(N) < 0.9
            col = ss.prepare_column(lay, mask, 4.0, backend="numba")
            old_run, _ = old.prepare_runner(lay)
            w = lay.saturation_m[:, None] * rng.random((L, N))
            add = rng.random((L, N)) * 1e-3 * mask
            pet = rng.random(N) * 1e-7 * mask
            reps = 15 if N <= 10_000 else 7
            row = {"layers": L, "cells": N,
                   "common_pre_a3_kernel": median_ms(lambda: old_run(col, w, add, pet, 3600.0), reps),
                   "common_new_kernel": median_ms(lambda: col.runner(col, w, add, pet, 3600.0), reps)}
            # spatial: shared layouts and fully per-cell layouts, numba vs numpy
            for label, kinds in (("shared", dict(edges="shared", res="shared", fc="shared", sat="shared", iface="shared")), ("per_cell", dict(edges="cell", res="cell", fc="cell", sat="cell", iface="cell"))):
                inputs, p = layout_case(rng, L, N, **kinds)
                cols = {b: sp.prepare_spatial_column(spec_from(inputs, N, bottom="closed"), mask, 4.0, backend=b, memory_budget_bytes=BIG) for b in ("numba", "numpy")}
                water = a2.water_kind(rng, p, "random")
                states = {b: sp.spatial_column_state(c, water, 0.0, memory_budget_bytes=BIG) for b, c in cols.items()}
                added = rng.random((L, N)) * 1e-3 * mask
                for b in ("numba", "numpy"):
                    row[f"spatial_{label}_{b}"] = median_ms(lambda b=b: sp.spatial_step(cols[b], states[b], 3600.0, added_water_m=added, potential_evaporation_m_s=pet, memory_budget_bytes=BIG), reps)
            result["cases"].append(row)
            print(json.dumps({k: (v["median_ms"] if isinstance(v, dict) else v) for k, v in row.items()}), flush=True)
    Path(out).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
