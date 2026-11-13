"""Interleaved A/B of the COMMON Numba column: pre-A3 kernel (evidence copy) vs the new unified kernel, same inputs, warmed, alternating calls so drift/thermal noise hits both equally. NOT a pytest module."""
from __future__ import annotations

import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conftest import PRE_A3, ss  # noqa: E402


def main(out):
    spec = importlib.util.spec_from_file_location("old_nb_ab", PRE_A3)
    old = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = old
    spec.loader.exec_module(old)
    rng = np.random.default_rng(0)
    rows = []
    for L, mode, bottom in ((3, "constant", "closed"), (5, "constant", "open"), (5, "tietjen_donor_depth", "closed")):
        for N in (1_000, 10_000, 200_000):
            edges = np.concatenate([[0.0], np.cumsum(np.linspace(0.08, 0.3, L))])
            kw = dict(diffusivity_m2_s=np.linspace(2e-7, 9e-7, L - 1)) if mode == "constant" else dict(interface_conductivity_m_s=np.linspace(1e-6, 8e-6, L - 1), diffusion_coefficient=0.25)
            lay = ss.soil_layers(edges, theta_residual=np.linspace(0.01, 0.04, L), theta_field_capacity=np.linspace(0.18, 0.25, L), theta_saturation=np.linspace(0.4, 0.5, L),
                                 theta_evaporation_stress=0.5, diffusion_mode=mode, bottom_drainage=bottom, **kw)
            mask = rng.random(N) < 0.9
            col = ss.prepare_column(lay, mask, 4.0, backend="numba")
            old_run, _ = old.prepare_runner(lay)
            w = lay.saturation_m[:, None] * rng.random((L, N))
            add = rng.random((L, N)) * 1e-3 * mask
            pet = rng.random(N) * 1e-7 * mask
            reps = {1_000: 400, 10_000: 100, 200_000: 25}[N]
            to, tn = [], []
            for _ in range(5):  # warm both
                old_run(col, w, add, pet, 3600.0)
                col.runner(col, w, add, pet, 3600.0)
            for _ in range(reps):
                a = time.perf_counter(); old_run(col, w, add, pet, 3600.0); to.append(time.perf_counter() - a)
                a = time.perf_counter(); col.runner(col, w, add, pet, 3600.0); tn.append(time.perf_counter() - a)
            mo, mn = statistics.median(to) * 1e3, statistics.median(tn) * 1e3
            rows.append({"layers": L, "mode": mode, "bottom": bottom, "cells": N, "pre_a3_median_ms": mo, "new_median_ms": mn, "ratio_new_over_old": mn / mo, "repeats": reps,
                         "pre_a3_min_ms": min(to) * 1e3, "new_min_ms": min(tn) * 1e3})
            print(json.dumps(rows[-1]), flush=True)
    Path(out).write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
