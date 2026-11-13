"""Modest CPU timing/memory comparison of the pre-B1.a and refactored common withdrawal bodies (NOT a pytest module): interleaved warmed medians for `_fused_numpy` and `_fused_numba`, the Numba
compile/first-call time per module in a fresh process, and the traced NumPy peak. Single thread, no GPU, small kernels only; no whole-storm or scaling claim."""
from __future__ import annotations

import importlib.util
import json
import statistics
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from conftest import OLD_PATH  # noqa: E402


def main(out):
    spec = importlib.util.spec_from_file_location("old_subsurface_transpiration_pre_b1a", OLD_PATH)
    old = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = old
    spec.loader.exec_module(old)
    from maple_syrup import subsurface_transpiration as new

    rng = np.random.default_rng(0)
    rows = []
    first = {}
    for L in (3, 5, 9):
        for N in (10_000, 200_000):
            res = np.sort(rng.uniform(0, 0.01, L)); satb = res + rng.uniform(0.05, 0.2, L)
            act = (rng.random(N) < 0.9).astype(np.uint8)
            w = satb[:, None] * rng.random((L, N)); q = rng.uniform(0, 0.05, (L, N)) * act[None, :]
            outs = {m: ([np.empty((L, N)) for _ in range(4)], np.empty((8, N))) for m in ("old", "new")}
            for fn in ("_fused_numpy", "_fused_numba"):
                fo, fnw = getattr(old, fn), getattr(new, fn)
                if fn == "_fused_numba":  # first call (compile or cache load) timed per module
                    for label, f in (("old", fo), ("new", fnw)):
                        if (label, L, N) in first:
                            continue
                        a = time.perf_counter(); f(w, q, res, satb, act, *outs[label][0], outs[label][1]); first[(label, L, N)] = time.perf_counter() - a
                reps = 40 if N <= 10_000 else 12
                to, tn = [], []
                for _ in range(3):
                    fo(w, q, res, satb, act, *outs["old"][0], outs["old"][1]); fnw(w, q, res, satb, act, *outs["new"][0], outs["new"][1])
                for _ in range(reps):
                    a = time.perf_counter(); fo(w, q, res, satb, act, *outs["old"][0], outs["old"][1]); to.append(time.perf_counter() - a)
                    a = time.perf_counter(); fnw(w, q, res, satb, act, *outs["new"][0], outs["new"][1]); tn.append(time.perf_counter() - a)
                row = {"fn": fn, "L": L, "N": N, "old_ms": statistics.median(to) * 1e3, "new_ms": statistics.median(tn) * 1e3, "ratio": statistics.median(tn) / statistics.median(to)}
                if fn == "_fused_numpy":
                    peaks = []
                    for f in (fo, fnw):
                        tracemalloc.start(); tracemalloc.reset_peak(); b = tracemalloc.get_traced_memory()[0]
                        f(w, q, res, satb, act, *outs["old"][0], outs["old"][1]); peaks.append(tracemalloc.get_traced_memory()[1] - b); tracemalloc.stop()
                    row["old_traced_peak_B"], row["new_traced_peak_B"] = peaks
                rows.append(row)
                print(json.dumps(row), flush=True)
    Path(out).write_text(json.dumps({"rows": rows, "first_call_s": {f"{k[0]}_L{k[1]}_N{k[2]}": v for k, v in first.items()}}, indent=2) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
