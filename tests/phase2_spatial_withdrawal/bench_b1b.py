"""Modest CPU timings (NOT a pytest module): (a) interleaved A/B of the COMMON compiled withdrawal body before/after it was split into the shared inline helpers; (b) warmed spatial withdrawal (NumPy, Numba;
shared and per-cell layouts) next to the accepted common operator run on a DIFFERENT illustrative workload (separate uniform properties, `cw = min(w, 0.9 saturation)`, `cq = 0.1 q`) -- NOT a matched same-data cost comparison, so no spatial/common overhead ratio may be derived from the `illustrative_common_*` figures; cold Numba build time in a fresh process. Single thread, small kernels; no storm, LEAF, moving-bed or scaling claim."""
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
from conftest import BIG, a2, a3, sp, ss, st, sw  # noqa: E402

OLD = HERE.parents[2] / "s2_b1_b_pre_sources" / "subsurface_transpiration.py"


def med(fn, reps):
    for _ in range(3):
        fn()
    xs = []
    for _ in range(reps):
        a = time.perf_counter()
        fn()
        xs.append((time.perf_counter() - a) * 1e3)
    return statistics.median(xs)


def main(out):
    spec = importlib.util.spec_from_file_location("old_subsurface_transpiration_pre_b1b", OLD)
    old = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = old
    spec.loader.exec_module(old)
    rng = np.random.default_rng(0)
    res = {"common_numba_ab": [], "spatial": []}
    for L in (3, 5, 9):
        for N, reps in ((10_000, 400), (200_000, 60)):
            r = np.sort(rng.uniform(0, 0.01, L)); sb = r + rng.uniform(0.05, 0.2, L); act = (rng.random(N) < 0.9).astype(np.uint8)
            w = sb[:, None] * rng.random((L, N)); q = rng.uniform(0, 0.05, (L, N)) * act[None, :]
            o = [np.empty((L, N)) for _ in range(4)]; rw = np.empty((8, N))
            a, b = [], []
            for f in (old._fused_numba, st._fused_numba):
                for _ in range(5):
                    f(w, q, r, sb, act, *o, rw)
            for _ in range(reps):
                t = time.perf_counter(); old._fused_numba(w, q, r, sb, act, *o, rw); a.append(time.perf_counter() - t)
                t = time.perf_counter(); st._fused_numba(w, q, r, sb, act, *o, rw); b.append(time.perf_counter() - t)
            res["common_numba_ab"].append({"L": L, "N": N, "old_ms": statistics.median(a) * 1e3, "new_ms": statistics.median(b) * 1e3, "ratio": statistics.median(b) / statistics.median(a)})
            print(json.dumps(res["common_numba_ab"][-1]), flush=True)
    for L in (3, 5):
        for N, reps in ((10_000, 60), (200_000, 12)):
            mask = rng.random(N) < 0.9
            for label, kinds in (("shared", dict(edges="shared", res="shared", fc="shared", sat="shared", iface="shared")), ("cell", dict(edges="cell", res="cell", fc="cell", sat="cell", iface="cell"))):
                inputs, p = a3.layout_case(rng, L, N, **kinds)
                w = a2.water_kind(rng, p, "random"); q = rng.uniform(0, 0.05, (L, N)) * mask
                row = {"L": L, "N": N, "layout": label}
                for be in ("numpy", "numba"):
                    col = a2.prepare(a3.spec_from(inputs, N), backend=be, mask=mask)
                    op = sw.prepare_spatial_withdrawal(col, memory_budget_bytes=BIG)
                    stt = sp.spatial_column_state(col, w, 0.0, memory_budget_bytes=BIG)
                    row[f"spatial_{be}_ms"] = med(lambda: sw.spatial_withdraw(op, stt, q, memory_budget_bytes=BIG), reps)
                lay = ss.soil_layers(np.concatenate([[0.0], np.cumsum(np.full(L, 0.1))]), theta_residual=0.02, theta_field_capacity=0.2, theta_saturation=0.4, theta_evaporation_stress=0.5, diffusivity_m2_s=np.full(max(L - 1, 1), 1e-7) if L > 1 else 1e-7)
                for be in ("numpy", "numba"):
                    ccol = ss.prepare_column(lay, mask, 4.0, backend=be)
                    cop = st.prepare_transpiration(ccol)
                    cw = np.minimum(w, lay.saturation_m[:, None] * 0.9)
                    cst = ss.column_state(ccol, cw, 0.0)
                    cq = q * 0.1
                    row[f"illustrative_distinct_workload_common_{be}_ms"] = med(lambda: st.withdraw_transpiration(cop, cst, cq), reps)
                res["spatial"].append(row)
                print(json.dumps(row), flush=True)
    Path(out).write_text(json.dumps(res, indent=2) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
