"""Phase 7j sweep micro-benchmark: original serial sweep vs a dry-skip-only serial sweep vs the level-batched sweep.

    python benchmarks/phase7j/bench_sweep.py [--output out.json] [--repeats 30] [--wet-fractions 0,0.45,1]

Synthetic grids (valley and random terrain from the Phase 4 test builders) with a chosen fraction of cells holding
water (the remainder have a zero own base, so only donor-fed cells are positive). Captured Plot1 states are not
read by this script. Every candidate is first checked BIT FOR BIT against the original sweep (uint64 views); a mismatch aborts without
timing. Timing is warm (compilation excluded), candidates alternate in every repetition, medians/min/IQR are
reported. The report also records platform/CPU flags and whether the compiled batched sweep contains packed
double-precision sqrt (`vsqrtpd`/`sqrtpd`) instructions. No speed or vectorization claim is encoded here: the numbers
are whatever the machine produced. Run it on an otherwise idle machine. CPU only; no GPU code.
"""
from __future__ import annotations

import argparse
import json
import platform
import re
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "phase4"))
sys.path.insert(0, str(ROOT / "src"))

from maple_syrup import routing_numba as rn

from test_routing import make_graph, random_full, valley_full  # isort: skip


def dry_skip_only_sweep():
    """Serial per-cell sweep identical to `_sweep` except that cells with `not (rhs > 0.0)` skip the iterations
    (attribution variant for this benchmark only; never imported by the package)."""
    import numba

    def _sweep_skip(bounds, k_lo, donor_position, donor_mask, base_lo, c, iterations,
                    qin_new_lo, q_new_lo, flow_lo, rhs_lo):
        for lev in range(bounds.shape[0] - 1):
            for p in range(bounds[lev], bounds[lev + 1]):
                qin = 0.0
                for s in range(4):
                    if donor_mask[s, p]:
                        qin = qin + q_new_lo[donor_position[s, p]]
                    else:
                        qin = qin + 0.0
                rhs = base_lo[p] + qin * c
                k = k_lo[p]
                lo = 0.0
                if rhs > 0.0:
                    w = rhs
                    for _ in range(iterations):
                        w = w * 0.5
                        mid = lo + w
                        t = np.sqrt(mid)
                        t = t * mid
                        t = t * k
                        t = t * c
                        t = t + mid
                        if t < rhs:
                            lo = mid
                q = np.sqrt(lo)
                q = q * lo
                q = q * k
                qin_new_lo[p] = qin
                q_new_lo[p] = q
                flow_lo[p] = lo
                rhs_lo[p] = rhs

    return numba.njit(cache=False, fastmath=False, nogil=True, boundscheck=False)(_sweep_skip)


def make_args(graph, base, c, iterations):
    n = graph.n_active
    outs = tuple(np.zeros(n) for _ in range(4))
    return (np.asarray(graph.level_bounds, dtype=np.int64), np.array(graph.conveyance_lo), graph.donor_position,
            graph.donor_mask, np.array(base, dtype=np.float64), float(c), int(iterations), *outs)


def graphs():
    rng = np.random.default_rng(1)
    return {
        "valley_60x21": make_graph(valley_full(60, 21), ff=5.0),
        "valley_128x129": make_graph(valley_full(128, 129), ff=5.0),
        "random_256x257": make_graph(random_full(rng, 256, 257), ff=rng.uniform(5.0, 30.0, (256, 257))),
    }


def bits(a):
    return np.asarray(a).view(np.uint64)


def time_candidates(candidates, graph, base, c, iterations, repeats):
    arg = {name: make_args(graph, base, c, iterations) for name in candidates}
    for name, fn in candidates.items():  # warm-up and parity (original is `serial`)
        fn(*arg[name])
    for name in candidates:
        for ref, got, label in zip(arg["serial"][7:], arg[name][7:], ("qin", "q", "flow", "rhs"), strict=True):
            if not np.array_equal(bits(ref), bits(got)):
                raise SystemExit(f"PARITY FAILURE: {name} differs from the original sweep in {label}; not timing")
    samples = {name: [] for name in candidates}
    names = list(candidates)
    for r in range(repeats):
        order = names[r % len(names):] + names[:r % len(names)]  # alternate who goes first
        for name in order:
            a = arg[name]
            t0 = time.perf_counter()
            candidates[name](*a)
            samples[name].append(time.perf_counter() - t0)
    out = {}
    for name, s in samples.items():
        q = statistics.quantiles(s, n=4) if len(s) >= 4 else [min(s), statistics.median(s), max(s)]
        out[name] = {"median_s": statistics.median(s), "min_s": min(s), "iqr_s": q[2] - q[0], "samples_s": s}
    return out


def asm_report(dispatcher):
    """Packed-instruction census of the compiled batched sweep (what the compiler emitted, not a speed claim)."""
    text = "\n".join(dispatcher.inspect_asm().values())
    return {
        "packed_sqrt_pd": len(re.findall(r"\bv?sqrtpd\b", text)),
        "scalar_sqrt_sd": len(re.findall(r"\bv?sqrtsd\b", text)),
        "packed_mul_pd": len(re.findall(r"\bv?mulpd\b", text)),
        "scalar_mul_sd": len(re.findall(r"\bv?mulsd\b", text)),
        "fma_instructions": len(re.findall(r"\bvfn?m(?:add|sub)\w*\b", text)),
    }


def cpu_flags():
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("flags"):
                have = set(line.split(":", 1)[1].split())
                return sorted(have & {"sse2", "sse4_2", "avx", "avx2", "fma", "avx512f", "avx512dq", "avx512vl"})
    except OSError:
        pass
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--output", type=Path)
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--iterations", type=int, default=40)
    ap.add_argument("--wet-fractions", default="0,0.45,1.0")
    ap.add_argument("--c", type=float, default=0.5 / 3.0)
    args = ap.parse_args(argv)
    if not rn.numba_available():
        raise SystemExit("Numba is required")
    candidates = {"serial": rn.compiled_sweep(), "serial_dry_skip": dry_skip_only_sweep(),
                  "batched": rn.compiled_sweep_batched()}
    rng = np.random.default_rng(7)
    records = []
    for gname, graph in graphs().items():
        n = graph.n_active
        for frac in (float(f) for f in args.wet_fractions.split(",")):
            base = rng.uniform(1e-4, 3e-3, n) * (rng.random(n) < frac)
            timing = time_candidates(candidates, graph, base, args.c, args.iterations, args.repeats)
            records.append({"graph": gname, "n_active": n, "n_levels": graph.n_levels,
                            "max_level_width": graph.max_level_width, "wet_fraction": frac,
                            "base_positive_fraction": float(np.mean(base > 0.0)), "timing": timing})
            med = {k: v["median_s"] for k, v in timing.items()}
            print(f"{gname:16s} wet={frac:4.2f}  " + "  ".join(f"{k}={v * 1e3:8.3f}ms" for k, v in med.items()))
    report = {"method": "warm, bitwise-checked, alternating order, perf_counter; no claims encoded",
              "iterations": args.iterations, "c": args.c, "repeats": args.repeats, "records": records,
              "asm_batched": asm_report(candidates["batched"]), "numba": rn.numba_versions(),
              "platform": platform.platform(), "machine": platform.machine(), "python": sys.version,
              "numpy": np.__version__, "cpu_flags": cpu_flags()}
    if args.output:
        args.output.write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
