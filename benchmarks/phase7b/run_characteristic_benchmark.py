"""Frozen Plot1 characteristic benchmark with low-overhead phase timers.

Run only after the candidate source is frozen. The normal runner binds source,
case and reference provenance and refuses an existing output directory.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import time
from pathlib import Path

import numpy as np

import maple_syrup.sediment_event as event
from maple_syrup.benchmark_experiment import run_plot1_matched_benchmark

parser = argparse.ArgumentParser()
parser.add_argument("--bins", type=int, default=32)
parser.add_argument("--dt", type=float, default=1.0)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    raise SystemExit("new output directory required")

timers = {}
originals = {}
for name in ("coupled_step", "sediment_physics_step", "characteristic_step", "apply_bed_demand", "with_water"):
    fn = getattr(event, name)
    originals[name] = fn

    def wrap(fn=fn, name=name):
        @functools.wraps(fn)
        def measured(*a, **kw):
            row = timers.setdefault(name, {"calls": 0, "wall_s": 0.0})
            label = name
            if name == "apply_bed_demand":
                label = "maple_pickup" if row["calls"] % 2 == 0 else "maple_deposit_export"
            start = time.perf_counter()
            try:
                return fn(*a, **kw)
            finally:
                elapsed = time.perf_counter() - start
                row["calls"] += 1
                row["wall_s"] += elapsed
                if label != name:
                    sub = timers.setdefault(label, {"calls": 0, "wall_s": 0.0})
                    sub["calls"] += 1
                    sub["wall_s"] += elapsed
        return measured

    setattr(event, name, wrap())

original_attempt = event.sediment_coupled_step
peak = {"outlet_discharge_m3_s": -1.0}


@functools.wraps(original_attempt)
def observe_attempt(*a, **kw):
    result = original_attempt(*a, **kw)
    route = result.storm.route
    q = float(route.outlet_discharge_m3_s)
    if q > peak["outlet_discharge_m3_s"]:
        peak.update(outlet_discharge_m3_s=q, t_s=a[0].storm.t_s + a[6],
                    depth_m=route.depth_m.copy(), velocity_m_s=route.velocity_m_s.copy())
    return result


event.sediment_coupled_step = observe_attempt
try:
    run = run_plot1_matched_benchmark(
        "outputs/plot1", args.output,
        applied_rainfall_csv="outputs/phase7/mahleran_reference_audit/applied_rainfall.csv",
        reference_run_dir="outputs/phase7/mahleran_deterministic_ksat_run",
        max_dt_s=args.dt, implementation="numba", transport_scheme="characteristic", phase_bins=args.bins,
    )
finally:
    for name, fn in originals.items():
        setattr(event, name, fn)
    event.sediment_coupled_step = original_attempt

assert run.result.n_rejected_attempts == 0, "peak observer requires no rejected attempts"
assert timers["apply_bed_demand"]["calls"] == 2 * run.result.n_accepted_steps
assert peak["t_s"] == float(run.result.time_of_peak_outlet_s)
assert peak["outlet_discharge_m3_s"] == float(run.result.peak_outlet_discharge_m3_s)
np.savez(args.output / "peak_outlet_snapshot.npz", **peak)
report = {"component_timers": timers, "performance": run.summary["performance"],
          "source": run.summary["provenance"], "closure": run.result.closure(),
          "phase_bins": args.bins, "max_dt_s": args.dt,
          "measurement_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          "measurement_script_text": Path(__file__).read_text(),
          "note": "MAPLE subrows overlap apply_bed_demand. CPU run; timers and peak snapshot observe unchanged returns."}
(args.output / "component_times.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({"output": str(args.output), "sediment": run.summary["sediment"]["totals"],
                  "component_timers": timers}, indent=2))
