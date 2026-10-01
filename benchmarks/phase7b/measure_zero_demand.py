"""Measure an unchanged shared MAPLE transaction with a zero demand.

An isolated API probe on the imported initial bed, not a storm speedup claim.
"""
from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path

import numpy as np
from maple.water import WaterProcessDemand

from maple_syrup.case_import import verify_plot1_case
from maple_syrup.sediment_bed import apply_bed_demand, bed_from_case

verified = verify_plot1_case("outputs/plot1")
bed, context = bed_from_case(verified.case, xp=np)
zero = np.zeros_like(bed.water.mobile_mass_by_cell_class_kg)
demand = WaterProcessDemand(zero, zero)


def arrays(value, prefix=""):
    if isinstance(value, np.ndarray):
        return {prefix: value}
    result = {}
    if dataclasses.is_dataclass(value):
        for field in dataclasses.fields(value):
            result.update(arrays(getattr(value, field.name), prefix + "." + field.name))
    return result


original = {key: value.copy() for key, value in arrays(bed).items()}
for _ in range(3):
    after, transfer = apply_bed_demand(bed, context, demand)
times = []
for _ in range(100):
    start = time.perf_counter()
    after, transfer = apply_bed_demand(bed, context, demand)
    times.append(time.perf_counter() - start)
result_arrays = arrays(after)
assert set(original) == set(result_arrays)
differences = [key for key in original if not np.array_equal(original[key], result_arrays[key])]
delta = {key: {"max_abs_kg": float(np.max(np.abs(result_arrays[key] - original[key]))),
               "signed_sum_kg": float(np.sum(result_arrays[key] - original[key])),
               "absolute_sum_kg": float(np.sum(np.abs(result_arrays[key] - original[key])))}
         for key in differences}
assert all(np.array_equal(original[key], value) for key, value in arrays(bed).items())
report = {
    "scope": __doc__, "calls": len(times),
    "median_s": float(np.median(times)), "mean_s": float(np.mean(times)),
    "min_s": min(times), "max_s": max(times),
    "bed_arrays_compared": len(original), "changed_array_paths": differences,
    "changed_array_deltas": delta,
    "input_arrays_unchanged": True,
    "maple_package_digest": verified.binding["maple_source_digest_after"],
    "qualification": "Only initial Plot1 bed; not proof a zero demand is a no-op on arbitrary states or that safely skipping it achieves this saving."
}
Path("benchmarks/phase7b/zero_demand_measurement.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({k: v for k, v in report.items() if k != "maple_package_digest"}, indent=2))
