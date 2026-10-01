"""Audit diagnostic-only reference outputs and the CN sediment identity."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np

reference = Path("outputs/phase7/mahleran_deterministic_ksat_run/Output")
run = Path("outputs/phase7b/mahleran_ledger_run_v3/Output")
identical = []
for path in sorted(reference.iterdir()):
    candidate = run / path.name
    if path.name == "param001.dat":
        # The model writes the execution clock, the only permissible difference.
        clean = lambda p: [s for s in p.read_text().splitlines() if "Output time" not in s]
        assert clean(path) == clean(candidate)
    else:
        assert path.read_bytes() == candidate.read_bytes(), path.name
        identical.append(path.name)


def number(value):
    # Fortran ES24.16 omits E for a three-digit subnormal exponent.
    return float(re.sub(r"(?<=\d)([+-]\d{3})$", r"e\1", value))


ledger = run / "syrup_sediment_ledger.dat"
data = np.loadtxt(ledger, converters=number).reshape(5400, 6, 14)
assert np.array_equal(data[:, :, 0], np.broadcast_to(np.arange(1, 5401)[:, None], (5400, 6)))
assert np.array_equal(data[:, :, 1], data[:, :, 0])
assert np.array_equal(data[:, :, 2], np.broadcast_to(np.arange(1, 7), (5400, 6)))
assert np.array_equal(data[1:, :, 7], data[:-1, :, 8])
names = ["pickup", "deposition_active", "deposition_outside_active", "effective_clip_source",
         "old_mobile", "new_mobile", "cn_export", "endpoint_export", "net_cell_flux",
         "algebra_residual", "internal_flux_residual"]
totals = {name: data[:, :, i+3].sum(axis=0).tolist() for i, name in enumerate(names)
          if name not in ("old_mobile", "new_mobile")}
report = {
    "reference": str(reference), "run": str(run), "identical_numeric_files": identical,
    "param_only_execution_clock_differs": True,
    "diagnostic_sha256": hashlib.sha256(ledger.read_bytes()).hexdigest(),
    "diagnostic_source_manifest": "outputs/phase7b/mahleran_ledger_source_v3/benchmark_manifest.json",
    "execution": "outputs/phase7b/mahleran_ledger_run_v3/execution.json",
    "sums_by_class_kg": totals,
    "initial_mobile_by_class_kg": data[0, :, 7].tolist(),
    "final_mobile_by_class_kg": data[-1, :, 8].tolist(),
    "max_abs_step_algebra_residual_kg": float(abs(data[:, :, 12]).max()),
    "max_abs_step_internal_flux_residual_kg": float(abs(data[:, :, 13]).max()),
    "meaning": "CN identity includes effective clipping source (-negative trial depth)*(1+dt*v/(2*dx)); outside-active deposition is a separate legacy diagnostic, not an additional CN sink."
}
target = Path("benchmarks/phase7b/mahleran_ledger_audit.json")
target.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
