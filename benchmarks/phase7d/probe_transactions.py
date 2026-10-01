"""Compare an isolated MAPLE voxel-extraction candidate to the accepted source.

Private-kernel differential checks complement upstream public atomicity tests.
No kernel implementation is copied into SYRUP.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
from maple.surface.voxels import transfer
from maple.surface.voxels.capacity import max_voxel_mass_kg

from maple_syrup.case_import import verify_plot1_case
from maple_syrup.sediment_bed import apply_bed_demand, bed_from_case

BASELINE = Path('outputs/dependencies/maple_d3d007024/source/src/maple/surface/voxels/transfer.py')


def load_reference():
    spec = importlib.util.spec_from_file_location('maple_transfer_reference', BASELINE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def arrays(value, prefix=''):
    if isinstance(value, np.ndarray):
        return {prefix: value}
    result = {}
    if dataclasses.is_dataclass(value):
        for field in dataclasses.fields(value):
            result.update(arrays(getattr(value, field.name), prefix + '.' + field.name))
    return result


def main(output):
    reference = load_reference()
    verified = verify_plot1_case('outputs/plot1', allow_maple_source_change=True)
    bed, context = bed_from_case(verified.case)
    rng = np.random.default_rng(893)
    comparisons = []
    for nc in (1, 2, 6, 9, 17):
        for nv in (1, 2, 20, 64):
            g = dataclasses.replace(context.geometry, ny=4, nx=3)
            capacity = max_voxel_mass_kg(g)
            mass = rng.uniform(.01, 1, (4, 3, nv, nc))
            mass *= capacity / mass.sum(-1)[..., None]
            occupied = rng.integers(0, nv + 1, size=(4, 3))
            for y in range(4):
                for x in range(3):
                    mass[y, x, occupied[y, x]:] = 0
                    if occupied[y, x]:
                        mass[y, x, occupied[y, x]-1] *= rng.uniform(.05, .95)
            total = mass.sum((-1, -2))
            for request in (np.zeros((4, 3)), np.full((4, 3), 1e-14), total,
                            total * 1.1, total * .5, np.nextafter(total, 0)):
                a, b = mass.copy(), mass.copy()
                old = reference._extract_surface_mixture_batched_inplace(a, request, g, context.mass_resolution_kg)
                new = transfer._extract_surface_mixture_batched_inplace(b, request, g, context.mass_resolution_kg)
                aa, bb = arrays(old), arrays(new)
                differences = {k: float(np.max(np.abs(aa[k].astype(float)-bb[k].astype(float))))
                               for k in aa if not np.array_equal(aa[k], bb[k])}
                if not np.array_equal(a, b):
                    differences['column'] = float(np.max(np.abs(a-b)))
                comparisons.append({'classes': nc, 'voxels': nv, 'differences': differences})
    times = {'baseline': [], 'candidate': []}
    m = bed.voxel_column.mass_kg
    requests = np.full(m.shape[:2], .001)
    for j in range(24):
        pair = [('baseline', reference._extract_surface_mixture_batched_inplace),
                ('candidate', transfer._extract_surface_mixture_batched_inplace)]
        if j % 2:
            pair.reverse()
        for name, fn in pair:
            working = m.copy()
            start = time.perf_counter()
            fn(working, requests, context.geometry, context.mass_resolution_kg)
            times[name].append(time.perf_counter() - start)
    medians = {k: float(np.median(v[4:])) for k, v in times.items()}
    from maple.water import WaterProcessDemand
    z = np.zeros_like(bed.water.mobile_mass_by_cell_class_kg)
    demand = WaterProcessDemand(z, z)
    # Shared public transaction timing; no substitutions or skipped directions.
    public_times = []
    for j in range(24):
        start = time.perf_counter()
        apply_bed_demand(bed, context, demand)
        public_times.append(time.perf_counter() - start)
    report = {'baseline_file_sha256': hashlib.sha256(BASELINE.read_bytes()).hexdigest(),
              'candidate_file_sha256': hashlib.sha256(Path(transfer.__file__).read_bytes()).hexdigest(),
              'cases': len(comparisons), 'exact_cases': sum(not r['differences'] for r in comparisons),
              'differences': [r for r in comparisons if r['differences']], 'samples_s': times,
              'kernel_median_s': medians, 'kernel_speedup': medians['baseline']/medians['candidate'],
              'public_zero_transaction_median_s': float(np.median(public_times[4:])),
              'public_samples_s': public_times,
              'note': 'Input column copying excluded equally from kernel timing; actual public timing includes all work.'}
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('samples_s','public_samples_s')}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    main(args.output)
