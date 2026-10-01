"""Resident GPU extraction parity/timing and allocator high-water comparison.

Run on an otherwise idle GPU. Host/device placement and input copies are
outside timing. Pool growth measures allocations retained by CuPy's pool,
not whole-process GPU memory or full-model peak memory.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import time
from pathlib import Path

import cupy as cp
import numpy as np
from maple.surface.voxels import transfer
from probe_transactions import BASELINE, load_reference

from maple_syrup.case_import import verify_plot1_case
from maple_syrup.sediment_bed import bed_from_case


def run():
    reference = load_reference()
    bed, ctx = bed_from_case(verify_plot1_case('outputs/plot1', allow_maple_source_change=True).case)
    original = cp.asarray(bed.voxel_column.mass_kg)
    request = cp.full(original.shape[:2], .001)
    functions = {'baseline': reference._extract_surface_mixture_batched_inplace,
                 'candidate': transfer._extract_surface_mixture_batched_inplace}
    results = {}
    for name, fn in functions.items():
        mass = original.copy()
        result = fn(mass, request, ctx.geometry, ctx.mass_resolution_kg)
        results[name] = {'mass': cp.asnumpy(mass)}
        # Return fields are device arrays; dataclass fields traversed directly.
        for field in dataclasses.fields(result):
            value = getattr(result, field.name)
            results[name][field.name] = cp.asnumpy(value)
    if not all(np.array_equal(results['baseline'][k], results['candidate'][k]) for k in results['baseline']):
        raise AssertionError('GPU baseline/candidate parity failed')
    del result, mass
    timings = {k: [] for k in functions}
    for i in range(30):
        names = list(functions)
        if i % 2:
            names.reverse()
        for name in names:
            mass = original.copy()
            cp.cuda.get_current_stream().synchronize()
            start = time.perf_counter()
            result = functions[name](mass, request, ctx.geometry, ctx.mass_resolution_kg)
            cp.cuda.get_current_stream().synchronize()
            timings[name].append(time.perf_counter() - start)
            del result, mass
    pool_growth = {}
    pool = cp.get_default_memory_pool()
    for name, fn in functions.items():
        mass = original.copy()
        cp.cuda.get_current_stream().synchronize()
        pool.free_all_blocks()
        before = pool.total_bytes()
        result = fn(mass, request, ctx.geometry, ctx.mass_resolution_kg)
        cp.cuda.get_current_stream().synchronize()
        pool_growth[name] = {'before_bytes': before, 'after_bytes': pool.total_bytes(),
                             'growth_bytes': pool.total_bytes() - before}
        del result, mass
    medians = {k: float(np.median(v)) for k, v in timings.items()}
    return {'scope': __doc__, 'all_saved_fields_exact': True,
            'baseline_source_sha256': hashlib.sha256(BASELINE.read_bytes()).hexdigest(),
            'candidate_source_sha256': hashlib.sha256(Path(transfer.__file__).read_bytes()).hexdigest(),
            'device': cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)['name'].decode(),
            'timing_samples_s': timings, 'warm_median_s': medians,
            'speedup': medians['baseline'] / medians['candidate'], 'pool_allocation_growth': pool_growth}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = run()
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'timing_samples_s'}, indent=2))
