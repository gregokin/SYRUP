"""Warm, interleaved old/new compiled-kernel timings on identical inputs.

Supply an archived pre-optimization characteristic_numba.py. This is a CPU
kernel benchmark, not a full event or a GPU measurement. Run without a storm
benchmark alongside it. Output equality is checked before timing.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import time
from pathlib import Path

import numba
import numpy as np

from maple_syrup.characteristic_numba import compiled_substep
from maple_syrup.characteristic_transport import NARROW_SPREAD


def inputs(n, nb, occupied, seed=734):
    rng = np.random.default_rng(seed)
    nc, dx, dt = 6, 0.5, 1.0
    w = np.where(rng.random((n, nc, nb)) < occupied,
                 10.0 ** rng.uniform(-18, -2, (n, nc, nb)), 0.0)
    x = (np.arange(nb)[None, None, :] + rng.random(w.shape)) * (dx / nb)
    x[w == 0] = 0
    p = np.where(rng.random((n, nc)) < occupied, rng.random((n, nc)) * 1e-3, 0.0)
    v = rng.uniform(0, 0.25, (n, nc))
    r = 10.0 ** rng.uniform(-2, 4, (n, nc))
    r[rng.random(r.shape) < .2] = 0
    settle = rng.random(r.shape) < .15
    receiver = np.minimum(np.arange(n, dtype=np.int64) + 1, n - 1)
    outlet = np.arange(n) == n - 1
    return w, x, p, v, r, settle, receiver, outlet, dx, dt, nb, NARROW_SPREAD


def run(baseline_module, repeats):
    import maple_syrup.characteristic_numba as current
    current_path = Path(current.__file__)
    candidate_hash = hashlib.sha256(current_path.read_bytes()).hexdigest()
    spec = importlib.util.spec_from_file_location('archived_characteristic', baseline_module)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    old_fn = numba.njit(fastmath=False, nogil=True)(old._substep)
    new_fn = compiled_substep()
    start = time.perf_counter()
    old_fn(*inputs(4, 4, .3))
    old_compile = time.perf_counter() - start
    start = time.perf_counter()
    new_fn(*inputs(4, 4, .3))
    new_compile = time.perf_counter() - start
    rows = []
    for n, nb, density in [(1200, 32, 0), (1200, 32, .03), (1200, 32, 1),
                           (1200, 128, .03), (4800, 32, .03)]:
        args = inputs(n, nb, density)
        before, after = old_fn(*args), new_fn(*args)
        exact = all(np.array_equal(a, b) for a, b in zip(before, after, strict=True))
        if not exact:
            raise AssertionError('Compiled reference parity failed')
        timings = {'baseline': [], 'candidate': []}
        for j in range(repeats):
            pairs = [('baseline', old_fn), ('candidate', new_fn)]
            if j % 2:
                pairs.reverse()
            for name, fn in pairs:
                start = time.perf_counter()
                result = fn(*args)
                timings[name].append(time.perf_counter() - start)
                del result
        medians = {k: float(np.median(v)) for k, v in timings.items()}
        rows.append({'cells': n, 'classes': 6, 'bins': nb, 'occupancy_probability': density,
                     'all_nine_outputs_bitwise_equal': exact, 'warm_median_s': medians,
                     'speedup': medians['baseline'] / medians['candidate'], 'samples_s': timings,
                     'warm_median_s_by_order': {
                         'baseline_first': float(np.median(timings['baseline'][::2])),
                         'baseline_second': float(np.median(timings['baseline'][1::2])),
                         'candidate_first': float(np.median(timings['candidate'][1::2])),
                         'candidate_second': float(np.median(timings['candidate'][::2]))},
                     'baseline_kernel_array_payload_bytes': 8 * n * 6 * (8 * nb + 4 * (nb + 1) + 7),
                     'candidate_kernel_array_payload_bytes': (8 * n * 6 * (2 * nb + 7) if density == 0 else
                         8 * (n * 6 * (6 * nb + 7) + 4 * (int(np.count_nonzero(args[0])) + int(np.count_nonzero(args[2])))))})
    if hashlib.sha256(current_path.read_bytes()).hexdigest() != candidate_hash:
        raise RuntimeError('source changed during timing')
    return {'scope': __doc__, 'baseline_module': str(baseline_module),
            'baseline_sha256': hashlib.sha256(baseline_module.read_bytes()).hexdigest(),
            'candidate_sha256': candidate_hash,
            'compile_and_first_call_s': {'baseline': old_compile, 'candidate': new_compile},
            'rows': rows, 'memory_note': 'Source-derived array payload including outputs; excludes inputs, allocator, JIT and wrapper. Not measured peak RSS.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-module', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--repeats', type=int, default=15)
    args = p.parse_args()
    result = run(args.baseline_module, args.repeats)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
