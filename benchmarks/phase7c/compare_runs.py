"""Compare saved numerical outputs and timings of two matched Plot1 runs.

This reports differences rather than imposing a new physical tolerance.
An optimization is accepted only with separate conservation and parity evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def compare(baseline: Path, candidate: Path) -> dict:
    rows = []
    for name in ('final_state.npz', 'forcing.npz', 'hydrograph.npz', 'peak_outlet_snapshot.npz'):
        with np.load(baseline / name, allow_pickle=False) as old, np.load(candidate / name, allow_pickle=False) as new:
            if set(old.files) != set(new.files):
                raise ValueError(f'{name}: saved field names differ')
            for key in old.files:
                a, b = old[key], new[key]
                if a.dtype.kind not in 'biufc' or b.dtype.kind not in 'biufc':
                    continue
                if a.shape != b.shape or a.dtype != b.dtype:
                    raise ValueError(f'{name}:{key}: shape or dtype differs')
                exact = np.array_equal(a, b)
                row = {'file': name, 'field': key, 'exact': exact}
                if not exact:
                    delta = np.abs(a.astype(np.float64) - b.astype(np.float64))
                    row.update(max_abs_difference=float(delta.max(initial=0)),
                               baseline_max_abs=float(np.abs(a).max(initial=0)),
                               finite=bool(np.isfinite(delta).all()))
                rows.append(row)
    reports = []
    for root in (baseline, candidate):
        summary_path = root / 'benchmark_summary.json'
        summary = json.loads(summary_path.read_text())
        timing = json.loads((root / 'component_times.json').read_text())
        reports.append({'path': str(root),
                        'summary_sha256': hashlib.sha256(summary_path.read_bytes()).hexdigest(),
                        'provenance': summary['provenance'],
                        'performance': summary['performance'],
                        'component_timers': timing['component_timers'],
                        'sediment_closure': summary['sediment']['closure'],
                        'sediment_totals': summary['sediment']['totals'],
                        'water_budget': summary['budget']})
    before, after = (r['performance']['step_loop']['wall_s'] for r in reports)
    return {'runs': reports, 'numeric_fields_compared': len(rows),
            'all_numeric_fields_exact': all(r['exact'] for r in rows),
            'differences': [r for r in rows if not r['exact']],
            'loop_speedup': before / after, 'loop_time_reduction_fraction': 1 - after / before,
            'note': 'Sequential same-machine observations; compilation is reported separately. '
                    'No numerical tolerance is relaxed by this reporting script.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.baseline, args.candidate)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'runs'}, indent=2))
