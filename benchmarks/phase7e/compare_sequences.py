"""Bitwise comparison of two transaction_sequence.py outputs (accepted vs candidate)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('baseline', type=Path)
p.add_argument('candidate', type=Path)
p.add_argument('--output', type=Path)
a = p.parse_args()
rows = []
with np.load(a.baseline) as old, np.load(a.candidate) as new:
    if set(old.files) != set(new.files):
        raise SystemExit(f'field sets differ: {sorted(set(old.files) ^ set(new.files))[:10]}')
    for key in sorted(old.files):
        x, y = old[key], new[key]
        exact = x.shape == y.shape and x.dtype == y.dtype and np.array_equal(x, y)
        row = {'field': key, 'exact': bool(exact)}
        if not exact and x.dtype.kind == 'f' and x.shape == y.shape:
            row['max_abs_difference'] = float(np.max(np.abs(x - y)))
        rows.append(row)
physical = [r for r in rows if not r['field'].endswith('.error')]
errors_match = True
with np.load(a.baseline) as old, np.load(a.candidate) as new:
    for key in old.files:
        if key.endswith('.error'):
            for arr in (old[key], new[key]):
                msg = arr.tobytes().decode()
                errors_match &= msg.startswith('ValueError(') and msg[12:].startswith('deposit_surface_mixture: insufficient allocated voxel capacity')
report = {'physical_fields': len(physical), 'physical_fields_exact': all(r['exact'] for r in physical),
          'error_categories_match': bool(errors_match), 'baseline' : str(a.baseline), 'candidate': str(a.candidate), 'fields': len(rows),
          'all_exact': all(r['exact'] for r in rows), 'differences': [r for r in rows if not r['exact']],
          'provenance': {k: json.loads(Path(str(v)).with_suffix('.json').read_text()) for k, v in (('baseline', a.baseline), ('candidate', a.candidate))}}
if a.output:
    a.output.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({k: v for k, v in report.items() if k != 'differences'}, indent=2))
print('n_differences', len(report['differences']), report['differences'][:5])
