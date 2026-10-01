"""Compare complete-event outputs, including every saved continuation array."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def compare(full: Path, resumed: Path) -> dict:
    a = json.loads((full / 'completion_summary.json').read_text())
    b = json.loads((resumed / 'completion_summary.json').read_text())
    for name in ('status', 'identity', 'time_s', 'n_accepted_steps', 'n_rejected_attempts',
                 'n_commits', 'quiet_progress', 'water_budget_before_reset', 'sediment_closure', 'reset_accounting'):
        if a[name] != b[name]:
            raise AssertionError(f'restart differs: {name}')
    if a['status'] != 'complete':
        raise AssertionError('comparison requires completed dry handoffs')
    counts = {}
    for filename in ('final_state.npz', 'hydrograph.npz', 'handoff/continuation.npz'):
        with np.load(full / filename, allow_pickle=False) as first, np.load(resumed / filename, allow_pickle=False) as second:
            if set(first.files) != set(second.files):
                raise AssertionError(f'array fields differ: {filename}')
            for name in first.files:
                np.testing.assert_array_equal(first[name], second[name], err_msg=f'{filename}:{name}')
            counts[filename] = len(first.files)
    cp = Path(b['resumed_from'])
    meta = json.loads((cp / 'checkpoint.json').read_text())
    root = meta['payload']['fields']
    state = root['result']['fields']['state']['state']
    with np.load(cp / 'continuation.npz', allow_pickle=False) as data:
        water = state['bed']['bed']['water']['fields']
        mobile = data[water['mobile_mass_by_cell_class_kg']['array']]
        depth = data[water['depth_m']['array']]
        ledger = state['bed']['bed']['ledger']['fields']
        pending = data[ledger['pending_bed_mass_change_kg']['array']]
        checkpoint_info = {'time_s': state['storm']['fields']['t_s'],
                           'mobile_kg': float(mobile.sum()), 'max_depth_m': float(depth.max()),
                           'pending_bed_abs_kg': float(np.abs(pending).sum()),
                           'bytes': sum(p.stat().st_size for p in cp.iterdir() if p.is_file())}
    return {'schema': 'maple-syrup-phase6-validation/1',
            'full_path': str(full), 'resumed_path': str(resumed),
            'source_sha256': a['identity']['syrup_source_sha256'],
            'maple_source_sha256': a['identity']['maple_source_sha256'],
            'bitwise_equal_array_counts': counts, 'checkpoint': checkpoint_info,
            'time_s': a['time_s'], 'steps': a['n_accepted_steps'], 'commits': a['n_commits'],
            'quiet_progress': a['quiet_progress'], 'water_budget': a['water_budget_before_reset'],
            'sediment_closure': a['sediment_closure'], 'reset_accounting': a['reset_accounting'],
            'observational_timings_full': a['timings'], 'observational_timings_resumed': b['timings'],
            'performance_note': 'Runs overlap with regression/each other; no controlled speed comparison.',
            'gpu': 'not exercised', 'claude_review': 'pending usage-limit reset'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('full', type=Path)
    p.add_argument('resumed', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    report = compare(args.full, args.resumed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: report[k] for k in ('time_s', 'steps', 'commits', 'bitwise_equal_array_counts', 'checkpoint')}, indent=2))


if __name__ == '__main__':
    main()
