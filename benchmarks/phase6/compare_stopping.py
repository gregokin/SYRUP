"""Compare baseline and stricter stopping policies for the same complete event."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('strict', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    a = json.loads((args.baseline / 'completion_summary.json').read_text())
    b = json.loads((args.strict / 'completion_summary.json').read_text())
    assert a['status'] == b['status'] == 'complete'
    changed = {key for key in a['identity'] if a['identity'][key] != b['identity'][key]}
    assert changed == {'completion_policy'}, changed
    checks = {}
    with np.load(args.baseline / 'final_state.npz') as first, np.load(args.strict / 'final_state.npz') as second:
        for name in ('voxel_mass_kg', 'active_mass_kg', 'available_mass_kg', 'bound_mass_kg',
                     'committed_elevation_m', 'depth_m', 'soil_water_m', 'discharge_m2_s',
                     'mobile_mass_kg', 'sediment_velocity_m_s', 'terminal_deposition_kg'):
            checks[name] = {'bitwise_equal': bool(np.array_equal(first[name], second[name])),
                            'max_abs_difference': float(np.max(np.abs(first[name] - second[name])))}
            np.testing.assert_array_equal(first[name], second[name], err_msg=name)
    np.testing.assert_array_equal(a['sediment_closure']['export_actual_kg'], b['sediment_closure']['export_actual_kg'])
    for s in (a, b):
        assert s['sediment_closure']['closed'] and s['water_budget_before_reset']['closed']
    report = {'schema': 'maple-syrup-stopping-sensitivity/1',
              'source_sha256': a['identity']['syrup_source_sha256'],
              'baseline_policy': a['identity']['completion_policy'], 'strict_policy': b['identity']['completion_policy'],
              'baseline_time_s': a['time_s'], 'strict_time_s': b['time_s'],
              'baseline_quiet': a['quiet_progress'], 'strict_quiet': b['quiet_progress'],
              'final_array_comparisons': checks, 'export_by_class_bitwise_equal': True,
              'baseline_water_budget': a['water_budget_before_reset'], 'strict_water_budget': b['water_budget_before_reset'],
              'baseline_soil_removed_m3': a['reset_accounting']['soil_removed_m3'],
              'strict_soil_removed_m3': b['reset_accounting']['soil_removed_m3'],
              'interpretation': 'Case-specific stopping robustness only. Additional recession time permits more physical drainage before the explicit dry reset; no general grid/backend qualification.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: report[k] for k in ('baseline_time_s', 'strict_time_s', 'baseline_quiet', 'strict_quiet',
                                          'export_by_class_bitwise_equal', 'baseline_soil_removed_m3', 'strict_soil_removed_m3')}, indent=2))


if __name__ == '__main__':
    main()
