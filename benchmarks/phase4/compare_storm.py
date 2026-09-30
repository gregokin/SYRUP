"""Summarize the recorded Phase 4 controlled Plot 1 study; run from repo root."""
import hashlib
import json
import re
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    out = Path('docs/phase4')
    production = Path('outputs/phase4_storm')
    reference = Path('outputs/phase4_reference_final')
    result = {'scope': 'Controlled water-only original-routine comparison; not full MAHLERAN application',
              'runs': {}, 'source_files': {}}
    for name in ['numba_dt1', 'numba_dt0p5', 'numba_dt0p25', 'array_dt1', 'numba_rain_end']:
        path = production / name / 'storm_summary.json'
        s = read(path)
        stderr = Path('agent_handoffs/tasks/phase4c_storm') / (name + '.stderr')
        match = re.search(r'Maximum resident set size \(kbytes\): (\d+)', stderr.read_text())
        row = {k: s[k] for k in ['budget', 'final_state', 'timings', 'time']}
        row['process_peak_rss_kib'] = int(match[1]) if match else None
        row['outputs_sha256'] = s['outputs']
        result['runs'][name] = row
        result['source_files'][str(path)] = digest(path)
    for name in ['dt1', 'dt0p5', 'dt0p25', 'rain_end']:
        path = reference / name / 'reference_summary.json'
        r = read(path)
        result['runs']['fortran_' + name] = {k: r[k] for k in [
            'final', 'peak_outlet_m3_s', 'peak_time_s', 'budget_minus_stale_minus_closure_m3',
            'output_sha256', 'input_sha256']}
        result['source_files'][str(path)] = digest(path)
        if name != 'rain_end':
            a = result['runs']['numba_' + name]['budget']['export_m3']
            result['runs']['numba_' + name]['export_difference_percent_of_fortran'] = (
                100 * (a / r['final']['export_m3'] - 1))
    result['identity'] = {k: r[k] for k in ['case_identity_sha256', 'graph_sha256',
        'rainfall_sha256', 'source_sha256', 'driver_sha256', 'writer_sha256', 'helper_sha256',
        'compiler_version', 'departures_from_application']}
    result['identity']['maple_digest'] = s['provenance']['source_stability']['after']['maple']
    result['identity']['syrup_digest'] = s['provenance']['source_stability']['after']['maple_syrup']
    parity = {}
    for filename in ['final_water.npz', 'hydrograph.npz']:
        with np.load(production / 'numba_dt1' / filename) as a, np.load(production / 'array_dt1' / filename) as b:
            assert a.files == b.files
            parity[filename] = {k: bool(np.array_equal(a[k], b[k])) for k in a.files}
            assert all(parity[filename].values()), parity
    result['bitwise_array_numba_parity'] = parity
    f = np.loadtxt(reference / 'rain_end' / 'final.dat')
    fields = {name: f[:, col].reshape(60, 20)[::-1] for name, col in
              [('depth_m', 2), ('soil_water_m', 3), ('discharge_m2_s', 4)]}
    fields['velocity_m_s'] = np.divide(fields['discharge_m2_s'], fields['depth_m'],
        out=np.zeros((60, 20)), where=fields['depth_m'] > 0)
    result['rain_end_spatial_difference'] = {}
    with np.load(production / 'numba_rain_end' / 'final_water.npz') as a:
        for key, value in fields.items():
            difference = a[key] - value
            result['rain_end_spatial_difference'][key] = {
                'max_abs': float(np.abs(difference).max()),
                'rmse': float(np.sqrt(np.mean(difference**2))),
                'reference_max': float(value.max())}
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True, layout='constrained')
    with np.load(production / 'numba_dt1' / 'hydrograph.npz') as a:
        axes[0].plot(a['t_s'] / 60, a['outlet_discharge_m3_s'] * 1000,
                     label='SYRUP, 1 s', linewidth=2)
        axes[1].plot(a['t_s'] / 60, a['row_water_residual_m3'] * 1000, label='SYRUP, 1 s')
    for name, label in [('dt1', 'Original routines, 1 s'), ('dt0p25', 'Original routines, 0.25 s')]:
        r = read(reference / name / 'reference_summary.json')
        a = np.loadtxt(reference / name / 'hydrograph.dat', skiprows=1, max_rows=r['n_steps'])
        axes[0].plot(a[:, 0] / 60, a[:, 1] * 1000, '--', label=label, linewidth=1)
        axes[1].plot(a[:, 0] / 60, a[:, 7] * 1000, '--', label=label)
    axes[0].set(ylabel='Outlet discharge (L/s)', title='Plot 1: controlled water-only storm comparison')
    axes[1].set(xlabel='Time (minutes)', ylabel='Water-balance excess (L)')
    for ax in axes:
        ax.axvline(27, color='gray', linestyle=':', label='Rain ends')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    fig.savefig(out / 'storm_comparison.svg')
    plt.close(fig)
    (out / 'storm_comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    print('Bitwise array/Numba parity: all final grids and hydrograph arrays.')
    for name, row in result['runs'].items():
        if 'budget' in row:
            print(name, row['budget']['export_m3'], row['timings']['step_loop']['wall_s'],
                  row['process_peak_rss_kib'], row.get('export_difference_percent_of_fortran'))
    print(json.dumps(result['rain_end_spatial_difference'], indent=2))


if __name__ == '__main__':
    main()
